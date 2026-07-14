"""
tools/annotate/ops.py

打标（annotate）与标签覆盖（reannotate）共用的底层能力，与「具体类别」和
「具体模型后端」都无关：

- 框几何：clip_box / sanitize_boxes / classify_box_by_ratio /
  split_boxes_by_ratio / subtract_box_regions / collect_blackout_regions /
  concat_boxes；
- 打码：blackout_region（纯黑）/ mosaic_region（马赛克）；
- LabelMe 写出 / 读取：build_labelme_shapes / rewrite_labelme_dict /
  extract_existing_label_boxes。

各检测器后端（ONNX / SAM / YOLO / DETR）见 ``tools/annotate/backends/``；
本模块只做「拿到框之后」的几何、打码与 LabelMe 落盘，供编排脚本复用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from tools.core import load_labelme


# =============================================================================
# 1. 框几何
# =============================================================================


def clip_box(box: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """把框裁剪到图像范围内。"""
    image_h, image_w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    clipped = np.array(
        [
            np.clip(x1, 0, image_w),
            np.clip(y1, 0, image_h),
            np.clip(x2, 0, image_w),
            np.clip(y2, 0, image_h),
        ],
        dtype=np.float32,
    )
    return clipped


def sanitize_boxes(
    boxes: np.ndarray, image_shape: tuple[int, int, int]
) -> np.ndarray:
    """裁剪并过滤无效框。"""
    sanitized: list[np.ndarray] = []
    for box in boxes:
        clipped = clip_box(box, image_shape)
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        sanitized.append(clipped)
    if not sanitized:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(sanitized, dtype=np.float32)


def classify_box_by_ratio(
    box: np.ndarray,
    image_shape: tuple[int, int, int],
    min_ratio: float,
) -> bool:
    """按面积占比判断框是否达到保留阈值。"""
    image_h, image_w = image_shape[:2]
    image_area = float(image_w * image_h)
    if image_area <= 0:
        return False
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return (box_area / image_area) >= min_ratio


def split_boxes_by_ratio(
    boxes: np.ndarray,
    image_shape: tuple[int, int, int],
    min_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按面积占比分成保留框与删除框。"""
    kept: list[np.ndarray] = []
    removed: list[np.ndarray] = []
    for box in boxes:
        if classify_box_by_ratio(box, image_shape, min_ratio):
            kept.append(box)
        else:
            removed.append(box)
    kept_array = np.array(kept, dtype=np.float32) if kept else np.empty((0, 4), dtype=np.float32)
    removed_array = np.array(removed, dtype=np.float32) if removed else np.empty((0, 4), dtype=np.float32)
    return kept_array, removed_array


def subtract_box_regions(box: np.ndarray, keep_boxes: np.ndarray) -> np.ndarray:
    """从待删除小框中减去所有保留大框重叠区域。"""
    pieces = [clip_box(box, (10**9, 10**9, 3))]
    for keep_box in keep_boxes:
        next_pieces: list[np.ndarray] = []
        for piece in pieces:
            x1, y1, x2, y2 = [float(v) for v in piece.tolist()]
            kx1, ky1, kx2, ky2 = [float(v) for v in keep_box.tolist()]
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                next_pieces.append(piece)
                continue
            if y1 < iy1:
                next_pieces.append(np.array([x1, y1, x2, iy1], dtype=np.float32))
            if iy2 < y2:
                next_pieces.append(np.array([x1, iy2, x2, y2], dtype=np.float32))
            if x1 < ix1:
                next_pieces.append(np.array([x1, iy1, ix1, iy2], dtype=np.float32))
            if ix2 < x2:
                next_pieces.append(np.array([ix2, iy1, x2, iy2], dtype=np.float32))
        pieces = [piece for piece in next_pieces if piece[2] > piece[0] and piece[3] > piece[1]]
        if not pieces:
            return np.empty((0, 4), dtype=np.float32)
    return np.array(pieces, dtype=np.float32) if pieces else np.empty((0, 4), dtype=np.float32)


def collect_blackout_regions(
    removed_boxes: np.ndarray,
    kept_boxes: np.ndarray,
) -> np.ndarray:
    """收集所有待删除小框的非重叠残余区域。"""
    regions: list[np.ndarray] = []
    for removed_box in removed_boxes:
        for piece in subtract_box_regions(removed_box, kept_boxes):
            regions.append(piece)
    return np.array(regions, dtype=np.float32) if regions else np.empty((0, 4), dtype=np.float32)


def concat_boxes(box_groups: Iterable[np.ndarray]) -> np.ndarray:
    """把多组框（允许空）拼接成一个数组。"""
    parts = [g for g in box_groups if len(g)]
    if not parts:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


# 默认马赛克块大小（像素），越大越糊。
DEFAULT_MOSAIC_BLOCK = 16


# =============================================================================
# 2. 打码
# =============================================================================


def blackout_region(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """把框内区域直接填充为纯黑。"""
    clipped = clip_box(box, image.shape)
    x1, y1, x2, y2 = [int(round(v)) for v in clipped.tolist()]
    if x2 <= x1 or y2 <= y1:
        return image
    image[y1:y2, x1:x2] = 0
    return image


def mosaic_region(
    image: np.ndarray,
    box: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """把框内区域打成马赛克（先缩小再最近邻放大，保留边界硬块感）。

    只处理框内像素；框外与重叠部分由调用方通过 ``box`` 预先扣减好，
    本函数不关心重叠逻辑。
    """
    clipped = clip_box(box, image.shape)
    x1, y1, x2, y2 = (int(round(v)) for v in clipped.tolist())
    if x2 <= x1 or y2 <= y1:
        return image
    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return image
    # 缩小到 (w//block, h//block)，再放大回原尺寸，得到马赛克块。
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    return image


def apply_blackout(
    image: np.ndarray,
    removed_boxes: np.ndarray,
    kept_boxes: np.ndarray,
    *,
    use_mosaic: bool = False,
    mosaic_block: int = DEFAULT_MOSAIC_BLOCK,
    apply: bool = True,
) -> tuple[np.ndarray, int]:
    """对小框区域打码（马赛克或纯黑）后返回图像与打码区域数。

    仅对「待删除小框」中不与「保留大框」重叠的残余区域打码（重叠保护由
    ``collect_blackout_regions`` 处理）。打码策略由 ``use_mosaic`` 决定：
    ``True`` 用 ``mosaic_region``，``False`` 用 ``blackout_region``。

    ``apply=False`` 时只计算并返回打码区域数、不修改图像（供预览统计）。
    """
    regions = collect_blackout_regions(removed_boxes, kept_boxes)
    if apply:
        for box in regions:
            if use_mosaic:
                image = mosaic_region(image, box, mosaic_block)
            else:
                image = blackout_region(image, box)
    return image, len(regions)


# =============================================================================
# 3. LabelMe 写出 / 读取
# =============================================================================


def build_labelme_shapes(boxes: np.ndarray, labels: list[str]) -> list[dict]:
    """把矩形框与标签转成 LabelMe shapes。"""
    shapes: list[dict] = []
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        shapes.append({
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
        })
    return shapes


def rewrite_labelme_dict(
    json_path: Path,
    boxes: np.ndarray,
    labels: list[str],
    image_shape: tuple[int, int, int],
    image_name: str,
) -> dict:
    """用新的框整体覆盖 LabelMe。"""
    if json_path.exists():
        data = load_labelme(json_path)
    else:
        data = {}
    image_h, image_w = image_shape[:2]
    data["version"] = data.get("version", "5.3.1")
    data["flags"] = data.get("flags", {})
    data["shapes"] = build_labelme_shapes(boxes, labels)
    data["imagePath"] = image_name
    data["imageData"] = None
    data["imageHeight"] = int(image_h)
    data["imageWidth"] = int(image_w)
    return data


def extract_existing_label_boxes(json_path: Path, label: str) -> np.ndarray:
    """从现有 LabelMe JSON 中提取指定标签的矩形框。"""
    if not json_path.exists():
        return np.empty((0, 4), dtype=np.float32)
    data = load_labelme(json_path)
    boxes: list[list[float]] = []
    for shape in data.get("shapes", []):
        if shape.get("label") != label:
            continue
        if shape.get("shape_type") != "rectangle":
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        (x1, y1), (x2, y2) = points[:2]
        boxes.append([
            float(min(x1, x2)),
            float(min(y1, y2)),
            float(max(x1, x2)),
            float(max(y1, y2)),
        ])
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)
