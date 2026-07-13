"""
tools/annotate/reannotate_face_hand_onnx.py
使用两个 ONNX 模型重标注 behavior 批次中的 face / hand，
并对过小人脸先做马赛克再删除对应框。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import find_json_for_image, list_images, load_labelme, save_labelme


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_FACE_MODEL = r"weight\yolov5s-lmk.onnx"
DEFAULT_HAND_MODEL = r"weight\DNTC_Ariya_Gesture_HandDetect_20260205_640x640_fp32.onnx"
DEFAULT_INPUT_ROOT = r"datasets\behavior"
DEFAULT_START_BATCH = 23
DEFAULT_END_BATCH = 37
DEFAULT_MIN_FACE_RATIO = 0.05
DEFAULT_FACE_CONF = 0.25
DEFAULT_HAND_CONF = 0.25
DEFAULT_FACE_IOU = 0.45
DEFAULT_HAND_IOU = 0.45
DEFAULT_MOSAIC_SIZE = 10
DEFAULT_DRY_RUN = True
DEFAULT_FACE_LABEL = "face"
DEFAULT_HAND_LABEL = "hand"


# =============================================================================
# 2. 数据结构
# =============================================================================


@dataclass
class ProcessResult:
    image: np.ndarray
    boxes: np.ndarray
    labels: list[str]
    mosaic_count: int
    face_count: int
    hand_count: int


# =============================================================================
# 3. 纯函数
# =============================================================================


def classify_face_box(
    box: np.ndarray,
    image_shape: tuple[int, int, int],
    min_ratio: float,
) -> bool:
    """判断人脸框面积占比是否达到保留阈值。"""
    image_h, image_w = image_shape[:2]
    image_area = float(image_w * image_h)
    if image_area <= 0:
        return False
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return (box_area / image_area) >= min_ratio


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


def mosaic_region(
    image: np.ndarray,
    box: np.ndarray,
    mosaic_size: int = DEFAULT_MOSAIC_SIZE,
) -> np.ndarray:
    """对框内区域做马赛克。"""
    clipped = clip_box(box, image.shape)
    x1, y1, x2, y2 = [int(round(v)) for v in clipped.tolist()]
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    small_w = max(1, roi.shape[1] // mosaic_size)
    small_h = max(1, roi.shape[0] // mosaic_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(
        small,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    image[y1:y2, x1:x2] = mosaic
    return image


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
    """用新的人脸与手部框整体覆盖 LabelMe。"""
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


# =============================================================================
# 4. 文件发现
# =============================================================================


def iter_batch_dirs(root: Path, start_batch: int, end_batch: int) -> list[Path]:
    """收集指定 batch 范围内存在的目录。"""
    batch_dirs: list[Path] = []
    for batch_id in range(start_batch, end_batch + 1):
        batch_dir = root / f"{batch_id:04d}"
        if batch_dir.is_dir():
            batch_dirs.append(batch_dir)
    return batch_dirs


def iter_image_files(root: Path, start_batch: int, end_batch: int) -> list[Path]:
    """收集指定 batch 范围内的全部图片。"""
    image_paths: list[Path] = []
    for batch_dir in iter_batch_dirs(root, start_batch, end_batch):
        image_paths.extend(list_images(batch_dir, recursive=True))
    return image_paths


# =============================================================================
# 5. ONNX 适配
# =============================================================================


def resolve_input_size(shape: list[int | str | None]) -> tuple[int, int]:
    """从 ONNX 输入 shape 里解析输入宽高。"""
    if len(shape) >= 4:
        input_h = int(shape[2] or 640)
        input_w = int(shape[3] or 640)
        return input_w, input_h
    return 640, 640


def preprocess_image(
    image: np.ndarray,
    input_size: tuple[int, int],
    *,
    normalize: bool,
) -> np.ndarray:
    """按模型输入尺寸直接缩放，并按模型要求决定是否归一化。"""
    input_w, input_h = input_size
    resized = cv2.resize(image, (input_w, input_h))
    resized = resized.astype(np.float32)
    if normalize:
        resized /= 255.0
    tensor = np.transpose(resized, (2, 0, 1))[None, ...]
    return tensor


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """把中心点格式的 xywh 转成 xyxy。"""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    centers_x = boxes[:, 0]
    centers_y = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    converted = np.stack(
        [
            centers_x - widths / 2.0,
            centers_y - heights / 2.0,
            centers_x + widths / 2.0,
            centers_y + heights / 2.0,
        ],
        axis=1,
    )
    return converted.astype(np.float32)


def scale_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int, int],
    input_size: tuple[int, int],
) -> np.ndarray:
    """把输入尺度上的框映射回原图尺度。"""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    image_h, image_w = image_shape[:2]
    input_w, input_h = input_size
    scaled = boxes.copy().astype(np.float32)
    scaled[:, [0, 2]] *= image_w / float(input_w)
    scaled[:, [1, 3]] *= image_h / float(input_h)
    return scaled


def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """对检测框做 NMS，返回保留索引。"""
    if len(boxes) == 0:
        return np.empty((0,), dtype=int)
    nms_boxes = [
        [
            float(box[0]),
            float(box[1]),
            float(box[2] - box[0]),
            float(box[3] - box[1]),
        ]
        for box in boxes
    ]
    kept = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=scores.astype(float).tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if len(kept) == 0:
        return np.empty((0,), dtype=int)
    kept_indices = np.array(kept).reshape(-1).astype(int)
    return kept_indices


def decode_face_outputs(
    outputs: list[np.ndarray],
    image_shape: tuple[int, int, int],
    input_size: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """解码人脸模型输出。"""
    rows = outputs[0][0]
    scores = rows[:, 4] * rows[:, 15]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32)
    boxes = xywh_to_xyxy(rows[keep, :4])
    scores = scores[keep]
    boxes = scale_boxes(boxes, image_shape, input_size)
    kept_indices = apply_nms(boxes, scores, iou_threshold)
    return boxes[kept_indices]


def decode_hand_outputs(
    outputs: list[np.ndarray],
    image_shape: tuple[int, int, int],
    input_size: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """解码手部模型输出。"""
    rows = outputs[0][0].transpose(1, 0)
    scores = rows[:, 4]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32)
    boxes = xywh_to_xyxy(rows[keep, :4])
    scores = scores[keep]
    boxes = scale_boxes(boxes, image_shape, input_size)
    kept_indices = apply_nms(boxes, scores, iou_threshold)
    return boxes[kept_indices]


class OnnxDetector:
    """ONNX 检测器。"""

    def __init__(self, model_path: Path, label: str, conf: float) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.label = label
        self.conf = conf
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = resolve_input_size(self.session.get_inputs()[0].shape)
        self.normalize = label == DEFAULT_FACE_LABEL
        self.iou_threshold = (
            DEFAULT_FACE_IOU if label == DEFAULT_FACE_LABEL else DEFAULT_HAND_IOU
        )

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """执行 ONNX 推理并返回框与标签。"""
        tensor = preprocess_image(
            image,
            self.input_size,
            normalize=self.normalize,
        )
        outputs = self.session.run(None, {self.input_name: tensor})
        if self.label == DEFAULT_FACE_LABEL:
            boxes = decode_face_outputs(
                outputs=outputs,
                image_shape=image.shape,
                input_size=self.input_size,
                conf_threshold=self.conf,
                iou_threshold=self.iou_threshold,
            )
        else:
            boxes = decode_hand_outputs(
                outputs=outputs,
                image_shape=image.shape,
                input_size=self.input_size,
                conf_threshold=self.conf,
                iou_threshold=self.iou_threshold,
            )
        labels = [self.label] * len(boxes)
        return boxes, labels


# =============================================================================
# 6. 单图处理
# =============================================================================


def combine_boxes(face_boxes: list[np.ndarray], hand_boxes: np.ndarray) -> np.ndarray:
    """把保留的人脸框与手部框合并成统一数组。"""
    parts: list[np.ndarray] = []
    if face_boxes:
        parts.append(np.array(face_boxes, dtype=np.float32))
    if len(hand_boxes):
        parts.append(hand_boxes.astype(np.float32))
    if not parts:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def sanitize_boxes(boxes: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
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


def process_image_file(
    image_path: Path,
    face_detector: OnnxDetector,
    hand_detector: OnnxDetector,
    min_face_ratio: float,
    dry_run: bool,
    mosaic_size: int = DEFAULT_MOSAIC_SIZE,
) -> ProcessResult:
    """对单张图片执行 face / hand 重标注。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")

    face_boxes, _ = face_detector.predict(image)
    hand_boxes, _ = hand_detector.predict(image)
    hand_boxes = sanitize_boxes(hand_boxes, image.shape)

    kept_face_boxes: list[np.ndarray] = []
    mosaic_face_boxes: list[np.ndarray] = []
    for face_box in face_boxes:
        clipped_box = clip_box(face_box, image.shape)
        if classify_face_box(clipped_box, image.shape, min_face_ratio):
            kept_face_boxes.append(clipped_box)
        else:
            mosaic_face_boxes.append(clipped_box)
            if not dry_run:
                image = mosaic_region(image, clipped_box, mosaic_size=mosaic_size)

    final_boxes = combine_boxes(kept_face_boxes, hand_boxes)
    final_labels = [DEFAULT_FACE_LABEL] * len(kept_face_boxes)
    final_labels.extend([DEFAULT_HAND_LABEL] * len(hand_boxes))
    return ProcessResult(
        image=image,
        boxes=final_boxes,
        labels=final_labels,
        mosaic_count=len(mosaic_face_boxes),
        face_count=len(kept_face_boxes),
        hand_count=len(hand_boxes),
    )


# =============================================================================
# 7. 运行逻辑
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(
        description="用两个 ONNX 模型重标注 behavior 批次中的 face / hand。",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path(DEFAULT_FACE_MODEL),
        help=f"人脸 ONNX 模型路径（默认：{DEFAULT_FACE_MODEL}）。",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path(DEFAULT_HAND_MODEL),
        help=f"手部 ONNX 模型路径（默认：{DEFAULT_HAND_MODEL}）。",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(DEFAULT_INPUT_ROOT),
        help=f"behavior 根目录（默认：{DEFAULT_INPUT_ROOT}）。",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=DEFAULT_START_BATCH,
        help=f"起始 batch 编号（默认：{DEFAULT_START_BATCH}）。",
    )
    parser.add_argument(
        "--end-batch",
        type=int,
        default=DEFAULT_END_BATCH,
        help=f"结束 batch 编号（默认：{DEFAULT_END_BATCH}）。",
    )
    parser.add_argument(
        "--min-face-ratio",
        type=float,
        default=DEFAULT_MIN_FACE_RATIO,
        help=f"保留人脸的最小面积占比（默认：{DEFAULT_MIN_FACE_RATIO}）。",
    )
    parser.add_argument(
        "--face-conf",
        type=float,
        default=DEFAULT_FACE_CONF,
        help=f"人脸置信度阈值（默认：{DEFAULT_FACE_CONF}）。",
    )
    parser.add_argument(
        "--hand-conf",
        type=float,
        default=DEFAULT_HAND_CONF,
        help=f"手部置信度阈值（默认：{DEFAULT_HAND_CONF}）。",
    )
    parser.add_argument(
        "--mosaic-size",
        type=int,
        default=DEFAULT_MOSAIC_SIZE,
        help=f"马赛克缩小倍数（默认：{DEFAULT_MOSAIC_SIZE}）。",
    )
    parser.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="真正写回图片与 JSON；默认仅 dry-run 统计。",
    )
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


def run(args: argparse.Namespace) -> int:
    """执行批量重标注。"""
    image_paths = iter_image_files(args.input_root, args.start_batch, args.end_batch)
    if not image_paths:
        print("[错误] 指定 batch 范围内没有可处理图片。", file=sys.stderr)
        return 1

    face_detector = OnnxDetector(args.face_model, DEFAULT_FACE_LABEL, args.face_conf)
    hand_detector = OnnxDetector(args.hand_model, DEFAULT_HAND_LABEL, args.hand_conf)

    total_face = 0
    total_hand = 0
    total_mosaic = 0
    total_json = 0

    for image_path in tqdm(image_paths, desc="重标注", unit="img"):
        result = process_image_file(
            image_path=image_path,
            face_detector=face_detector,
            hand_detector=hand_detector,
            min_face_ratio=args.min_face_ratio,
            dry_run=args.dry_run,
            mosaic_size=args.mosaic_size,
        )
        total_face += result.face_count
        total_hand += result.hand_count
        total_mosaic += result.mosaic_count
        total_json += 1

        if args.dry_run:
            continue

        json_path = find_json_for_image(image_path)
        data = rewrite_labelme_dict(
            json_path,
            result.boxes,
            result.labels,
            result.image.shape,
            image_path.name,
        )
        cv2.imwrite(str(image_path), result.image)
        save_labelme(data, json_path)

    mode = "预览" if args.dry_run else "完成"
    print(f"[{mode}] 图片数：{len(image_paths)}")
    print(f"[{mode}] 保留 face：{total_face}")
    print(f"[{mode}] 保留 hand：{total_hand}")
    print(f"[{mode}] 小脸马赛克并删除：{total_mosaic}")
    print(f"[{mode}] 覆盖 JSON：{total_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
