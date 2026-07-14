"""
tools/annotate/reannotate_onnx.py
通用重标注：用 ONNX 检测器重新标出指定类别（``--onnx-label``），覆盖该类别的
旧框；同时保留现有 JSON 中其它指定类别（``--keep-labels``，默认保留除
``--onnx-label`` 外的所有现有类别）的框。对两路中面积过小（按各自
``--*-min-ratio``）的框执行打码（默认纯黑，可用 ``--mosaic`` 改用马赛克）后删除，
且小框与同路保留大框重叠区域不打码（重叠保护）。

类别与模型完全由参数决定（不再写死 face / hand）。
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

from tools.annotate.backends.onnx import OnnxDetector
from tools.annotate.ops import (
    blackout_region,
    collect_blackout_regions,
    concat_boxes,
    extract_existing_label_boxes,
    mosaic_region,
    rewrite_labelme_dict,
    sanitize_boxes,
    split_boxes_by_ratio,
)
from tools.core import find_json_for_image, list_images, load_labelme, save_labelme


# =============================================================================
# 1. 默认参数
# =============================================================================

# 默认 ONNX 模型（示例：人脸 yolov5s-lmk.onnx）及其输出解码配套参数。
DEFAULT_ONNX_MODEL = r"weight\yolov5s-lmk.onnx"
DEFAULT_ONNX_LABEL = "face"
DEFAULT_ONNX_CONF = 0.25
DEFAULT_ONNX_MIN_RATIO = 0.01
DEFAULT_ONNX_TRANSPOSE = False
DEFAULT_ONNX_SCORE_INDICES = "4,15"   # 人脸模型：obj 分 * 关键点置信度
DEFAULT_ONNX_NORMALIZE = True

DEFAULT_INPUT_ROOT = r"datasets\behavior"
DEFAULT_START_BATCH = 23
DEFAULT_END_BATCH = 37
DEFAULT_MOSAIC_BLOCK = 16
DEFAULT_DRY_RUN = True


# =============================================================================
# 2. 数据结构
# =============================================================================


@dataclass
class ProcessResult:
    image: np.ndarray
    boxes: np.ndarray
    labels: list[str]
    onnx_blackout_count: int
    keep_blackout_count: int
    onnx_count: int
    keep_count: int


# =============================================================================
# 3. 文件发现
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


def resolve_keep_labels(
    json_path: Path,
    onnx_label: str,
    explicit: list[str] | None,
) -> list[str]:
    """解析需要保留的现有类别。

    ``explicit`` 为 ``None`` 时，自动保留 JSON 中除 ``onnx_label`` 外的所有
    矩形类别；否则使用显式指定的类别列表。
    """
    if explicit is not None:
        return explicit
    if not json_path.exists():
        return []
    data = load_labelme(json_path)
    labels: list[str] = []
    seen: set[str] = set()
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape.get("label", "")
        if label == onnx_label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


# =============================================================================
# 4. 单图处理
# =============================================================================


def process_image_file(
    image_path: Path,
    onnx_detector: OnnxDetector,
    onnx_label: str,
    min_onnx_ratio: float,
    keep_labels: list[str] | None,
    min_keep_ratio: float,
    use_mosaic: bool,
    mosaic_block: int,
    dry_run: bool,
) -> ProcessResult:
    """对单张图片执行 ONNX 覆盖标注 + 保留现有类别 + 小框打码过滤。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")

    json_path = find_json_for_image(image_path)

    # ONNX 一路：重新标出目标类别。
    onnx_boxes, _ = onnx_detector.predict(image)
    onnx_boxes = sanitize_boxes(onnx_boxes, image.shape)
    kept_onnx, removed_onnx = split_boxes_by_ratio(onnx_boxes, image.shape, min_onnx_ratio)
    onnx_blackouts = collect_blackout_regions(removed_onnx, kept_onnx)

    # 保留一路：从现有 JSON 取其它类别框。
    resolved_keep = resolve_keep_labels(json_path, onnx_label, keep_labels)
    keep_parts: list[tuple[np.ndarray, str]] = []
    for label in resolved_keep:
        boxes = extract_existing_label_boxes(json_path, label)
        boxes = sanitize_boxes(boxes, image.shape)
        if len(boxes):
            keep_parts.append((boxes, label))
    keep_boxes_all = concat_boxes([b for b, _ in keep_parts])
    kept_keep, removed_keep = split_boxes_by_ratio(keep_boxes_all, image.shape, min_keep_ratio)
    keep_blackouts = collect_blackout_regions(removed_keep, kept_keep)

    # 打码（纯黑或马赛克）。
    if not dry_run:
        for box in onnx_blackouts:
            image = mosaic_region(image, box, mosaic_block) if use_mosaic else blackout_region(image, box)
        for box in keep_blackouts:
            image = mosaic_region(image, box, mosaic_block) if use_mosaic else blackout_region(image, box)

    final_boxes = concat_boxes([kept_onnx, kept_keep])
    final_labels = [onnx_label] * len(kept_onnx)
    for boxes, label in keep_parts:
        final_labels.extend([label] * len(boxes))

    return ProcessResult(
        image=image,
        boxes=final_boxes,
        labels=final_labels,
        onnx_blackout_count=len(onnx_blackouts),
        keep_blackout_count=len(keep_blackouts),
        onnx_count=len(kept_onnx),
        keep_count=len(kept_keep),
    )


# =============================================================================
# 5. 命令行参数
# =============================================================================


def _parse_score_indices(text: str) -> tuple[int, ...]:
    """解析 ``"4,15"`` 形式的分通道下标为 int 元组。"""
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(
        description="用 ONNX 重新标出指定类别并覆盖其旧框，保留其它现有类别，"
        "小框打码（默认纯黑）后删除。",
    )
    parser.add_argument(
        "--onnx-model", type=Path, default=Path(DEFAULT_ONNX_MODEL),
        help=f"ONNX 模型路径（默认：{DEFAULT_ONNX_MODEL}）。",
    )
    parser.add_argument(
        "--onnx-label", default=DEFAULT_ONNX_LABEL,
        help=f"要覆盖的类别名（默认：{DEFAULT_ONNX_LABEL}）。",
    )
    parser.add_argument(
        "--onnx-conf", type=float, default=DEFAULT_ONNX_CONF,
        help=f"ONNX 置信度阈值（默认：{DEFAULT_ONNX_CONF}）。",
    )
    parser.add_argument(
        "--onnx-min-ratio", type=float, default=DEFAULT_ONNX_MIN_RATIO,
        help=f"ONNX 保留框的最小面积占比（默认：{DEFAULT_ONNX_MIN_RATIO}）。",
    )
    parser.add_argument(
        "--onnx-transpose", action="store_true", default=DEFAULT_ONNX_TRANSPOSE,
        help="ONNX 输出布局为 (C, N)，需转置为 (N, C) 再解码（默认关闭）。",
    )
    parser.add_argument(
        "--onnx-score-indices", default=DEFAULT_ONNX_SCORE_INDICES,
        help=f"参与相乘得到分数的输出通道下标，逗号分隔（默认：{DEFAULT_ONNX_SCORE_INDICES}）。",
    )
    parser.add_argument(
        "--onnx-normalize", dest="onnx_normalize", action="store_true",
        default=DEFAULT_ONNX_NORMALIZE,
        help="预处理时对像素除以 255 归一化（默认开启）。",
    )
    parser.add_argument(
        "--no-onnx-normalize", dest="onnx_normalize", action="store_false",
        help="关闭像素归一化。",
    )
    parser.add_argument(
        "--input-root", type=Path, default=Path(DEFAULT_INPUT_ROOT),
        help=f"按 batch 子目录组织的数据根目录（默认：{DEFAULT_INPUT_ROOT}）。",
    )
    parser.add_argument(
        "--start-batch", type=int, default=DEFAULT_START_BATCH,
        help=f"起始 batch 编号（默认：{DEFAULT_START_BATCH}）。",
    )
    parser.add_argument(
        "--end-batch", type=int, default=DEFAULT_END_BATCH,
        help=f"结束 batch 编号（默认：{DEFAULT_END_BATCH}）。",
    )
    parser.add_argument(
        "--keep-labels",
        default=None,
        help="保留的现有类别，逗号分隔（如 'hand,phone'）。留空则自动保留除 "
             "--onnx-label 外的所有现有类别。",
    )
    parser.add_argument(
        "--keep-min-ratio", type=float, default=None,
        help="保留类别框的最小面积占比；默认与 --onnx-min-ratio 相同。",
    )
    parser.add_argument(
        "--mosaic", dest="use_mosaic", action="store_true",
        help="小框用马赛克而非纯黑打码。",
    )
    parser.add_argument(
        "--mosaic-block", type=int, default=DEFAULT_MOSAIC_BLOCK,
        help=f"马赛克块大小（像素），越大越糊（默认：{DEFAULT_MOSAIC_BLOCK}）。",
    )
    parser.add_argument(
        "--apply", dest="dry_run", action="store_false",
        help="真正写回图片与 JSON；默认仅 dry-run 统计。",
    )
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


# =============================================================================
# 6. 运行逻辑
# =============================================================================


def run(args: argparse.Namespace) -> int:
    """执行批量重标注。"""
    image_paths = iter_image_files(args.input_root, args.start_batch, args.end_batch)
    if not image_paths:
        print("[错误] 指定 batch 范围内没有可处理图片。", file=sys.stderr)
        return 1

    onnx_detector = OnnxDetector(
        args.onnx_model,
        args.onnx_label,
        args.onnx_conf,
        normalize=args.onnx_normalize,
        transpose=args.onnx_transpose,
        score_indices=_parse_score_indices(args.onnx_score_indices),
    )
    keep_labels = (
        [s.strip() for s in args.keep_labels.split(",") if s.strip()]
        if args.keep_labels
        else None
    )
    min_keep_ratio = args.keep_min_ratio if args.keep_min_ratio is not None else args.onnx_min_ratio

    total_onnx = total_keep = total_blackout = total_json = 0

    for image_path in tqdm(image_paths, desc="重标注", unit="img"):
        result = process_image_file(
            image_path=image_path,
            onnx_detector=onnx_detector,
            onnx_label=args.onnx_label,
            min_onnx_ratio=args.onnx_min_ratio,
            keep_labels=keep_labels,
            min_keep_ratio=min_keep_ratio,
            use_mosaic=args.use_mosaic,
            mosaic_block=args.mosaic_block,
            dry_run=args.dry_run,
        )
        total_onnx += result.onnx_count
        total_keep += result.keep_count
        total_blackout += result.onnx_blackout_count + result.keep_blackout_count
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
    print(f"[{mode}] 覆盖 {args.onnx_label}：{total_onnx}")
    print(f"[{mode}] 保留其它类别框：{total_keep}")
    print(f"[{mode}] 小框打码并删除：{total_blackout}")
    print(f"[{mode}] 覆盖 JSON：{total_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
