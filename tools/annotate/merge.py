"""
tools/annotate/merge.py
合并 LabelMe 中距离相近的同标签矩形框。

默认针对 ``face`` 标签做右侧过滤与最小尺寸过滤，也支持对任意标签执行邻近合并。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import (
    list_labelme_files,
    load_labelme,
    save_labelme,
)
from tools.core.geometry import (
    merge_near_boxes,
    rect_to_xyxy,
    xyxy_to_points,
)

# ---------- 默认参数 ----------
DEFAULT_JSON_DIR = Path(r"D:\A_CJET_WORKSPACE\01_CJET_DATASET\tmp")
DEFAULT_FACE_LABEL = "face"
DEFAULT_RIGHT_CUT_RATIO = 0.0
DEFAULT_MIN_FACE_WIDTH = 100
DEFAULT_MIN_FACE_HEIGHT = 50
DEFAULT_MERGE_DISTANCE_X = 100
DEFAULT_MERGE_DISTANCE_Y = 200
DEFAULT_MAX_MERGE_WIDTH = 600
DEFAULT_MAX_MERGE_HEIGHT = 600
DEFAULT_MAX_MERGE_COUNT = 10
DEFAULT_MAX_WORKERS = 12


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="递归合并 LabelMe JSON 中距离相近的同标签矩形框。"
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help=f"待处理的 LabelMe JSON 根目录（默认：{DEFAULT_JSON_DIR}）。",
    )
    parser.add_argument(
        "--face-label",
        default=DEFAULT_FACE_LABEL,
        help=f"需要特殊过滤/后处理的标签名（默认：{DEFAULT_FACE_LABEL}）。",
    )
    parser.add_argument(
        "--right-cut-ratio",
        type=float,
        default=DEFAULT_RIGHT_CUT_RATIO,
        help="face 框中心位于图片右侧该比例以内时跳过（默认：%(default)s）。",
    )
    parser.add_argument(
        "--min-face-width",
        type=float,
        default=DEFAULT_MIN_FACE_WIDTH,
        help="face 合并后最小宽度（默认：%(default)s）。",
    )
    parser.add_argument(
        "--min-face-height",
        type=float,
        default=DEFAULT_MIN_FACE_HEIGHT,
        help="face 合并后最小高度（默认：%(default)s）。",
    )
    parser.add_argument(
        "--merge-distance-x",
        type=float,
        default=DEFAULT_MERGE_DISTANCE_X,
        help="x 方向合并距离阈值（默认：%(default)s）。",
    )
    parser.add_argument(
        "--merge-distance-y",
        type=float,
        default=DEFAULT_MERGE_DISTANCE_Y,
        help="y 方向合并距离阈值（默认：%(default)s）。",
    )
    parser.add_argument(
        "--max-merge-width",
        type=float,
        default=DEFAULT_MAX_MERGE_WIDTH,
        help="合并后框最大宽度（默认：%(default)s）。",
    )
    parser.add_argument(
        "--max-merge-height",
        type=float,
        default=DEFAULT_MAX_MERGE_HEIGHT,
        help="合并后框最大高度（默认：%(default)s）。",
    )
    parser.add_argument(
        "--max-merge-count",
        type=int,
        default=DEFAULT_MAX_MERGE_COUNT,
        help="单框最多允许合并次数（默认：%(default)s）。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"并发线程数（默认：{DEFAULT_MAX_WORKERS}）。",
    )
    return parser


def _extract_shape_meta(shape: dict[str, Any]) -> dict[str, Any]:
    """从 shape 中提取生成新 shape 所需的公共元数据。"""
    return {
        "label": shape["label"],
        "group_id": shape.get("group_id"),
        "description": shape.get("description", ""),
        "shape_type": "rectangle",
        "flags": shape.get("flags", {}),
    }


def process_labelme_json(json_path: Path, args: argparse.Namespace) -> str | None:
    """处理单个 LabelMe JSON 文件。

    Args:
        json_path: 待处理的 JSON 路径。
        args: 命令行参数命名空间。

    Returns:
        出错时返回错误信息，成功返回 ``None``。
    """
    try:
        data = load_labelme(json_path)
    except Exception as e:
        return f"Skip {json_path}: Read error {e}"

    shapes = data.get("shapes", [])
    img_w = data.get("imageWidth")
    if img_w is None:
        return f"Skip {json_path}: imageWidth missing"

    right_cut_x = img_w * (1 - args.right_cut_ratio)

    label_boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    label_meta: dict[str, dict[str, Any]] = {}

    for shape in shapes:
        if shape.get("shape_type") != "rectangle":
            continue

        label = shape["label"]
        box = rect_to_xyxy(shape["points"])

        if label == args.face_label:
            x_center = (box[0] + box[2]) / 2
            if x_center >= right_cut_x:
                continue

        label_boxes[label].append(box)
        if label not in label_meta:
            label_meta[label] = _extract_shape_meta(shape)

    new_shapes: list[dict[str, Any]] = []
    for label, boxes in label_boxes.items():
        if not boxes:
            continue

        merged_boxes = merge_near_boxes(
            boxes,
            distance_x=args.merge_distance_x,
            distance_y=args.merge_distance_y,
            max_width=args.max_merge_width,
            max_height=args.max_merge_height,
            max_count=args.max_merge_count,
        )

        for box in merged_boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]

            if label == args.face_label and (
                width < args.min_face_width or height < args.min_face_height
            ):
                continue

            new_shapes.append({
                **label_meta[label],
                "points": xyxy_to_points(box),
            })

    data["shapes"] = new_shapes
    try:
        save_labelme(data, json_path)
    except Exception as e:
        return f"Error saving {json_path}: {e}"

    return None


def process_json_dir_mt(dir_path: Path, args: argparse.Namespace) -> None:
    """递归处理目录下所有 LabelMe JSON 文件。"""
    if not dir_path.exists():
        print(f"Error: Directory {dir_path} does not exist.")
        return

    json_files = list_labelme_files(dir_path, recursive=True)
    if not json_files:
        print(f"No JSON files found in {dir_path} or its subdirectories.")
        return

    print(f"Found {len(json_files)} JSON files. Starting recursive merge...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_labelme_json, p, args): p for p in json_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging"):
            error_msg = future.result()
            if error_msg:
                print(error_msg)


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()
    process_json_dir_mt(args.json_dir.resolve(), args)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
