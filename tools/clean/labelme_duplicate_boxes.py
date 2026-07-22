from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import (
    DEFAULT_DATASET_PATH,
    build_timestamped_output_dir,
    find_image_for_json,
    list_labelme_files,
    load_labelme,
    save_labelme,
)

# 默认参数（集中放文件顶部，argparse 统一引用）
DEFAULT_OUTPUT_DIR = r"datasets\_duplicate_cleaned"
DEFAULT_TOLERANCE = 1e-6
DEFAULT_DRY_RUN = True
DEFAULT_RECURSIVE = True


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="清理 LabelMe JSON 中的重复矩形框，把结果复制到输出目录。"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"数据集目录（默认：{DEFAULT_DATASET_PATH}）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "输出目录（默认在 dataset-path 同级生成 "
            "duplicate_cleaned_YYYYMMDD_HHMMSS）。"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=DEFAULT_RECURSIVE,
        help="递归处理子目录（默认递归）。",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"坐标重复判定容差（默认：{DEFAULT_TOLERANCE}）。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="确认预览无误后，加 --apply 才真正复制并清理文件（默认 dry-run）。",
    )
    return parser


def _normalize_points(points: list[list[float]]) -> tuple[float, float, float, float]:
    """把两点矩形归一化为 (x_min, y_min, x_max, y_max)。"""
    x1, y1 = float(points[0][0]), float(points[0][1])
    x2, y2 = float(points[1][0]), float(points[1][1])
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _shape_key(shape: dict, decimals: int) -> tuple[str, tuple[float, ...]] | None:
    """把 shape 转成可去重的键；非矩形或异常 shape 返回 None。"""
    if shape.get("shape_type") != "rectangle":
        return None
    points = shape.get("points", [])
    if len(points) < 2:
        return None
    try:
        box = _normalize_points(points)
    except (ValueError, TypeError, IndexError):
        return None
    label = shape.get("label", "")
    rounded = tuple(round(v, decimals) for v in box)
    return (label, rounded)


def _clean_json(data: dict, decimals: int) -> tuple[dict, int]:
    """返回清理后的 LabelMe dict 以及被移除的重复 shape 数。"""
    seen: set[tuple[str, tuple[float, ...]]] = set()
    cleaned_shapes: list[dict] = []
    duplicate_count = 0

    for shape in data.get("shapes", []):
        key = _shape_key(shape, decimals)
        if key is None:
            cleaned_shapes.append(shape)
            continue
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        cleaned_shapes.append(shape)

    new_data = dict(data)
    new_data["shapes"] = cleaned_shapes
    return new_data, duplicate_count


def main() -> int:
    """脚本入口。"""
    args = _build_parser().parse_args()
    input_dir = args.dataset_path.resolve()
    dry_run = not args.apply
    decimals = max(0, int(-math.log10(args.tolerance)))

    if args.output_dir is None:
        output_dir = build_timestamped_output_dir(
            input_dir.parent, "duplicate_cleaned"
        )
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list_labelme_files(input_dir, recursive=args.recursive)

    total_duplicates = 0
    modified_files = 0
    copied_images = 0

    for json_file in json_files:
        rel_path = json_file.relative_to(input_dir)
        dst_json = output_dir / rel_path
        dst_json.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = load_labelme(json_file)
        except Exception as exc:  # noqa: BLE001
            print(f"读取失败跳过: {rel_path} ({exc})")
            continue

        cleaned_data, duplicate_count = _clean_json(data, decimals)
        if duplicate_count:
            modified_files += 1
            total_duplicates += duplicate_count
            print(f"发现重复框: {rel_path} -> {duplicate_count} 个")

        if not dry_run:
            save_labelme(cleaned_data, dst_json)
            image_file = find_image_for_json(json_file)
            if image_file is not None and image_file.exists():
                dst_image = dst_json.with_suffix(image_file.suffix)
                shutil.copy2(image_file, dst_image)
                copied_images += 1

    action = "已清理" if not dry_run else "可清理"
    print(
        f"\n完成，{action} {modified_files} 个文件，"
        f"共 {total_duplicates} 个重复矩形框"
    )
    if not dry_run:
        print(f"复制图片 {copied_images} 张")
    print(f"输出目录: {output_dir}")
    print(f"OUTPUT_PATH:{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
