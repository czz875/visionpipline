from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import (
    DEFAULT_DATASET_PATH,
    build_timestamped_output_dir,
    find_json_for_image,
    list_images,
)

# 默认参数（集中放文件顶部，argparse 统一引用）
DEFAULT_OUTPUT_DIR = r"datasets\_orphan_cleaned"
DEFAULT_DRY_RUN = True
DEFAULT_RECURSIVE = False


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="清理没有对应 LabelMe JSON 的图片文件，把保留的 PNG+JSON 对复制到输出目录。"
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
            "orphan_cleaned_YYYYMMDD_HHMMSS）。"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=DEFAULT_RECURSIVE,
        help="递归处理子目录（默认只处理顶层）。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="确认预览无误后，加 --apply 才真正复制文件（默认 dry-run）。",
    )
    return parser


def main() -> int:
    """脚本入口。"""
    args = _build_parser().parse_args()
    input_dir = args.dataset_path.resolve()
    dry_run = not args.apply

    if args.output_dir is None:
        output_dir = build_timestamped_output_dir(
            input_dir.parent, "orphan_cleaned"
        )
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_images(input_dir, recursive=args.recursive)

    kept = 0
    orphaned = 0
    plan: list[tuple[Path, Path, Path]] = []  # (image_src, json_src, image_dst)
    for image_file in image_files:
        json_file = find_json_for_image(image_file)
        if not json_file.exists():
            orphaned += 1
            print(f"无JSON跳过: {image_file.relative_to(input_dir)}")
            continue

        rel_path = image_file.relative_to(input_dir)
        dst_image = output_dir / rel_path
        dst_json = dst_image.with_suffix(".json")
        plan.append((image_file, json_file, dst_image))

    if dry_run:
        print(f"\n[预览] 将保留 {len(plan)} 张图，跳过 {orphaned} 张无JSON图")
        print(f"[预览] 输出目录: {output_dir}")
    else:
        for image_src, json_src, image_dst in plan:
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_src, image_dst)
            json_dst = image_dst.with_suffix(".json")
            shutil.copy2(json_src, json_dst)
            kept += 1
        print(f"\n完成，保留 {kept} 张图，跳过 {orphaned} 张无JSON图")
        print(f"输出目录: {output_dir}")

    print(f"OUTPUT_PATH:{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
