from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import (
    DEFAULT_DATASET_PATH,
    find_json_for_image,
    list_images,
)

# 默认参数（集中放文件顶部，argparse 统一引用）
DEFAULT_RECURSIVE = False
DEFAULT_DRY_RUN = True


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="删除没有对应 LabelMe JSON 的图片文件（默认预览，加 --apply 才真删）。"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"数据集目录（默认：{DEFAULT_DATASET_PATH}）。",
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
        help="确认预览无误后，加 --apply 才真正删除（默认 dry-run）。",
    )
    return parser


def main() -> int:
    """脚本入口。"""
    args = _build_parser().parse_args()
    input_dir = args.dataset_path.resolve()
    dry_run = not args.apply

    if not input_dir.is_dir():
        print(f"[错误] 数据集目录不存在：{input_dir}")
        return 1

    image_files = list_images(input_dir, recursive=args.recursive)

    deleted = 0
    for image_file in image_files:
        json_file = find_json_for_image(image_file)
        if json_file.exists():
            continue
        rel_path = image_file.relative_to(input_dir)
        if dry_run:
            print(f"[预览] 将删除无JSON图片: {rel_path}")
        else:
            print(f"删除无JSON图片: {rel_path}")
            image_file.unlink()
        deleted += 1

    if dry_run:
        print(f"\n[预览] 共 {deleted} 张图片无对应JSON，未删除；确认无误请加 --apply。")
    else:
        print(f"\n完成，共删除 {deleted} 张无JSON图片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
