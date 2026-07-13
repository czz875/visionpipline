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


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="删除没有对应 LabelMe JSON 的图片文件。"
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
        help="递归处理子目录（默认只处理顶层）。",
    )
    return parser


def main() -> int:
    """脚本入口。"""
    args = _build_parser().parse_args()
    image_files = list_images(args.dataset_path.resolve(), recursive=args.recursive)

    deleted = 0
    for image_file in image_files:
        json_file = find_json_for_image(image_file)

        if not json_file.exists():
            print(f"删除无JSON图片: {image_file.name}")
            image_file.unlink()
            deleted += 1

    print(f"\n完成，共删除 {deleted} 张图片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
