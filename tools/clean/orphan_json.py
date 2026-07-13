from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import (
    DEFAULT_DATASET_PATH,
    find_image_for_json,
    list_labelme_files,
)


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="删除没有对应图片的 LabelMe JSON 文件。"
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
    json_files = list_labelme_files(args.dataset_path.resolve(), recursive=args.recursive)

    deleted = 0
    for json_file in json_files:
        if find_image_for_json(json_file) is None:
            print(f"删除无图片JSON: {json_file.name}")
            json_file.unlink()
            deleted += 1

    print(f"\n完成，共删除 {deleted} 个JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
