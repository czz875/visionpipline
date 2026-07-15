from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import DEFAULT_DATASET_PATH


# 默认参数（集中放文件顶部，argparse / 回退统一引用）
DEFAULT_INPUT = DEFAULT_DATASET_PATH
DEFAULT_OUTPUT = DEFAULT_DATASET_PATH


def copy_dataset(src: Path, dst: Path) -> int:
    """把 src 目录复制到 dst。

    src 与 dst 相同（解析后）则跳过，返回 0；否则用 copytree 整体复制。
    可用于清理 / 标注前先把源目录留底，输入目录始终不动。
    """
    src = Path(src).resolve()
    dst = Path(dst).resolve()

    if src == dst:
        print(f"[跳过] 输入与输出相同，原地清理：{dst}")
        return 0

    print(f"[复制] {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print("\n完成，已复制到输出目录")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="把输入目录复制到输出目录（输入=输出则跳过，用于清理前留底）。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入目录（源，默认：{DEFAULT_INPUT}）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出目录（目标，默认：{DEFAULT_OUTPUT}）。",
    )
    return parser


def main() -> int:
    """命令行入口。"""
    args = _build_parser().parse_args()
    return copy_dataset(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
