from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import DEFAULT_DATASET_PATH

# 默认参数（集中放文件顶部，argparse / 模型回退统一引用）
DEFAULT_LABEL_EXT = ".txt"
DEFAULT_TOLERANCE = 1e-6


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="清理 YOLO 标签文件中的重复标注行（默认只预览，加 --apply 才写盘）。"
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
    parser.add_argument(
        "--apply",
        action="store_true",
        help="确认预览无误后，加 --apply 才真正修改文件。",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"判定 bbox 重复时的数值容差（默认：{DEFAULT_TOLERANCE}）。",
    )
    return parser


def _list_label_files(root: Path, recursive: bool) -> list[Path]:
    """列出目标目录下所有 YOLO 标签文件。"""
    pattern = f"*{DEFAULT_LABEL_EXT}"
    if recursive:
        return sorted(root.rglob(pattern))
    return sorted(root.glob(pattern))


def _parse_line(line: str, decimals: int):
    """把一行 YOLO 标签解析成可去重的键；非标准行返回 None。"""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        bbox = tuple(round(float(part), decimals) for part in parts[1:])
        return (cls,) + bbox
    except ValueError:
        return None


def _clean_file(label_file: Path, decimals: int) -> tuple[list[str], int]:
    """返回清理后的行列表以及被移除的重复行数。"""
    original_lines = label_file.read_text(encoding="utf-8").splitlines()
    seen: set[tuple[int, float, float, float, float]] = set()
    cleaned_lines: list[str] = []
    duplicate_count = 0

    for line in original_lines:
        key = _parse_line(line, decimals)
        if key is None:
            # 保留空行或格式异常行（让调用方自己决定后续如何处理）
            cleaned_lines.append(line)
            continue
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        cleaned_lines.append(line)

    return cleaned_lines, duplicate_count


def main() -> int:
    """脚本入口。"""
    args = _build_parser().parse_args()
    root = args.dataset_path.resolve()
    decimals = max(0, int(-math.log10(args.tolerance)))

    label_files = _list_label_files(root, args.recursive)
    total_duplicates = 0
    modified_files = 0

    for label_file in label_files:
        cleaned_lines, duplicate_count = _clean_file(label_file, decimals)
        if duplicate_count == 0:
            continue

        total_duplicates += duplicate_count
        modified_files += 1
        print(f"{label_file}: {duplicate_count} duplicate labels removed")

        if args.apply:
            content = "\n".join(cleaned_lines)
            if cleaned_lines:
                content += "\n"
            label_file.write_text(content, encoding="utf-8")

    action = "已清理" if args.apply else "可清理"
    print(f"\n完成，{action} {modified_files} 个文件，共 {total_duplicates} 行重复标签")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
