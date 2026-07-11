"""
tools/train/create_tar_gz.py
把指定文件夹压缩为 ``.tar.gz`` 归档。

典型用法：

    .conda\python.exe tools\train\create_tar_gz.py ^
        --input archive\20260711_163012_cjet_dataset ^
        --level 6
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

from tqdm import tqdm

# 允许以 `python tools/train/create_tar_gz.py` 直接运行。
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_INPUT_DIR = r"D:\A_CJET_WORKSPACE\YOLO_Detect\BEHAVIOR_DETECT\BehaviorData\Datasets_train\20250701_Call_Smoke_Face_IR"
DEFAULT_GZ_LEVEL = 6
DEFAULT_SUFFIX = ".tar.gz"


# =============================================================================
# 2. 压缩逻辑
# =============================================================================


def _collect_files(folder: Path) -> list[tuple[Path, str]]:
    """枚举 folder 下所有文件，返回 (绝对路径, 归档内相对名) 列表。"""
    parent_dir = folder.parent
    file_list: list[tuple[Path, str]] = []
    for root, _, files in os.walk(folder):
        for name in files:
            full_path = Path(root) / name
            arcname = str(full_path.relative_to(parent_dir))
            file_list.append((full_path, arcname))
    return file_list


def compress_folder_to_targz(
    folder: Path,
    *,
    output: Path | None = None,
    gz_level: int = DEFAULT_GZ_LEVEL,
) -> Path:
    """把 folder 压缩为 tar.gz。

    Args:
        folder: 待压缩的文件夹。
        output: 目标 tar.gz 路径，默认放在 folder 的同级目录，名为 ``<folder>.tar.gz``。
        gz_level: gzip 压缩等级 0-9，0=不压缩，9=最慢最小，默认 6。

    Returns:
        最终生成的 tar.gz 文件路径。
    """
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在或不是目录：{folder}")

    if output is None:
        output = folder.parent / f"{folder.name}{DEFAULT_SUFFIX}"
    else:
        output = Path(output)

    file_list = _collect_files(folder)
    print(f"[信息] 待压缩文件数：{len(file_list)}")
    print(f"[信息] 输出路径：{output}")

    with tarfile.open(output, f"w:gz", compresslevel=gz_level) as tar:
        for full_path, arcname in tqdm(file_list, desc="压缩进度"):
            tar.add(str(full_path), arcname=arcname)

    print(f"[完成] 压缩完成：{output}")
    return output


# =============================================================================
# 3. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把指定文件夹压缩为 tar.gz 归档。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"待压缩的文件夹（默认：{DEFAULT_INPUT_DIR}）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="目标 tar.gz 路径，默认写到 <input 的父目录>/<input 的名字>.tar.gz。",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=DEFAULT_GZ_LEVEL,
        choices=range(0, 10),
        help=f"gzip 压缩等级 0-9（默认 {DEFAULT_GZ_LEVEL}）。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    compress_folder_to_targz(
        folder=args.input,
        output=args.output,
        gz_level=args.level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
