"""
每日交付/归档脚本。

把训练相关的目录或文件复制到按时间戳命名的归档目录，并记录元数据。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_ARCHIVE_ROOT = Path("archive").resolve()
# 硬链接归档：同一卷内不额外占用空间，避免把 datasets/ 的硬链接图片实拷一份。
DEFAULT_USE_HARDLINK = True


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="把训练产物归档到按时间戳命名的目录。"
    )
    parser.add_argument(
        "--sources",
        required=True,
        help="待归档的路径，逗号分隔。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help=f"归档根目录（默认：{DEFAULT_ARCHIVE_ROOT}）。",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="额外标签，会附加到归档目录名中。",
    )
    parser.add_argument(
        "--note",
        default="",
        help="备注信息，写入 metadata.json。",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink" if DEFAULT_USE_HARDLINK else "copy",
        help=(
            "目录归档方式：hardlink 用硬链接（同一卷、不额外占空间，默认），"
            "copy 逐文件实拷。修改硬链接归档里的图片会影响原数据，只读使用。"
        ),
    )
    return parser


def _copy_source(src: Path, archive_dir: Path, *, use_hardlink: bool) -> None:
    """复制单个源路径到归档目录。

    ``use_hardlink=True`` 时用 ``shutil.copytree(copy_function=os.link)``：
    数据集里的图片本身多为硬链接，实拷会把整份数据集展开成真实字节
    （``datasets/yolo`` 可达数十 GB），因此默认走硬链接。
    """
    dst = archive_dir / src.name
    if not src.exists():
        print(f"[跳过] 源路径不存在：{src}")
        return

    if src.is_dir():
        if use_hardlink:
            try:
                shutil.copytree(
                    src, dst, dirs_exist_ok=True, copy_function=os.link
                )
                return
            except OSError as e:
                print(
                    f"[警告] 硬链接归档失败，回退为复制：{src}（{e}）"
                )
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def archive(
    sources: list[str],
    output_root: Path,
    tag: str = "",
    note: str = "",
    *,
    use_hardlink: bool = DEFAULT_USE_HARDLINK,
) -> Path:
    """执行归档并返回归档目录路径。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}"
    if tag:
        name = f"{name}_{tag}"
    archive_dir = output_root / name
    archive_dir.mkdir(parents=True, exist_ok=True)

    for src_str in sources:
        src = Path(src_str.strip())
        _copy_source(src, archive_dir, use_hardlink=use_hardlink)

    metadata = {
        "timestamp": timestamp,
        "tag": tag,
        "note": note,
        "sources": sources,
        "copy_mode": "hardlink" if use_hardlink else "copy",
    }
    (archive_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return archive_dir


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    sources = [s for s in args.sources.split(",") if s.strip()]
    if not sources:
        print("[错误] --sources 不能为空。")
        return 1

    archive_dir = archive(
        sources,
        args.output.resolve(),
        tag=args.tag,
        note=args.note,
        use_hardlink=args.copy_mode == "hardlink",
    )
    print(f"已归档到：{archive_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
