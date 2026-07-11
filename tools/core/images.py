"""
tools/core/images.py

图片相关工具函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tools.core.constants import IMAGE_EXTENSIONS


def list_images(
    folder: Path,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
    recursive: bool = True,
) -> list[Path]:
    """收集目录下受支持的图片路径，按路径排序保证可复现。

    Args:
        folder: 待扫描目录。
        extensions: 视为图片的扩展名集合，匹配时大小写不敏感。
        recursive: 是否递归子目录；False 时只扫描顶层。

    Returns:
        按路径字符串排序后的图片路径列表。
    """
    ext_set = {ext.lower() for ext in extensions}
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p for p in candidates
        if p.is_file() and p.suffix.lower() in ext_set
    )
