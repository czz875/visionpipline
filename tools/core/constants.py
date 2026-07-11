"""
tools/core/constants.py

工具脚本共享的基础常量。
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATASET_PATH = Path("datasets/autolabel").resolve()
LABELME_EXT = ".json"
IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
