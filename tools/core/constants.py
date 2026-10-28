"""
tools/core/constants.py

工具脚本共享的基础常量。
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATASET_PATH = Path("datasets/autolabel").resolve()
LABELME_EXT = ".json"
# 必须是有序容器：find_image_for_json 按此顺序决定同名多格式图片的优先级。
# 用 frozenset 时字符串哈希每进程随机化，迭代顺序随之变化，同一份数据在不同
# 进程会选中不同图片（孤儿判定、尺寸修正时对时错）。按常见优先级排序：
# 无损优先于有损，PNG 为流水线主格式。
IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
)
