"""
tools/core/constants.py

工具脚本共享的基础常量。
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_DATASET_PATH = Path("datasets/behavior").resolve()
LABELME_EXT = ".json"

# 批次目录名（4 位数字，如 ``0001`` / ``0156``）。数据集按批次分目录存放，
# 扫描批次范围时用它判定「哪些子目录算批次」，多处共用此口径。
BATCH_DIR_RE = re.compile(r"^\d{4}$")

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

# YOLO 类别表：**唯一权威来源**。顺序即 class id，改动会让全部已转好的
# 标签与 data.yaml 错位（ultralytics 会把越界 id 的图判为 corrupt 整张丢弃）。
# 磁盘上 datasets/yolo/*/data.yaml 的 nc / names 必须与此一致。
YOLO_CLASS_NAMES: tuple[str, ...] = (
    "phone",
    "cigarette",
    "face",
    "hand",
    "helmet",
)

# 类别判定优先级：一张图命中多类时按此顺序归类（先到先得）。
CLASS_PRIORITY: tuple[str, ...] = (
    "helmet",
    "cigarette",
    "phone",
    "hand",
    "face",
)
