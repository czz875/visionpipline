"""
tools/core/output_dir.py

通用输出目录工具：为任意模块生成带时间戳的子目录，避免多次运行覆盖旧结果。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_timestamped_output_dir(base_dir: Path | str, prefix: str) -> Path:
    """在 ``base_dir`` 下生成 ``<prefix>_YYYYMMDD_HHMMSS`` 目录并返回。

    目录不会被自动创建，调用方需要时自行 ``mkdir(parents=True, exist_ok=True)``。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / f"{prefix}_{timestamp}"
