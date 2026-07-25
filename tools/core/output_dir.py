"""
tools/core/output_dir.py

通用输出目录工具：为任意模块生成带时间戳的子目录，避免多次运行覆盖旧结果。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


_TIMESTAMP_RE = re.compile(r"^(.*)_(\d{8}_\d{6})(?:_(\d+))?$")


def build_timestamped_output_dir(base_dir: Path | str, prefix: str) -> Path:
    """在 ``base_dir`` 下创建并返回 ``<prefix>_YYYYMMDD_HHMMSS`` 目录。

    若同一秒内多次调用，自动追加 ``_001``、``_002`` ... 序号避免冲突。
    调用方仍可继续对该目录执行 ``mkdir(parents=True, exist_ok=True)``，
    不会因重复创建而报错。
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{timestamp}"
    candidate = base / name

    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    # 解决秒级冲突：找最大序号并 +1
    max_seq = 0
    for item in base.iterdir():
        if not item.is_dir():
            continue
        m = _TIMESTAMP_RE.match(item.name)
        if m and m.group(1) == prefix and m.group(2) == timestamp:
            seq = int(m.group(3)) if m.group(3) else 0
            max_seq = max(max_seq, seq)

    candidate = base / f"{name}_{max_seq + 1:03d}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate
