"""
tools/core/paths.py

路径相关通用工具。
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path


if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_timestamped_output_dir(base_dir: Path | str, function_name: str) -> Path:
    """在 base_dir 下创建并返回 `[function_name]_[YYYYMMDD_HHMMSS]` 目录。

    若同一秒内多次调用，自动追加 `_001`、`_002` ... 序号避免冲突。
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{function_name}_{timestamp}"
    candidate = base / name

    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    # 查找同名同时间戳且带三位序号的目录，取最大序号
    pattern = re.compile(rf"^{re.escape(name)}_(?P<seq>\d{{3}})$")
    max_seq = 0
    for item in base.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                max_seq = max(max_seq, int(match.group("seq")))

    new_name = f"{name}_{max_seq + 1:03d}"
    new_path = base / new_name
    new_path.mkdir(parents=True, exist_ok=True)
    return new_path
