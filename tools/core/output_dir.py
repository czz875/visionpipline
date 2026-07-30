"""
tools/core/output_dir.py

通用输出目录工具：为任意模块生成带时间戳的子目录，避免多次运行覆盖旧结果。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


_TIMESTAMP_RE = re.compile(r"^(.*)_(\d{8}_\d{6})(?:_(\d+))?$")
DEFAULT_BATCH_PREFIX = "batch"
_BATCH_DIR_RE = re.compile(r"^batch_(\d{8}_\d{6})(?:_(\d{3,}))?$")


def _batch_dir_sort_key(batch_dir: Path) -> tuple[str, int]:
    """返回批次目录的时间戳与数值序号排序键。"""
    match = _BATCH_DIR_RE.fullmatch(batch_dir.name)
    assert match is not None
    return match.group(1), int(match.group(2) or 0)


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


def build_batch_stage_dir(logical_stage_dir: Path | str) -> Path:
    """创建批次根目录，并返回其中的阶段目录。"""
    logical = Path(logical_stage_dir)
    batch_root = build_timestamped_output_dir(logical.parent, DEFAULT_BATCH_PREFIX)
    stage_dir = batch_root / logical.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def resolve_latest_batch_stage_dir(logical_stage_dir: Path | str) -> Path:
    """解析逻辑阶段路径对应的最新批次目录，无批次时返回原路径。"""
    logical = Path(logical_stage_dir)
    if _BATCH_DIR_RE.fullmatch(logical.parent.name):
        return logical
    batch_dirs = sorted(
        (
            item
            for item in logical.parent.iterdir()
            if item.is_dir() and _BATCH_DIR_RE.fullmatch(item.name)
        ),
        key=_batch_dir_sort_key,
        reverse=True,
    ) if logical.parent.is_dir() else []
    return next(
        (
            item / logical.name
            for item in batch_dirs
            if (item / logical.name).is_dir()
        ),
        logical,
    )


def build_related_batch_stage_dir(
    source_stage_dir: Path | str,
    logical_output_dir: Path | str,
) -> Path:
    """在来源阶段所属批次中创建关联输出阶段目录。"""
    source = Path(source_stage_dir)
    logical_output = Path(logical_output_dir)
    if _BATCH_DIR_RE.fullmatch(source.parent.name):
        output = source.parent / logical_output.name
        output.mkdir(parents=True, exist_ok=True)
        return output
    return build_batch_stage_dir(logical_output)
