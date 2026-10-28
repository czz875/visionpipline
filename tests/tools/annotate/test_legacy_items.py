"""多检测器组合的打码策略优先级与批次目录回归测试。

覆盖：
- ``--blackout`` / ``--mosaic`` 显式指定时，必须覆盖 YAML 里的打码策略，
  而不是被 YAML 静默盖掉（用户要纯黑却拿到马赛克，无任何提示）；
- supervision 模式不落批次目录，导致下游 ``merge.py`` 处理到旧批次。
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image


def _write_detectors_cfg(tmp_path: Path, *, blackout: bool | None) -> Path:
    """写一份单路 yolo 的 detectors 配置。"""
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    Image.new("RGB", (100, 80), color="white").save(src / "a.png")
    cfg = {
        "source": str(src),
        "output": str(tmp_path / "out"),
        "detectors": [
            {"type": "yolo", "name": "y", "model": "yolov8n.pt"},
        ],
    }
    if blackout is not None:
        cfg["blackout"] = blackout
    path = tmp_path / "detectors.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def _resolve_use_mosaic(cfg: dict, args) -> bool:
    """复现 multi.py 的打码策略解析逻辑（提取自 run_multi）。"""
    from tools.annotate.runners.multi import resolve_use_mosaic

    return resolve_use_mosaic(cfg, args)


class _Args:
    """模拟 argparse 结果：--blackout / --mosaic 默认都是 False。"""

    def __init__(self, blackout=False, mosaic=False):
        self.blackout = blackout
        self.mosaic = mosaic


# ---------------------------------------------------------------------------
# 遗留项 1：CLI 打码策略被 YAML 忽略
# ---------------------------------------------------------------------------


def test_cli_blackout_overrides_yaml_default() -> None:
    """CLI 显式 --blackout 时，即使 YAML 没写 blackout 也要用纯黑。"""
    cfg = {}  # YAML 未指定
    assert _resolve_use_mosaic(cfg, _Args(blackout=True)) is False, (
        "--blackout 被忽略，仍在用马赛克"
    )


def test_cli_mosaic_overrides_yaml_blackout() -> None:
    """CLI 显式 --mosaic 时，即使 YAML 写了 blackout: true 也要用马赛克。"""
    cfg = {"blackout": True}
    assert _resolve_use_mosaic(cfg, _Args(mosaic=True)) is True, (
        "--mosaic 被 YAML 的 blackout: true 盖掉"
    )


def test_yaml_blackout_used_when_cli_silent() -> None:
    """CLI 未指定时，采用 YAML 的策略（守护）。"""
    assert _resolve_use_mosaic({"blackout": True}, _Args()) is False
    assert _resolve_use_mosaic({"blackout": False}, _Args()) is True


def test_default_is_mosaic_when_nothing_specified() -> None:
    """两边都没指定时默认马赛克（守护）。"""
    assert _resolve_use_mosaic({}, _Args()) is True


def test_cli_blackout_and_mosaic_conflict_raises() -> None:
    """同时指定 --blackout 与 --mosaic 是矛盾输入，应报错。"""
    with pytest.raises(ValueError, match="不能同时"):
        _resolve_use_mosaic({}, _Args(blackout=True, mosaic=True))


# ---------------------------------------------------------------------------
# 遗留项 2：supervision 不落批次目录
# ---------------------------------------------------------------------------


def test_supervision_writes_into_batch_dir(tmp_path: Path) -> None:
    """supervision 输出应落进 batch_* 批次目录，与其它 runner 一致。

    否则 merge.py 的 resolve_latest_batch_stage_dir 会解析到历史批次，
    把新标注整个跳过。
    """
    from tools.annotate.runners.supervision import resolve_supervision_output

    logical = tmp_path / "01_annotated"
    resolved = resolve_supervision_output(logical)

    assert resolved.parent.name.startswith("batch_"), (
        f"输出没落进批次目录：{resolved}"
    )
    assert resolved.name == "01_annotated"
