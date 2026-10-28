"""``merge.py`` 目录解析的回归测试。

缺陷：``merge.py`` 用 ``resolve_latest_batch_stage_dir`` 把逻辑目录解析到
「父目录下最新的 batch_*」。当上游直接写逻辑目录（如 ``01_annotated``）、
历史批次仍存在时，merge 会去处理**旧批次**，新标注完全被跳过，却报告成功。
"""

import json
from pathlib import Path

from PIL import Image

from tools.annotate.merge import _resolve_json_dir


def _make_annotation(dir_path: Path, stem: str) -> None:
    """在 dir_path 下造一对同名 PNG + LabelMe JSON。"""
    dir_path.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color="white").save(dir_path / f"{stem}.png")
    data = {
        "shapes": [
            {"label": "face", "shape_type": "rectangle", "points": [[10, 10], [60, 60]]}
        ],
        "imagePath": f"{stem}.png",
        "imageWidth": 100,
        "imageHeight": 80,
    }
    (dir_path / f"{stem}.json").write_text(json.dumps(data), encoding="utf-8")


def test_prefers_logical_dir_when_it_has_json(tmp_path: Path) -> None:
    """逻辑目录自身有 JSON 时必须处理它，不能被历史批次抢走。"""
    logical = tmp_path / "01_annotated"
    _make_annotation(logical, "new_one")
    _make_annotation(tmp_path / "batch_20260101_000000" / "01_annotated", "old_one")

    resolved = _resolve_json_dir(logical)

    assert resolved == logical, (
        f"逻辑目录有数据时应处理它，实际解析到 {resolved}"
    )


def test_falls_back_to_latest_batch_when_logical_empty(tmp_path: Path) -> None:
    """逻辑目录不存在或为空时，仍应回退到最新批次（保留原行为）。"""
    _make_annotation(tmp_path / "batch_20260101_000000" / "01_annotated", "old_one")
    _make_annotation(tmp_path / "batch_20260202_000000" / "01_annotated", "newer_one")

    resolved = _resolve_json_dir(tmp_path / "01_annotated")

    assert resolved.name == "01_annotated"
    assert resolved.parent.name == "batch_20260202_000000", (
        f"应回退到最新批次，实际 {resolved}"
    )


def test_returns_logical_when_nothing_exists(tmp_path: Path) -> None:
    """完全没有数据时返回原路径，交由调用方报「目录不存在」。"""
    logical = tmp_path / "01_annotated"
    assert _resolve_json_dir(logical) == logical
