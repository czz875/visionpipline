"""
tools/annotate/auto.py 中 ``build_source`` 的单测：验证「框 -> 保留/删除 +
逐框标签对齐」逻辑，尤其是多检测器组合里同路多类别、退化框跳过等边界。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 允许以 `python tests/tools/annotate/test_auto_sources.py` 直接运行 / pytest 发现。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

np = pytest.importorskip("numpy")
from tools.annotate.runners.common import BoxSource, build_source  # noqa: E402


def test_build_source_single_label_all_same() -> None:
    """单字符串标签 -> 所有保留框统一标注。"""
    boxes = np.array(
        [[0.0, 0.0, 40.0, 50.0], [0.0, 0.0, 30.0, 30.0]], dtype=np.float32
    )
    src = build_source(boxes, "face", 0.01, (100, 200, 3))
    assert isinstance(src, BoxSource)
    assert len(src.kept) == 2
    assert src.kept_labels == ["face", "face"]


def test_build_source_list_labels_aligned() -> None:
    """列表标签 -> 与框逐框对齐（多类别 detectors 场景）。"""
    boxes = np.array(
        [[0.0, 0.0, 40.0, 50.0], [0.0, 0.0, 30.0, 30.0]], dtype=np.float32
    )
    src = build_source(boxes, ["face", "phone"], 0.01, (100, 200, 3))
    assert src.kept_labels == ["face", "phone"]
    assert len(src.kept) == 2


def test_build_source_skips_degenerate_box() -> None:
    """退化框（越界/负尺寸）被丢弃，且标签随保留框对齐。"""
    boxes = np.array(
        [[0.0, 0.0, 40.0, 50.0], [-5.0, -5.0, -1.0, -1.0]], dtype=np.float32
    )
    src = build_source(boxes, ["face", "phone"], 0.01, (100, 200, 3))
    # 第二框退化被跳过，只保留第一框，其标签应为 face
    assert len(src.kept) == 1
    assert src.kept_labels == ["face"]


def test_build_source_ratio_split() -> None:
    """面积占比低于阈值的框进入 removed，其余进入 kept。"""
    boxes = np.array(
        [[0.0, 0.0, 40.0, 50.0], [0.0, 0.0, 5.0, 5.0]], dtype=np.float32
    )
    src = build_source(boxes, "face", 0.01, (100, 200, 3))
    assert len(src.kept) == 1
    assert len(src.removed) == 1


def test_build_source_clips_out_of_bounds() -> None:
    """越界框被裁剪到范围内后仍然保留（只要仍有效）。"""
    boxes = np.array([[-5.0, -5.0, 300.0, 300.0]], dtype=np.float32)
    src = build_source(boxes, "face", 0.01, (100, 200, 3))
    assert len(src.kept) == 1
    # 裁剪后应为 [0,0,200,100]
    assert src.kept[0].tolist() == [0.0, 0.0, 200.0, 100.0]


def test_build_source_empty() -> None:
    """空输入返回空保留/删除数组。"""
    boxes = np.empty((0, 4), dtype=np.float32)
    src = build_source(boxes, "face", 0.01, (100, 200, 3))
    assert src.kept.shape == (0, 4)
    assert src.removed.shape == (0, 4)
    assert src.kept_labels == []
