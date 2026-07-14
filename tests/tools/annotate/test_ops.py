"""
tools/annotate/ops.py 的纯函数单测：框几何与打码。

这些函数不依赖具体模型/类别，是适合单测的纯逻辑。
环境缺少 numpy/opencv 时整体跳过，避免误报。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 允许以 `python tests/tools/annotate/test_ops.py` 直接运行 / pytest 发现。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
from tools.annotate.ops import (  # noqa: E402
    blackout_region,
    build_labelme_shapes,
    classify_box_by_ratio,
    clip_box,
    collect_blackout_regions,
    mosaic_region,
    subtract_box_regions,
)


def test_clip_box_inside_unchanged() -> None:
    """完全在图内的框裁剪后不变。"""
    shape = (100, 200, 3)
    box = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    out = clip_box(box, shape)
    assert out.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_clip_box_out_of_bounds() -> None:
    """越界框被裁剪到图像边界。"""
    shape = (100, 200, 3)
    box = np.array([-5.0, -5.0, 300.0, 300.0], dtype=np.float32)
    out = clip_box(box, shape)
    assert out.tolist() == [0.0, 0.0, 200.0, 100.0]


def test_classify_box_by_ratio_keep() -> None:
    """面积占比达到阈值 -> 保留。"""
    shape = (100, 200, 3)  # 面积 20000
    box = np.array([0.0, 0.0, 40.0, 50.0], dtype=np.float32)  # 面积 2000 -> 0.1
    assert classify_box_by_ratio(box, shape, 0.01) is True


def test_classify_box_by_ratio_drop() -> None:
    """面积占比低于阈值 -> 不保留。"""
    shape = (100, 200, 3)
    box = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)  # 面积 25 -> 0.00125
    assert classify_box_by_ratio(box, shape, 0.01) is False


def test_classify_box_by_ratio_zero_area() -> None:
    """零面积图像直接判否，避免除零。"""
    shape = (0, 0, 3)
    box = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
    assert classify_box_by_ratio(box, shape, 0.01) is False


def test_subtract_box_regions_fully_inside() -> None:
    """小框完全落在大框内 -> 无残余区域。"""
    box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    keep = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
    out = subtract_box_regions(box, keep)
    assert out.shape == (0, 4)


def test_subtract_box_regions_partial() -> None:
    """小框与保留大框部分重叠 -> 仅剩不重叠的左侧残余。"""
    box = np.array([0.0, 0.0, 100.0, 50.0], dtype=np.float32)
    keep = np.array([[50.0, 0.0, 100.0, 100.0]], dtype=np.float32)
    out = subtract_box_regions(box, keep)
    assert out.shape[0] == 1
    # 残余应为左侧条形 [0,0,50,50]
    assert out[0].tolist() == [0.0, 0.0, 50.0, 50.0]


def test_collect_blackout_regions_no_keep() -> None:
    """没有保留框时，所有待删除框原样返回。"""
    removed = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
    out = collect_blackout_regions(removed, np.empty((0, 4), dtype=np.float32))
    assert out.shape == (2, 4)


def test_collect_blackout_regions_protected() -> None:
    """与保留框重叠的小框被整体保护（无残余）。"""
    removed = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    keep = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
    out = collect_blackout_regions(removed, keep)
    assert out.shape == (0, 4)


def test_blackout_region_zeroes_roi() -> None:
    """纯黑打码把框内区域置零。"""
    image = np.full((50, 50, 3), 200, dtype=np.uint8)
    image = blackout_region(image, np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32))
    assert int(image[0:10, 0:10].max()) == 0
    # 框外不受影响
    assert int(image[20, 20].max()) == 200


def test_mosaic_region_keeps_shape_dtype() -> None:
    """马赛克不改变图像尺寸与 dtype；纯色区域马赛克后仍是纯色。"""
    image = np.full((50, 50, 3), 100, dtype=np.uint8)
    out = mosaic_region(image, np.array([0.0, 0.0, 20.0, 20.0], dtype=np.float32), block_size=4)
    assert out.shape == (50, 50, 3)
    assert out.dtype == np.uint8
    assert int(out[0:20, 0:20].min()) == 100  # 纯色输入马赛克后不变


def test_build_labelme_shapes_alignment() -> None:
    """框与标签逐框对齐成 LabelMe shape 字典。"""
    boxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32
    )
    shapes = build_labelme_shapes(boxes, ["a", "b"])
    assert len(shapes) == 2
    assert shapes[0]["label"] == "a"
    assert shapes[1]["label"] == "b"
    assert shapes[0]["shape_type"] == "rectangle"
    assert shapes[0]["points"] == [[0.0, 0.0], [10.0, 10.0]]
