"""框合并核心算法的回归测试。

``tools/core/geometry.py`` 的 ``merge_near_boxes`` 是「合并邻近同标签矩形框」
阶段的核心算法，此前**零测试**。这里把当前行为锁死，防止后续改动悄悄改变
合并结果——合并结果直接决定送进训练集的标注框。

相邻判定当前用的是 OR（x 方向或 y 方向的距离小于等于对应阈值即算相邻），
见 ``geometry.py`` 里 ``dist_x <= distance_x or dist_y <= distance_y``。
这个语义是否符合业务预期需要确认；在确认之前，测试按**现状**锁定，
避免有人无意中改成 AND 后无人察觉。
"""

from __future__ import annotations

from tools.core.geometry import get_boxes_dist, merge_near_boxes, rect_to_xyxy


def test_empty_input_returns_empty_list() -> None:
    assert merge_near_boxes([], distance_x=100, distance_y=200) == []


def test_boxes_far_apart_in_both_axes_are_not_merged() -> None:
    """x、y 两个方向都超过阈值时不该合并。"""
    boxes = [(0, 0, 50, 50), (500, 500, 550, 550)]

    merged = merge_near_boxes(boxes, distance_x=100, distance_y=200)

    assert len(merged) == 2


def test_boxes_close_in_x_axis_are_merged() -> None:
    """x 方向够近时合并（当前 OR 语义）。"""
    boxes = [(0, 0, 50, 50), (60, 10, 110, 60)]

    merged = merge_near_boxes(boxes, distance_x=100, distance_y=200)

    assert len(merged) == 1
    assert merged[0] == (0, 0, 110, 60)


def test_merged_box_is_the_union_of_its_parts() -> None:
    """合并后的框必须等于子框的并集。"""
    boxes = [(10, 10, 60, 60), (70, 20, 120, 70)]

    merged = merge_near_boxes(boxes, distance_x=100, distance_y=200)

    assert len(merged) == 1
    x1, y1, x2, y2 = merged[0]
    assert x1 == min(b[0] for b in boxes)
    assert y1 == min(b[1] for b in boxes)
    assert x2 == max(b[2] for b in boxes)
    assert y2 == max(b[3] for b in boxes)


def test_max_width_prevents_oversized_merge() -> None:
    """合并后宽度超限就放弃这次合并，两个框保持独立。"""
    boxes = [(0, 0, 100, 100), (150, 0, 400, 100)]

    merged = merge_near_boxes(
        boxes, distance_x=200, distance_y=200, max_width=250, max_height=600
    )

    assert len(merged) == 2


def test_max_height_prevents_oversized_merge() -> None:
    """合并后高度超限就放弃这次合并。"""
    boxes = [(0, 0, 50, 50), (0, 100, 50, 400)]

    merged = merge_near_boxes(
        boxes, distance_x=600, distance_y=600, max_width=600, max_height=250
    )

    assert len(merged) == 2


def test_max_count_limits_participation() -> None:
    """单个合并框参与合并的次数达到上限后不再吸收新框。

    计数在框加入时就从 1 起算，因此 ``max_count=1`` 等于「完全不允许合并」，
    ``max_count=2`` 才允许每个框吸收 1 个新框。
    """
    boxes = [(0, 0, 50, 50), (10, 0, 60, 50), (20, 0, 70, 50)]

    limited = merge_near_boxes(boxes, distance_x=100, distance_y=200, max_count=1)
    assert len(limited) == 3, "max_count=1 时不应发生任何合并"

    allowed = merge_near_boxes(boxes, distance_x=100, distance_y=200, max_count=2)
    assert len(allowed) == 2, "max_count=2 时每个框可吸收 1 个新框"


def test_max_count_zero_disables_merging_entirely() -> None:
    """``max_count=0`` 与 1 一样，等于关闭合并。"""
    boxes = [(0, 0, 50, 50), (10, 0, 60, 50)]

    assert len(merge_near_boxes(boxes, distance_x=100, distance_y=200, max_count=0)) == 2


def test_unlimited_max_count_merges_all() -> None:
    """``max_count=None`` 表示不限次数。"""
    boxes = [(0, 0, 50, 50), (10, 0, 60, 50), (20, 0, 70, 50)]

    merged = merge_near_boxes(boxes, distance_x=100, distance_y=200, max_count=None)

    assert len(merged) == 1


def test_input_order_does_not_change_result() -> None:
    """输入顺序打乱后，输出集合应当一致（算法按 x 排序，结果应确定）。"""
    boxes = [(0, 0, 50, 50), (60, 10, 110, 60), (200, 200, 250, 250)]

    forward = merge_near_boxes(boxes, distance_x=100, distance_y=200)
    backward = merge_near_boxes(list(reversed(boxes)), distance_x=100, distance_y=200)

    assert forward == backward


def test_get_boxes_dist_returns_gap_on_each_axis() -> None:
    """两个框在 x、y 上的间隔应分别算出。"""
    dist_x, dist_y = get_boxes_dist((0, 0, 10, 10), (20, 30, 40, 40))

    assert dist_x == 10
    assert dist_y == 20


def test_get_boxes_dist_is_zero_for_overlapping_boxes() -> None:
    """重叠的框在两个方向上的间隔都应为 0。"""
    dist_x, dist_y = get_boxes_dist((0, 0, 100, 100), (50, 50, 150, 150))

    assert dist_x == 0
    assert dist_y == 0


def test_rect_to_xyxy_normalizes_swapped_coordinates() -> None:
    """points 顺序颠倒时应归一化成左上-右下。"""
    x1, y1, x2, y2 = rect_to_xyxy([[100, 200], [10, 20]])

    assert (x1, y1, x2, y2) == (10, 20, 100, 200)
