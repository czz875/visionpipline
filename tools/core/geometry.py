"""
tools/core/geometry.py

边界框等几何工具函数。
"""

from __future__ import annotations

from typing import Iterable, Sequence


def rect_to_xyxy(
    points: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """把 LabelMe rectangle 的 ``[[x1, y1], [x2, y2]]`` 转成 ``xyxy`` 元组。"""
    (x1, y1), (x2, y2) = points
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def xyxy_to_points(box: Sequence[float]) -> list[list[float]]:
    """把 ``xyxy`` 元组转成 LabelMe rectangle 所需的 ``[[x1, y1], [x2, y2]]``。"""
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y2]]


def get_boxes_dist(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float]:
    """计算两个 ``xyxy`` 框在 x、y 方向上的投影距离，重叠时为 0。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dist_x = max(bx1 - ax2, ax1 - bx2, 0)
    dist_y = max(by1 - ay2, ay1 - by2, 0)
    return dist_x, dist_y


def merge_near_boxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    distance_x: float,
    distance_y: float,
    max_width: float | None = None,
    max_height: float | None = None,
    max_count: int | None = None,
) -> list[tuple[float, float, float, float]]:
    """按距离阈值贪婪合并相邻边界框。

    按 x 坐标排序后，每个框尝试合并到第一个满足条件的已有合并框中。
    当 x 方向或 y 方向的距离小于等于对应阈值时认为相邻。
    可通过 ``max_width``、``max_height``、``max_count`` 限制合并尺寸和次数。

    Args:
        boxes: 输入 ``xyxy`` 框列表。
        distance_x: x 方向最大允许间隔。
        distance_y: y 方向最大允许间隔。
        max_width: 合并后框的最大宽度，超出则放弃本次合并。
        max_height: 合并后框的最大高度，超出则放弃本次合并。
        max_count: 单个合并框最多允许参与合并的次数。

    Returns:
        合并后的 ``xyxy`` 框列表。
    """
    if not boxes:
        return []

    merged: list[tuple[float, float, float, float]] = []
    merge_count: list[int] = []

    for box in sorted(boxes, key=lambda b: b[0]):
        merged_flag = False
        for i, m in enumerate(merged):
            dist_x, dist_y = get_boxes_dist(box, m)
            if not (dist_x <= distance_x or dist_y <= distance_y):
                continue

            new_x1 = min(m[0], box[0])
            new_y1 = min(m[1], box[1])
            new_x2 = max(m[2], box[2])
            new_y2 = max(m[3], box[3])

            if max_width is not None and (new_x2 - new_x1) > max_width:
                continue
            if max_height is not None and (new_y2 - new_y1) > max_height:
                continue
            if max_count is not None and merge_count[i] >= max_count:
                continue

            merged[i] = (new_x1, new_y1, new_x2, new_y2)
            merge_count[i] += 1
            merged_flag = True
            break

        if not merged_flag:
            merged.append(box)
            merge_count.append(1)

    return merged
