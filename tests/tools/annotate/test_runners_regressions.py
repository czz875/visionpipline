"""标注 runner 的回归测试：类别解析与覆盖模式取框。"""

import numpy as np

from tools.annotate.ops import group_rect_boxes_by_label
from tools.annotate.runners.supervision import parse_class_ids


def test_parse_class_ids_accepts_labeler_classes_list() -> None:
    """AutoLabeler.classes 是 list[str]，调用方按列表传入不能崩。"""
    assert parse_class_ids("person,car", ["person", "car", "bus"]) == {0, 1}


def test_parse_class_ids_accepts_model_names_dict() -> None:
    assert parse_class_ids("person,car", {0: "person", 1: "car"}) == {0, 1}


def test_parse_class_ids_supports_numeric_ids() -> None:
    assert parse_class_ids("0,2", ["person", "car", "bus"]) == {0, 2}


def test_parse_class_ids_returns_none_when_unfiltered() -> None:
    assert parse_class_ids(None, ["person"]) is None
    assert parse_class_ids("", ["person"]) is None


def test_parse_class_ids_rejects_unknown_names() -> None:
    """拼错的类别名必须报错。

    原先只打印警告并返回 {0}，``--classes fac``（拼错）会退化成「不过滤」，
    把不该标注的类别一并写进结果，而用户以为过滤已生效。
    """
    import pytest

    with pytest.raises(ValueError, match="未识别的类别"):
        parse_class_ids("person,unknown", ["person"])


def test_group_rect_boxes_by_label_groups_all_labels_in_one_pass() -> None:
    data = {
        "shapes": [
            {"label": "face", "shape_type": "rectangle", "points": [[10, 10], [50, 50]]},
            {"label": "hand", "shape_type": "rectangle", "points": [[60, 20], [20, 80]]},
            {"label": "face", "shape_type": "rectangle", "points": [[100, 100], [120, 140]]},
            {"label": "face", "shape_type": "polygon", "points": [[0, 0], [5, 5], [10, 0]]},
            {"label": "hand", "shape_type": "rectangle", "points": [[1, 1]]},
        ]
    }

    grouped = group_rect_boxes_by_label(data)

    assert set(grouped) == {"face", "hand"}
    assert grouped["face"].shape == (2, 4)
    # 两点矩形顺序颠倒时按 xyxy 归一化
    assert grouped["hand"].tolist() == [[20.0, 20.0, 60.0, 80.0]]
    assert grouped["face"].dtype == np.float32


def test_group_rect_boxes_by_label_returns_empty_for_no_shapes() -> None:
    assert group_rect_boxes_by_label({}) == {}
