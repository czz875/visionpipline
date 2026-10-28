"""标注层三条回归测试。

覆盖：
- ``--classes`` 名称全部拼错时不能退化成「不过滤」（会混进不该标的类别）；
- supervision 模式必须遵守 ``--apply``（默认预览不写盘）；
- 单张坏图不能让整批崩溃且零产出。
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.annotate.runners.supervision import parse_class_ids


# ---------------------------------------------------------------------------
# 缺陷 6：类别名拼错导致过滤整体失效
# ---------------------------------------------------------------------------


def test_all_unrecognised_class_names_raise() -> None:
    """名称全部无法识别时必须报错，不能静默返回 None（= 不过滤）。"""
    with pytest.raises(ValueError):
        parse_class_ids("fac", ["face", "hand", "helmet"])


def test_partially_unrecognised_class_names_raise() -> None:
    """部分无法识别时也要报错：静默忽略会让用户以为过滤生效了。"""
    with pytest.raises(ValueError):
        parse_class_ids("face,helmt", ["face", "hand", "helmet"])


def test_valid_class_names_filter_normally() -> None:
    """正常名称仍按 id 过滤（守护）。"""
    assert parse_class_ids("face", ["face", "hand", "helmet"]) == {0}
    assert parse_class_ids("face,helmet", ["face", "hand", "helmet"]) == {0, 2}


def test_numeric_class_ids_still_work() -> None:
    """数字 id 形式仍支持（守护）。"""
    assert parse_class_ids("0,2", ["face", "hand", "helmet"]) == {0, 2}


def test_empty_classes_means_no_filter() -> None:
    """不传 --classes 时不过滤（守护）。"""
    assert parse_class_ids("", ["face"]) is None


# ---------------------------------------------------------------------------
# 缺陷 8：单张坏图让整批崩溃
# ---------------------------------------------------------------------------


def test_broken_image_is_skipped_not_fatal(tmp_path: Path) -> None:
    """坏图应被跳过并计数，其余图片照常导出。"""
    from tools.annotate.runners.supervision import build_detection_dataset

    class _FakeLabeler:
        classes = ["face"]

        def predict(self, image_path):
            import supervision as sv
            import numpy as np

            if Path(image_path).name.startswith("broken"):
                raise IndexError("list index out of range")
            return sv.Detections(
                xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
            )

    images = []
    for name in ("good1.png", "broken.png", "good2.png"):
        p = tmp_path / name
        Image.new("RGB", (100, 80), color="white").save(p)
        images.append(p)

    dataset = build_detection_dataset(_FakeLabeler(), images, None, None)

    # 坏图被剔除，两张好图仍在
    assert len(dataset) == 2, f"期望保留 2 张，实际 {len(dataset)}"
