"""``--format yolo`` 同名图片覆盖检测的回归测试。

supervision 的 ``as_yolo`` 按 basename 命名标签文件，不同子目录的同名图片会
互相覆盖、静默丢标注，且报告的数量仍是全量。这里要求提前报错。
"""

from pathlib import Path

import pytest

from tools.annotate.runners.supervision import _check_yolo_basename_collisions


class _FakeDataset:
    def __init__(self, paths: list[str]):
        self.image_paths = paths


def test_collisions_raise() -> None:
    """同名图片必须报错，不能静默覆盖。"""
    dataset = _FakeDataset(["/a/shot.png", "/b/shot.png"])
    with pytest.raises(ValueError, match="同名"):
        _check_yolo_basename_collisions(dataset)


def test_unique_names_pass() -> None:
    """名称各不相同则正常通过（守护）。"""
    dataset = _FakeDataset(["/a/one.png", "/b/two.png"])
    _check_yolo_basename_collisions(dataset)


def test_same_path_repeated_is_not_collision() -> None:
    """同一路径重复出现不算冲突（守护）。"""
    dataset = _FakeDataset(["/a/one.png", "/a/one.png"])
    _check_yolo_basename_collisions(dataset)
