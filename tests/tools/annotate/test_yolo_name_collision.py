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


@pytest.mark.parametrize("names", [("same.png", "same.jpg"), ("X.png", "x.png")])
def test_yolo_rejects_label_stem_collisions_before_creating_output(tmp_path: Path, names: tuple[str, str]) -> None:
    import numpy as np
    import supervision as sv
    from PIL import Image
    from tools.annotate.runners.supervision import export_detection_dataset

    paths = []
    for i, name in enumerate(names):
        parent = tmp_path / str(i)
        parent.mkdir()
        image = parent / name
        Image.new("RGB", (20, 20), color="red").save(image)
        paths.append(str(image))
    annotations = {path: sv.Detections(
        xyxy=np.array([[1, 1, 10, 10]], dtype=np.float32),
        class_id=np.array([0]), confidence=np.array([1.0]),
    ) for path in paths}
    dataset = sv.DetectionDataset(classes=["face"], images=paths, annotations=annotations)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="同名"):
        export_detection_dataset(dataset, output, "yolo", copy_images=True)
    assert not output.exists()


def test_yolo_collision_check_uses_exported_alias_names(tmp_path: Path) -> None:
    import os

    originals, aliases = tmp_path / "originals", tmp_path / "aliases"
    originals.mkdir()
    paths = []
    for directory, target_name, alias_name in [("a", "p.png", "x.png"), ("b", "q.jpg", "x.jpg")]:
        original = originals / target_name
        original.write_bytes(b"image")
        parent = aliases / directory
        parent.mkdir(parents=True)
        alias = parent / alias_name
        try:
            os.symlink(original, alias)
        except OSError as exc:
            pytest.skip(f"当前环境无法创建测试符号链接：{exc}")
        paths.append(str(alias))

    with pytest.raises(ValueError, match="同名"):
        _check_yolo_basename_collisions(_FakeDataset(paths))
