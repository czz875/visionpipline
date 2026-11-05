"""``--format yolo`` 导出结构的回归测试。

缺陷：supervision 把标签写到 ``<output>/labels/``，而 ``yolo_to_labelme.py``
按 ``<labels>/<图片 stem>.txt`` 找标签 —— 差一层目录，回转后全部标注丢失
却仍报告成功。导出的 data.yaml 也只有 nc/names，ultralytics 拒绝加载。
"""

import json
from pathlib import Path

import pytest
import yaml

from tools.annotate.runners.supervision import export_detection_dataset


class _FakeDataset:
    """最小 sv.DetectionDataset 替身，只实现导出所需接口。"""

    def __init__(self, images: list[str], classes: list[str], boxes_per_image: int = 1):
        self.classes = classes
        self.image_paths = list(images)  # 真实 sv.DetectionDataset 也暴露此属性
        self._images = images
        self._boxes = boxes_per_image

    def as_yolo(self, images_directory_path=None, annotations_directory_path=None,
                data_yaml_path=None):
        """按 supervision 的真实行为写标签到 annotations_directory_path。"""
        ann = Path(annotations_directory_path)
        ann.mkdir(parents=True, exist_ok=True)
        for img in self._images:
            stem = Path(img).stem
            lines = [f"0 0.30000 0.30000 0.40000 0.40000"] * self._boxes
            (ann / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if images_directory_path:
            dst = Path(images_directory_path)
            dst.mkdir(parents=True, exist_ok=True)
            for img in self._images:
                (dst / Path(img).name).write_bytes(b"fake")
        if data_yaml_path:
            # supervision 的 save_data_yaml 只写 nc/names
            Path(data_yaml_path).write_text(
                f"nc: {len(self.classes)}\nnames: [{', '.join(self.classes)}]\n",
                encoding="utf-8",
            )

    def as_coco(self, images_directory_path=None, annotations_path=None):
        Path(annotations_path).write_text(json.dumps({"images": []}), encoding="utf-8")


def test_yolo_export_layout_matches_yolo_to_labelme_expectation(tmp_path: Path) -> None:
    """导出后 ``<output>/labels/<stem>.txt`` 必须能被 yolo_to_labelme 找到。"""
    src = tmp_path / "src"
    src.mkdir()
    image = src / "a.png"
    image.write_bytes(b"fake")
    out = tmp_path / "out"

    dataset = _FakeDataset([str(image)], ["phone", "face"])
    export_detection_dataset(dataset, out, "yolo", copy_images=True)

    # yolo_to_labelme 用 --labels 指向 <output>，然后找 <output>/<stem>.txt
    assert (out / "a.txt").is_file(), (
        "标签必须写在 <output>/ 顶层，否则 yolo_to_labelme 按 <labels>/<stem>.txt 找不到"
    )


def test_yolo_export_data_yaml_is_loadable(tmp_path: Path) -> None:
    """data.yaml 必须含 path/train/val，否则 ultralytics 报 'train:' key missing。"""
    src = tmp_path / "src"
    src.mkdir()
    image = src / "a.png"
    image.write_bytes(b"fake")
    out = tmp_path / "out"

    dataset = _FakeDataset([str(image)], ["phone", "face"])
    export_detection_dataset(dataset, out, "yolo", copy_images=True)

    text = (out / "data.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert "path" in data, "data.yaml 缺 path"
    assert "train" in data, "data.yaml 缺 train"
    assert "val" in data, "data.yaml 缺 val"
    assert data["nc"] == 2
    assert data["names"] == ["phone", "face"]


def test_labelme_export_keeps_case_only_names_as_distinct_pairs(tmp_path: Path) -> None:
    import supervision as sv
    from PIL import Image

    paths = []
    for folder, name, color in [("a", "X.png", "red"), ("b", "x.png", "blue")]:
        parent = tmp_path / folder
        parent.mkdir()
        image = parent / name
        Image.new("RGB", (20, 20), color=color).save(image)
        paths.append(str(image))
    dataset = sv.DetectionDataset(
        classes=["face"], images=paths, annotations={p: sv.Detections.empty() for p in paths},
    )
    output = tmp_path / "output"

    export_detection_dataset(dataset, output, "labelme", copy_images=True)

    jsons = list(output.glob("*.json"))
    assert len(jsons) == 2
    image_paths = [output / json.loads(p.read_text(encoding="utf-8"))["imagePath"] for p in jsons]
    assert len(set(image_paths)) == 2
    assert {p.read_bytes() for p in image_paths} == {Path(p).read_bytes() for p in paths}
