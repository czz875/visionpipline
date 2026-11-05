"""YOLO → LabelMe 的类别解析回归测试。"""

from pathlib import Path

import pytest
import yaml

from tools.convert.yolo_to_labelme import parse_classes


def test_converted_pairs_survive_final_orphan_cleanup(tmp_path: Path, monkeypatch) -> None:
    """回转必须复制图片，否则主工作流的 final_clean 会删掉全部 JSON。"""
    import json
    from PIL import Image
    from tools.convert.yolo_to_labelme import convert_yolo_to_labelme
    from tools.clean.orphan_json import main as clean_orphans

    images, labels, output = tmp_path / "images", tmp_path / "labels", tmp_path / "final"
    images.mkdir()
    labels.mkdir()
    for folder in ("a", "b"):
        (images / folder).mkdir()
        Image.new("RGB", (20, 20), color=folder == "a" and "red" or "blue").save(
            images / folder / "sample.png"
        )
    (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    assert convert_yolo_to_labelme(images, labels, ["phone"], output) == 2
    monkeypatch.setattr("sys.argv", [
        "orphan_json", "--dataset-path", str(output), "--recursive", "--apply",
    ])
    assert clean_orphans() == 0
    jsons = sorted(output.glob("*.json"))
    assert len(jsons) == 2
    assert len(list(output.glob("*.png"))) == 2
    for annotation in jsons:
        data = json.loads(annotation.read_text(encoding="utf-8"))
        assert (output / data["imagePath"]).is_file()
        assert len(data["shapes"]) == 1


def test_output_cannot_overlap_source_images(tmp_path: Path) -> None:
    """平铺复制不能覆盖输入目录里尚未处理的同名图片。"""
    from PIL import Image
    from tools.convert.yolo_to_labelme import convert_yolo_to_labelme

    images, labels = tmp_path / "images", tmp_path / "labels"
    (images / "a").mkdir(parents=True)
    labels.mkdir()
    Image.new("RGB", (20, 20), color="red").save(images / "a" / "sample.png")
    Image.new("RGB", (20, 20), color="blue").save(images / "sample.png")
    (labels / "sample.txt").write_text("", encoding="utf-8")
    before = (images / "sample.png").read_bytes()

    with pytest.raises(ValueError, match="重叠"):
        convert_yolo_to_labelme(images, labels, ["phone"], images)

    assert (images / "sample.png").read_bytes() == before


def test_nonempty_output_is_preserved_before_conversion(tmp_path: Path) -> None:
    from PIL import Image
    from tools.convert.yolo_to_labelme import convert_yolo_to_labelme

    images, labels, output = (tmp_path / name for name in ("images", "labels", "output"))
    for folder in (images, labels, output):
        folder.mkdir()
    Image.new("RGB", (20, 20), color="red").save(images / "sample.png")
    (labels / "sample.txt").write_text("", encoding="utf-8")
    existing = output / "sample.png"
    existing.write_bytes(b"existing content, possibly shared by hardlinks")

    with pytest.raises(ValueError, match="非空"):
        convert_yolo_to_labelme(images, labels, ["phone"], output)

    assert existing.read_bytes() == b"existing content, possibly shared by hardlinks"
    assert not (output / "sample.json").exists()


def test_case_only_stems_keep_both_output_images(tmp_path: Path) -> None:
    import json
    from PIL import Image
    from tools.convert.yolo_to_labelme import convert_yolo_to_labelme

    images, labels, output = tmp_path / "images", tmp_path / "labels", tmp_path / "output"
    labels.mkdir()
    for folder, stem, color in (("a", "X", "red"), ("b", "x", "blue")):
        parent = images / folder
        parent.mkdir(parents=True)
        Image.new("RGB", (20, 20), color=color).save(parent / f"{stem}.png")
        (labels / f"{stem}.txt").write_text("", encoding="utf-8")

    assert convert_yolo_to_labelme(images, labels, ["phone"], output) == 2
    jsons = list(output.glob("*.json"))
    assert len(jsons) == 2
    image_paths = [output / json.loads(path.read_text(encoding="utf-8"))["imagePath"] for path in jsons]
    assert len(set(image_paths)) == 2
    assert {path.read_bytes() for path in image_paths} == {
        (images / "a" / "X.png").read_bytes(), (images / "b" / "x.png").read_bytes(),
    }


def test_parse_classes_reads_names_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"nc": 2, "names": ["phone", "cigarette"]}, allow_unicode=True),
        encoding="utf-8",
    )

    assert parse_classes(str(yaml_path)) == ["phone", "cigarette"]


def test_parse_classes_reads_dict_names_in_id_order(tmp_path: Path) -> None:
    """names 写成 dict 时按数字 key 排序，不能按字符串排序（10 会排在 2 前面）。"""
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"names": {0: "a", 2: "c", 10: "k"}}, allow_unicode=True),
        encoding="utf-8",
    )

    assert parse_classes(str(yaml_path)) == ["a", "c", "k"]


def test_parse_classes_rejects_missing_yaml_instead_of_using_path_as_label(
    tmp_path: Path,
) -> None:
    """路径形似 YAML 但不存在时必须报错，不能把路径本身当成类名。"""
    missing = tmp_path / "not_exist.yaml"

    with pytest.raises(FileNotFoundError, match="不存在"):
        parse_classes(str(missing))


def test_parse_classes_rejects_yaml_without_names(tmp_path: Path) -> None:
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({"nc": 0}), encoding="utf-8")

    with pytest.raises(ValueError, match="names"):
        parse_classes(str(yaml_path))


def test_parse_classes_splits_plain_comma_list() -> None:
    assert parse_classes("phone, cigarette") == ["phone", "cigarette"]
