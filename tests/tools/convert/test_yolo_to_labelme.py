"""YOLO → LabelMe 的类别解析回归测试。"""

from pathlib import Path

import pytest
import yaml

from tools.convert.yolo_to_labelme import parse_classes


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
