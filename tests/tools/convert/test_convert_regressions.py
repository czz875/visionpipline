"""LabelMe → YOLO 转换的回归测试。

覆盖审查发现的四类静默数据损坏：
- JSON 缺 ``imageWidth/Height`` 时不能按 1 处理（否则坐标全错）；
- 越界坐标必须裁剪到图片范围内；
- 未注册标签必须被报告，不能静默丢弃；
- ``--seed`` 必须真正影响划分结果，``--ratios`` 必须被接受。
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.convert.labelme_to_yolo import (
    convert_labelme_to_yolo,
    convert_to_yolo,
    parse_ratios,
    process_batch,
    split_files,
)


def _write_sample(
    json_path: Path,
    image_path: Path,
    *,
    label: str = "phone",
    points: list[list[float]] | None = None,
    size: tuple[int, int] = (100, 80),
    write_size: bool = True,
) -> None:
    """写一张白图 + 对应 LabelMe JSON。"""
    Image.new("RGB", size, color="white").save(image_path)
    data: dict = {
        "shapes": [
            {
                "label": label,
                "shape_type": "rectangle",
                "points": points if points is not None else [[10, 20], [50, 60]],
            }
        ],
    }
    if write_size:
        data["imageWidth"], data["imageHeight"] = size
    json_path.write_text(json.dumps(data), encoding="utf-8")


def test_missing_image_size_falls_back_to_real_image_size(tmp_path: Path) -> None:
    """JSON 缺 imageWidth/Height 时按真实图片尺寸归一化，而不是按 1。"""
    json_path = tmp_path / "a.json"
    image_path = tmp_path / "a.png"
    _write_sample(json_path, image_path, write_size=False)
    label_path = tmp_path / "a.txt"

    convert_labelme_to_yolo(json_path, label_path, class_names=["phone"])

    line = label_path.read_text(encoding="utf-8").strip()
    # 框 (10,20)-(50,60) 在 100x80 图上：cx=0.3 cy=0.5 w=0.4 h=0.5
    assert line == "0 0.30000 0.50000 0.40000 0.50000"


def test_out_of_bounds_points_are_clipped(tmp_path: Path) -> None:
    """越界坐标必须裁剪到 [0,W]/[0,H]，不能写出 >1 的归一化值。"""
    json_path = tmp_path / "a.json"
    image_path = tmp_path / "a.png"
    _write_sample(json_path, image_path, points=[[-50, -10], [150, 120]])
    label_path = tmp_path / "a.txt"

    convert_labelme_to_yolo(json_path, label_path, class_names=["phone"])

    line = label_path.read_text(encoding="utf-8").strip()
    assert line == "0 0.50000 0.50000 1.00000 1.00000"


def test_fully_out_of_bounds_box_is_dropped(tmp_path: Path) -> None:
    """完全落在图片外的框裁剪后为空，应被丢弃而不是写成零面积标签。"""
    json_path = tmp_path / "a.json"
    image_path = tmp_path / "a.png"
    _write_sample(json_path, image_path, points=[[200, 200], [300, 300]])
    label_path = tmp_path / "a.txt"

    convert_labelme_to_yolo(json_path, label_path, class_names=["phone"])

    assert label_path.read_text(encoding="utf-8").strip() == ""


def test_unregistered_labels_are_reported(tmp_path: Path) -> None:
    """未注册标签要作为返回值上报，供调用方汇总警告。"""
    json_path = tmp_path / "a.json"
    image_path = tmp_path / "a.png"
    _write_sample(json_path, image_path, label="helmet")
    label_path = tmp_path / "a.txt"

    dropped = convert_labelme_to_yolo(json_path, label_path, class_names=["phone"])

    assert dropped == ["helmet"]
    assert label_path.read_text(encoding="utf-8").strip() == ""


def test_process_batch_writes_trainable_data_yaml(tmp_path: Path) -> None:
    """data.yaml 必须含 path/train/val，否则 yolo detect train 无法加载。"""
    source = tmp_path / "source"
    source.mkdir()
    json_path = source / "a.json"
    _write_sample(json_path, source / "a.png")
    output = tmp_path / "output"

    process_batch(
        "0001",
        [json_path],
        output,
        seed=3407,
        class_names=["phone", "cigarette"],
        force=True,
    )

    text = (output / "data.yaml").read_text(encoding="utf-8")
    assert "path:" in text
    assert "train: train.txt" in text
    assert "val: val.txt" in text
    assert "nc: 2" in text
    assert "names: [phone, cigarette]" in text
    # train.txt / val.txt 必须真实存在，否则 ultralytics 仍会报错
    assert (output / "train.txt").is_file()
    assert (output / "val.txt").is_file()


def test_seed_changes_split_and_is_reproducible(tmp_path: Path) -> None:
    """--seed 必须真正影响划分，且同一 seed 结果可复现。"""
    files = [tmp_path / f"{i}.json" for i in range(10)]

    a1 = split_files(files, seed=1, ratios=[0.5, 0.5])
    a2 = split_files(files, seed=1, ratios=[0.5, 0.5])
    b = split_files(files, seed=2, ratios=[0.5, 0.5])

    assert a1 == a2
    assert a1 != b


def test_convert_to_yolo_honours_seed_argument(tmp_path: Path) -> None:
    """convert_to_yolo 传入的 seed 不能被内部常量覆盖。"""
    source = tmp_path / "source" / "0001"
    source.mkdir(parents=True)
    for i in range(10):
        _write_sample(source / f"{i}.json", source / f"{i}.png")

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    convert_to_yolo(source.parent, out_a, class_names=["phone"], seed=1)
    convert_to_yolo(source.parent, out_b, class_names=["phone"], seed=999)

    train_a = (out_a / "0001" / "train.txt").read_text(encoding="utf-8")
    train_b = (out_b / "0001" / "train.txt").read_text(encoding="utf-8")
    assert train_a != train_b


def test_parse_ratios_accepts_config_value_and_rejects_bad_input() -> None:
    """tools/cfg/labelme_to_yolo.yaml 里的 "0.9,0.1" 必须被接受。"""
    assert parse_ratios("0.9,0.1") == [0.9, 0.1]
    with pytest.raises(ValueError):
        parse_ratios("0.9")  # 数量不足
    with pytest.raises(ValueError):
        parse_ratios("0.5,0.6")  # 之和不为 1
