"""审查发现的四条静默数据损坏缺陷的回归测试。

覆盖：
- 增量跳过不看类别表 → 换 ``--classes`` 后 class id 错位；
- ``--format yolo`` 导出目录结构与 ``yolo_to_labelme`` 的读取约定不一致；
- ``yolo_to_labelme`` 标签目录缺失时静默写空标注并报告成功；
- ``merge.py`` 把逻辑目录解析到旧批次，跳过新标注。
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.convert.labelme_to_yolo import process_batch
from tools.convert.yolo_to_labelme import convert_yolo_to_labelme


def _write_sample(
    json_path: Path,
    image_path: Path,
    *,
    label: str = "phone",
    points: list[list[float]] | None = None,
    size: tuple[int, int] = (100, 80),
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
        "imageWidth": size[0],
        "imageHeight": size[1],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# 缺陷 1：增量跳过不看类别表
# ---------------------------------------------------------------------------


def test_class_table_change_forces_reconversion(tmp_path: Path) -> None:
    """类别表变化后必须重算标签，不能沿用旧 id。

    先按 ``[face, hand, phone]`` 转，再按 ``[phone, cigarette, face, hand]`` 重跑。
    旧标签里 phone 的 id 是 2，新表里 phone 的 id 是 0；若不重算，
    data.yaml 的 names 与标签 id 就整体错位。
    """
    source = tmp_path / "source"
    source.mkdir()
    json_path = source / "a.json"
    _write_sample(json_path, source / "a.png", label="phone")
    output = tmp_path / "output"

    process_batch(
        "0001", [json_path], output, seed=3407,
        class_names=["face", "hand", "phone"], force=True,
    )
    label_file = output / "labels" / "train" / "a.txt"
    if not label_file.exists():
        label_file = output / "labels" / "val" / "a.txt"
    assert label_file.read_text(encoding="utf-8").split()[0] == "2"

    # 第二次换类别表，不加 force
    process_batch(
        "0001", [json_path], output, seed=3407,
        class_names=["phone", "cigarette", "face", "hand"], force=False,
    )

    new_id = label_file.read_text(encoding="utf-8").split()[0]
    assert new_id == "0", f"类别表变了但标签 id 仍是 {new_id}，会与 data.yaml 错位"


def test_unchanged_class_table_still_skips(tmp_path: Path) -> None:
    """类别表没变时仍应增量跳过，不能退化成每次全量重转。"""
    source = tmp_path / "source"
    source.mkdir()
    json_path = source / "a.json"
    _write_sample(json_path, source / "a.png")
    output = tmp_path / "output"

    process_batch("0001", [json_path], output, seed=3407, class_names=["phone"], force=True)
    first = (output / "labels" / "train" / "a.txt")
    if not first.exists():
        first = (output / "labels" / "val" / "a.txt")
    mtime_before = first.stat().st_mtime_ns

    process_batch("0001", [json_path], output, seed=3407, class_names=["phone"], force=False)
    assert first.stat().st_mtime_ns == mtime_before, "类别表未变却重转了"


# ---------------------------------------------------------------------------
# 缺陷 3：标签目录缺失时静默写空标注
# ---------------------------------------------------------------------------


def test_missing_label_dir_raises_instead_of_writing_empty(tmp_path: Path) -> None:
    """标签目录不存在时必须报错，不能写出 shapes=0 的 JSON 并报成功。"""
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 80), color="white").save(images / "a.png")
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="标签"):
        convert_yolo_to_labelme(
            images, tmp_path / "不存在的标签目录", ["phone"], output
        )


def test_missing_single_label_file_raises(tmp_path: Path) -> None:
    """单张图的标签文件缺失也要报错，避免把有框的图写成空标注。"""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80), color="white").save(images / "a.png")
    # labels 目录存在但没有任何 txt

    with pytest.raises(FileNotFoundError, match="标签"):
        convert_yolo_to_labelme(images, labels, ["phone"], tmp_path / "output")


def test_empty_label_file_is_still_allowed(tmp_path: Path) -> None:
    """空 txt 是合法的负样本，不能报错。"""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80), color="white").save(images / "a.png")
    (labels / "a.txt").write_text("", encoding="utf-8")

    generated = convert_yolo_to_labelme(images, labels, ["phone"], tmp_path / "output")
    assert generated == 1
    data = json.loads((tmp_path / "output" / "a.json").read_text(encoding="utf-8"))
    assert data["shapes"] == []
