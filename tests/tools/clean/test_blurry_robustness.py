"""清洗脚本的健壮性回归测试。

覆盖两类「不报错但结果不对」的缺陷：
- ``tools/clean/blurry.py`` 在数据集没有 LabelMe JSON 时会崩在
  ``crop_path.iterdir()``（目录压根没建出来）；
- ``tools/clean/blurry.py`` 移出模糊图时会把不同子目录的同名图片平铺覆盖，
  且不搬同名 JSON，留下孤儿。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.clean.blurry import _find_blurry_originals, _move_bad_images


def test_find_blurry_originals_handles_missing_crop_dir(tmp_path: Path) -> None:
    """裁剪目录不存在时必须返回空集合，而不是抛 FileNotFoundError。"""
    missing = tmp_path / "not_created"
    assert not missing.exists()

    assert _find_blurry_originals(missing, 0.18) == set()


def test_move_bad_images_preserves_subdirs_and_moves_json(tmp_path: Path) -> None:
    """不同子目录下的同名图片不能互相覆盖，同名 JSON 要一起移走。"""
    dataset = tmp_path / "ds"
    (dataset / "a").mkdir(parents=True)
    (dataset / "b").mkdir(parents=True)
    (dataset / "a" / "same.png").write_bytes(b"AAA")
    (dataset / "a" / "same.json").write_text("{}", encoding="utf-8")
    (dataset / "b" / "same.png").write_bytes(b"BBB")
    (dataset / "b" / "same.json").write_text("{}", encoding="utf-8")

    out = tmp_path / "blurry"
    moved = _move_bad_images(dataset, out, {"same"})

    assert moved == 2
    # 两张图都保住了，且内容没有被对方覆盖
    assert (out / "a" / "same.png").read_bytes() == b"AAA"
    assert (out / "b" / "same.png").read_bytes() == b"BBB"
    # 同名 JSON 跟着图片走，原目录不留孤儿
    assert (out / "a" / "same.json").is_file()
    assert (out / "b" / "same.json").is_file()
    assert not list(dataset.rglob("*.json"))


def test_move_bad_images_leaves_clean_images_alone(tmp_path: Path) -> None:
    """不在 bad_stems 里的图片必须原样留在原目录。"""
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "keep.png").write_bytes(b"K")
    (dataset / "drop.png").write_bytes(b"D")

    out = tmp_path / "blurry"
    moved = _move_bad_images(dataset, out, {"drop"})

    assert moved == 1
    assert (dataset / "keep.png").is_file()
    assert not (dataset / "drop.png").exists()
    assert (out / "drop.png").is_file()
