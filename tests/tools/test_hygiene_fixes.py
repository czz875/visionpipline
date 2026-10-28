"""批次四工程卫生改动的回归测试：归档硬链接、IR 增强异常传播、标签替换镜像目录。"""

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from tools.augment.ir_enhance import augment_dataset
from tools.label.replace import process_labelme_file
from tools.train.archive import archive


def _write_labelme(path: Path, label: str = "face") -> None:
    path.write_text(
        json.dumps(
            {
                "imagePath": path.with_suffix(".png").name,
                "imageWidth": 10,
                "imageHeight": 10,
                "shapes": [
                    {
                        "label": label,
                        "shape_type": "rectangle",
                        "points": [[1, 1], [5, 5]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (10, 10), color="white").save(path.with_suffix(".png"))


# ---------------------------------------------------------------------------
# archive：默认硬链接，避免把硬链接数据集实拷一份
# ---------------------------------------------------------------------------


def test_archive_uses_hardlink_by_default(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    image = source / "a.png"
    image.write_bytes(b"payload")
    (source / "a.json").write_text("{}", encoding="utf-8")

    archive_dir = archive([str(source)], tmp_path / "out", tag="t")

    archived_image = archive_dir / "dataset" / "a.png"
    assert archived_image.is_file()
    # 同一卷上硬链接的 inode 相同
    assert archived_image.stat().st_ino == image.stat().st_ino
    metadata = json.loads((archive_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["copy_mode"] == "hardlink"


def test_archive_copy_mode_copies_bytes(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    image = source / "a.png"
    image.write_bytes(b"payload")

    archive_dir = archive(
        [str(source)], tmp_path / "out", tag="t", use_hardlink=False
    )

    archived_image = archive_dir / "dataset" / "a.png"
    assert archived_image.read_bytes() == b"payload"
    assert archived_image.stat().st_ino != image.stat().st_ino


def test_archive_skips_missing_source(tmp_path: Path) -> None:
    archive_dir = archive(
        [str(tmp_path / "nope")], tmp_path / "out", tag="t"
    )

    assert archive_dir.is_dir()
    assert json.loads(
        (archive_dir / "metadata.json").read_text(encoding="utf-8")
    )["sources"]


# ---------------------------------------------------------------------------
# ir_enhance：worker 异常必须暴露，不能被 as_completed 吞掉
# ---------------------------------------------------------------------------


def test_augment_dataset_reports_missing_input_dir(tmp_path: Path) -> None:
    assert augment_dataset(tmp_path / "missing", tmp_path / "out") == 1


def test_augment_dataset_returns_failure_count_when_worker_raises(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_labelme(source / "a.json")

    def _boom(*args, **kwargs):
        raise RuntimeError("worker 失败")

    monkeypatch.setattr("tools.augment.ir_enhance.process_one", _boom)

    failed = augment_dataset(source, tmp_path / "out", repeat=1, max_workers=1)

    assert failed == 1


# ---------------------------------------------------------------------------
# replace：递归输入时按相对路径镜像，避免同名文件互相覆盖
# ---------------------------------------------------------------------------


def test_replace_mirrors_subdirectories_instead_of_flattening(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / "0001").mkdir(parents=True)
    (source / "0002").mkdir(parents=True)
    # 两个子目录下同名 JSON / 图片：扁平化会互相覆盖
    _write_labelme(source / "0001" / "same.json", "face")
    _write_labelme(source / "0002" / "same.json", "face")
    output = tmp_path / "out"

    for json_path in sorted(source.rglob("*.json")):
        process_labelme_file(
            json_path, output, {"face": "person"}, {}, source_root=source
        )

    assert (output / "0001" / "same.json").is_file()
    assert (output / "0002" / "same.json").is_file()
    assert (output / "0001" / "same.png").is_file()
    assert (output / "0002" / "same.png").is_file()
    data = json.loads((output / "0001" / "same.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["label"] == "person"


def test_replace_without_source_root_keeps_flat_output(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_labelme(source / "a.json", "face")
    output = tmp_path / "out"

    process_labelme_file(source / "a.json", output, {"face": "person"}, {})

    assert (output / "a.json").is_file()
