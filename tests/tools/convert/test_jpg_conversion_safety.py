"""JPG 转换不能覆盖冲突目标，也不能将部分失败报告为成功。"""

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.convert.jpg_to_png import convert_jpg_to_png, main


def _jpg(path: Path, annotation: str | None = '{"imagePath":"old.jpg","shapes":[]}') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 12), color="red").save(path)
    if annotation is not None:
        path.with_suffix(".json").write_text(annotation, encoding="utf-8")
    return path


def test_same_stem_extensions_are_rejected_before_output_creation(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    _jpg(source / "same.jpg")
    _jpg(source / "same.jpeg")
    with pytest.raises(ValueError, match="冲突"):
        convert_jpg_to_png(source, output)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["bad_image", "bad_json", "json_array"])
def test_cli_failure_returns_nonzero_and_does_not_publish_unpaired_files(tmp_path: Path, monkeypatch, failure: str) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    image = _jpg(source / "sample.jpg")
    if failure == "bad_image":
        image.write_bytes(b"not an image")
    elif failure == "bad_json":
        image.with_suffix(".json").write_text("{ broken", encoding="utf-8")
    else:
        image.with_suffix(".json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["jpg_to_png", "--input", str(source), "--output", str(output)])

    assert main() == 1
    assert not list(output.rglob("*.png"))
    assert not list(output.rglob("*.json"))


@pytest.mark.parametrize("existing", ["sample.png", "sample.json"])
def test_existing_destination_is_preserved(tmp_path: Path, existing: str) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    _jpg(source / "sample.jpg")
    output.mkdir()
    (output / existing).write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        convert_jpg_to_png(source, output)
    assert set(output.iterdir()) == {output / existing}
    assert (output / existing).read_bytes() == b"keep"


def test_successful_conversion_keeps_nested_pairs_and_real_dimensions(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    _jpg(source / "a" / "same.jpg", '{"imagePath":"old.jpg","imageWidth":999,"imageHeight":888,"shapes":[]}')
    _jpg(source / "b" / "same.jpg")

    assert convert_jpg_to_png(source, output, num_threads=2) == (2, 2)
    for folder in ("a", "b"):
        target = output / folder / "same.png"
        data = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        assert target.is_file()
        assert data["imagePath"] == "same.png"
        assert (data["imageWidth"], data["imageHeight"]) == (20, 12)


@pytest.mark.parametrize("source_exists", [False, True])
def test_missing_or_empty_source_returns_failure(tmp_path: Path, monkeypatch, source_exists: bool) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    if source_exists:
        source.mkdir()
    monkeypatch.setattr("sys.argv", ["jpg_to_png", "--input", str(source), "--output", str(output)])
    assert main() == 1
    assert not output.exists()


def test_json_publish_failure_rolls_back_only_this_pair(tmp_path: Path, monkeypatch) -> None:
    import tools.convert.jpg_to_png as converter

    source, output = tmp_path / "source", tmp_path / "out"
    image = _jpg(source / "sample.jpg")
    original = image.read_bytes()
    real_publish = converter.move_file_without_overwrite

    def fail_json(staged, target):
        if target.suffix == ".json":
            raise OSError("模拟 JSON 发布失败")
        return real_publish(staged, target)

    monkeypatch.setattr(converter, "move_file_without_overwrite", fail_json)
    with pytest.raises(RuntimeError, match="1 个样本转换失败"):
        convert_jpg_to_png(source, output)
    assert image.read_bytes() == original
    assert image.with_suffix(".json").exists()
    assert not list(output.iterdir())


def test_partial_batch_retains_only_complete_successful_pairs(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    _jpg(source / "good.jpg")
    _jpg(source / "bad.jpg", "{ invalid")
    with pytest.raises(RuntimeError, match="完整成功 1 张图片、1 个 JSON"):
        convert_jpg_to_png(source, output, num_threads=2)
    assert {p.name for p in output.iterdir()} == {"good.png", "good.json"}


@pytest.mark.parametrize("parameters", [{"num_threads": 0}, {"png_compression": -1}, {"png_compression": 10}])
def test_invalid_parameters_do_not_create_output(tmp_path: Path, parameters: dict) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    _jpg(source / "sample.jpg")
    with pytest.raises(ValueError):
        convert_jpg_to_png(source, output, **parameters)
    assert not output.exists()


def test_planned_file_cannot_also_be_a_planned_directory(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    _jpg(source / "a.jpg")
    _jpg(source / "a.png" / "b.jpg")
    with pytest.raises(ValueError, match="冲突"):
        convert_jpg_to_png(source, output)
    assert not output.exists()


def test_output_root_link_is_rejected_before_writing(tmp_path: Path) -> None:
    import os

    source, output, external = tmp_path / "source", tmp_path / "alias", tmp_path / "external"
    _jpg(source / "a.jpg")
    external.mkdir()
    os.symlink(external, output, target_is_directory=True)
    with pytest.raises(ValueError, match="目录链接"):
        convert_jpg_to_png(source, output / "new")
    assert not list(external.iterdir())


def test_dangling_annotation_is_a_failed_sample(tmp_path: Path, monkeypatch) -> None:
    import os

    source, output = tmp_path / "source", tmp_path / "out"
    image = _jpg(source / "a.jpg", None)
    os.symlink(source / "missing.json", image.with_suffix(".json"))
    monkeypatch.setattr("sys.argv", ["jpg_to_png", "--input", str(source), "--output", str(output)])
    assert main() == 1
    assert not list(output.rglob("*.png"))


def test_pair_rollback_does_not_delete_another_writers_replacement(tmp_path: Path, monkeypatch) -> None:
    import os
    import tools.convert.jpg_to_png as converter

    source, output = tmp_path / "source", tmp_path / "out"
    _jpg(source / "sample.jpg")
    real_publish = converter.move_file_without_overwrite

    def replaced_before_failure(staged, target):
        if target.suffix == ".json":
            replacement = output / ".replacement.png"
            replacement.write_bytes(b"another writer")
            os.replace(replacement, target.with_suffix(".png"))
            raise OSError("模拟 JSON 发布失败")
        return real_publish(staged, target)

    monkeypatch.setattr(converter, "move_file_without_overwrite", replaced_before_failure)
    with pytest.raises(RuntimeError, match="1 个样本转换失败"):
        convert_jpg_to_png(source, output)
    assert (output / "sample.png").read_bytes() == b"another writer"
