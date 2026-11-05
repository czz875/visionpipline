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

    assert _find_blurry_originals(missing, 0.18, {}) == set()


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
    images = {dataset / "a" / "same.png", dataset / "b" / "same.png"}
    moved = _move_bad_images(dataset, out, images, {p: {p.with_suffix(".json")} for p in images})

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
    moved = _move_bad_images(dataset, out, {dataset / "drop.png"}, {dataset / "drop.png": set()})

    assert moved == 1
    assert (dataset / "keep.png").is_file()
    assert not (dataset / "drop.png").exists()
    assert (out / "drop.png").is_file()


def _sample(path: Path, color: str = "red") -> Path:
    import json
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)
    path.with_suffix(".json").write_text(json.dumps({
        "imagePath": path.name,
        "shapes": [{"label": "face", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]}],
    }), encoding="utf-8")
    return path


def _fake_imagelab(monkeypatch):
    """只替代模糊打分：红色裁剪视为模糊，蓝色裁剪视为清晰。"""
    import pandas as pd
    from PIL import Image

    class FakeImagelab:
        def __init__(self, data_path):
            self.paths = sorted(Path(data_path).glob("*.png"))

        def find_issues(self, issue_types, n_jobs):
            flags = []
            for path in self.paths:
                with Image.open(path) as image:
                    red, _, blue = image.convert("RGB").getpixel((0, 0))
                    flags.append(red > blue)
            self.issues = pd.DataFrame({
                "is_blurry_issue": flags,
                "blurry_score": [0.1 if flag else 0.9 for flag in flags],
            }, index=[str(path) for path in self.paths])

    monkeypatch.setattr("tools.clean.blurry.Imagelab", FakeImagelab)


def test_recursive_crops_use_each_json_directory_and_do_not_collide(tmp_path: Path) -> None:
    from tools.clean.blurry import _crop_labelme

    dataset = tmp_path / "ds"
    first = _sample(dataset / "a" / "same.png", "red")
    second = _sample(dataset / "b" / "same.png", "blue")
    crops = tmp_path / "crops"

    _crop_labelme(dataset, crops, {"face"}, workers=2, recursive=True)

    from PIL import Image
    paths = list(crops.glob("*.png"))
    assert len(paths) == 2
    with Image.open(paths[0]) as a, Image.open(paths[1]) as b:
        assert {a.convert("RGB").getpixel((0, 0)), b.convert("RGB").getpixel((0, 0))} == {
            (255, 0, 0), (0, 0, 255),
        }
    assert first.exists() and second.exists()


def test_detection_moves_only_the_exact_flagged_source(tmp_path: Path, monkeypatch) -> None:
    from tools.clean.blurry import main

    dataset = tmp_path / "ds"
    bad = _sample(dataset / "a" / "same.png", "red")
    good = _sample(dataset / "b" / "same.png", "blue")
    _fake_imagelab(monkeypatch)
    monkeypatch.setattr("sys.argv", [
        "blurry", "--dataset-path", str(dataset), "--crop-path", str(tmp_path / "crops"),
        "--output-path", str(tmp_path / "out"), "--recursive", "--cleanup-crop",
    ])

    assert main() == 0
    assert not bad.exists() and not bad.with_suffix(".json").exists()
    assert (tmp_path / "out" / "a" / "same.png").exists()
    assert (tmp_path / "out" / "a" / "same.json").exists()
    assert good.exists() and good.with_suffix(".json").exists()


@pytest.mark.parametrize("layout", ["crop_same", "crop_parent", "crop_child", "out_same", "out_child", "out_crop"])
def test_main_rejects_overlapping_directories_before_writes(tmp_path: Path, monkeypatch, layout: str) -> None:
    from tools.clean import blurry

    dataset, crops, output = tmp_path / "ds", tmp_path / "crops", tmp_path / "out"
    image = _sample(dataset / "keep.png", "blue")
    before = image.read_bytes()
    if layout == "crop_same":
        crops = dataset
    elif layout == "crop_parent":
        crops = tmp_path
    elif layout == "crop_child":
        crops = dataset / "crops"
    elif layout == "out_same":
        output = dataset
    elif layout == "out_child":
        output = dataset / "out"
    else:
        output = crops
    cleanup_calls = []
    monkeypatch.setattr(blurry.shutil, "rmtree", lambda path, *a, **k: cleanup_calls.append(path))
    monkeypatch.setattr(blurry, "_find_blurry_originals", lambda *a, **k: set())
    monkeypatch.setattr("sys.argv", [
        "blurry", "--dataset-path", str(dataset), "--crop-path", str(crops),
        "--output-path", str(output), "--cleanup-crop",
    ])

    assert blurry.main() == 1
    assert image.read_bytes() == before
    assert set(dataset.iterdir()) == {image, image.with_suffix(".json")}
    assert cleanup_calls == []


def test_stale_crops_are_ignored_and_cleanup_keeps_user_files(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image
    from tools.clean.blurry import main

    dataset, crops = tmp_path / "ds", tmp_path / "crops"
    good = _sample(dataset / "same.png", "blue")
    crops.mkdir()
    stale = crops / "same_face_1.png"
    Image.new("RGB", (10, 10), color="red").save(stale)
    sentinel = crops / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    _fake_imagelab(monkeypatch)
    monkeypatch.setattr("sys.argv", [
        "blurry", "--dataset-path", str(dataset), "--crop-path", str(crops),
        "--output-path", str(tmp_path / "out"), "--cleanup-crop",
    ])

    assert main() == 0
    assert good.exists() and good.with_suffix(".json").exists()
    assert stale.exists() and sentinel.read_text() == "user data"
    assert set(crops.iterdir()) == {stale, sentinel}


def test_move_refuses_existing_destination_before_moving_any_pair(tmp_path: Path) -> None:
    dataset, output = tmp_path / "ds", tmp_path / "out"
    first = _sample(dataset / "a.png")
    second = _sample(dataset / "b.png")
    output.mkdir()
    (output / "b.json").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _move_bad_images(dataset, output, {first, second}, {p: {p.with_suffix(".json")} for p in (first, second)})

    assert first.exists() and second.exists()
    assert (output / "b.json").read_text() == "existing"


def test_real_scoring_does_not_spawn_a_process_pool(tmp_path: Path, monkeypatch) -> None:
    """真实 CleanVision 在本机应走单进程评分，避免重复加载 SciPy 耗尽提交内存。"""
    import multiprocessing
    import numpy as np
    from PIL import Image

    crops = tmp_path / "crops"
    crops.mkdir()
    sources = {}
    rng = np.random.default_rng(1234)
    window = np.sin(np.linspace(0, np.pi, 64))
    smooth = np.repeat((255 * np.outer(window, window)).astype(np.uint8)[..., None], 3, axis=2)
    for i in range(8):
        path = crops / f"{i:08d}_000000.png"
        pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) if i < 4 else smooth
        Image.fromarray(pixels).save(path)
        sources[path] = tmp_path / f"original_{i}.png"

    def forbidden_pool(*args, **kwargs):
        raise AssertionError("评分不应创建进程池")

    monkeypatch.setattr(multiprocessing, "Pool", forbidden_pool)
    selected = _find_blurry_originals(crops, 1.0, sources)
    assert selected
    assert selected <= set(sources.values())


def test_cross_directory_annotations_follow_the_actual_image(tmp_path: Path, monkeypatch) -> None:
    import json
    from PIL import Image
    from tools.clean.blurry import main

    dataset = tmp_path / "ds"
    (dataset / "images").mkdir(parents=True)
    (dataset / "annotations").mkdir()
    Image.new("RGB", (20, 20), color="red").save(dataset / "images" / "a.png")
    for name in ("first", "second"):
        (dataset / "annotations" / f"{name}.json").write_text(json.dumps({
            "imagePath": "../images/a.png",
            "shapes": [{"label": "face", "points": [[0, 0], [10, 10]]}],
        }), encoding="utf-8")
    Image.new("RGB", (20, 20), color="blue").save(dataset / "images" / "b.png")
    unrelated = dataset / "images" / "a.json"
    unrelated.write_text('{"imagePath":"b.png","shapes":[]}', encoding="utf-8")
    _fake_imagelab(monkeypatch)
    monkeypatch.setattr("sys.argv", [
        "blurry", "--dataset-path", str(dataset), "--crop-path", str(tmp_path / "crops"),
        "--output-path", str(tmp_path / "out"), "--recursive", "--cleanup-crop",
    ])

    assert main() == 0
    for annotation in ("annotations/first.json", "annotations/second.json"):
        assert not (dataset / annotation).exists()
        moved = tmp_path / "out" / annotation
        data = json.loads(moved.read_text(encoding="utf-8"))
        assert (moved.parent / data["imagePath"]).is_file()
    assert (tmp_path / "out" / "images" / "a.png").exists()
    assert unrelated.read_text() == '{"imagePath":"b.png","shapes":[]}'
    assert (dataset / "images" / "b.png").exists()


def test_blocked_parent_fails_before_any_source_is_moved(tmp_path: Path) -> None:
    dataset, output = tmp_path / "ds", tmp_path / "out"
    first = _sample(dataset / "a" / "first.png")
    second = _sample(dataset / "b" / "second.png")
    output.mkdir()
    (output / "b").write_bytes(b"existing file")

    with pytest.raises((OSError, ValueError)):
        _move_bad_images(dataset, output, {first, second}, {p: {p.with_suffix(".json")} for p in (first, second)})

    assert first.exists() and first.with_suffix(".json").exists()
    assert second.exists() and second.with_suffix(".json").exists()
    assert (output / "b").read_bytes() == b"existing file"


def test_failed_annotation_move_restores_previously_moved_files(tmp_path: Path, monkeypatch) -> None:
    from tools.clean import blurry

    dataset, output = tmp_path / "ds", tmp_path / "out"
    first = _sample(dataset / "a.png")
    second = _sample(dataset / "b.png")
    real_move = blurry._move_file_without_overwrite

    def fail_second_annotation(source, destination, *args, **kwargs):
        if Path(source) == second.with_suffix(".json"):
            raise OSError("模拟标注移动失败")
        return real_move(source, destination, *args, **kwargs)

    monkeypatch.setattr(blurry, "_move_file_without_overwrite", fail_second_annotation)
    with pytest.raises(OSError, match="模拟标注移动失败"):
        _move_bad_images(dataset, output, {first, second}, {p: {p.with_suffix(".json")} for p in (first, second)})
    for image in (first, second):
        assert image.exists() and image.with_suffix(".json").exists()
    assert not list(output.rglob("*.png"))
    assert not list(output.rglob("*.json"))


def test_cmyk_jpeg_can_be_cropped_as_png(tmp_path: Path) -> None:
    from PIL import Image
    from tools.clean.blurry import _crop_labelme

    dataset = tmp_path / "ds"
    dataset.mkdir()
    image = dataset / "color.jpg"
    Image.new("CMYK", (20, 20), color=(0, 255, 255, 0)).save(image)
    image.with_suffix(".json").write_text(
        '{"imagePath":"color.jpg","shapes":[{"label":"face","points":[[0,0],[10,10]]}]}', encoding="utf-8",
    )
    crops = tmp_path / "crops"
    _crop_labelme(dataset, crops, {"face"}, workers=1, recursive=True)
    assert len(list(crops.glob("*.png"))) == 1


def test_concurrent_runs_own_distinct_crop_directories(tmp_path: Path, monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    from threading import Barrier
    from tools.clean import blurry
    import tools.core.output_dir as output_dirs

    dataset, crops = tmp_path / "ds", tmp_path / "crops"
    _sample(dataset / "same.png", "blue")
    crops.mkdir()
    directories = []
    scoring_barrier, creation_barrier = Barrier(2), Barrier(2)
    real_mkdir = Path.mkdir

    class FixedDatetime:
        @staticmethod
        def now():
            return datetime(2026, 11, 5, 12, 0, 0)

    def synchronized_mkdir(path, *args, **kwargs):
        if path.name == "blurry_20261105_120000":
            creation_barrier.wait(timeout=5)
        return real_mkdir(path, *args, **kwargs)

    def score(path, *args):
        directories.append(path)
        scoring_barrier.wait(timeout=5)
        return set()

    monkeypatch.setattr(output_dirs, "datetime", FixedDatetime)
    monkeypatch.setattr(Path, "mkdir", synchronized_mkdir)
    monkeypatch.setattr(blurry, "_find_blurry_originals", score)
    monkeypatch.setattr("sys.argv", [
        "blurry", "--dataset-path", str(dataset), "--crop-path", str(crops),
        "--output-path", str(tmp_path / "out"), "--cleanup-crop",
    ])
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: blurry.main(), range(2)))
    assert results == [0, 0]
    assert len(set(directories)) == 2
    assert not list(crops.iterdir())


def test_destination_directory_link_cannot_escape_output_root(tmp_path: Path) -> None:
    import os

    dataset, output, outside = tmp_path / "ds", tmp_path / "out", tmp_path / "outside"
    image = _sample(dataset / "linked" / "same.png")
    output.mkdir()
    outside.mkdir()
    link = output / "linked"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"当前环境无法创建测试目录链接：{exc}")

    with pytest.raises(ValueError, match="目录链接"):
        _move_bad_images(dataset, output, {image}, {image: {image.with_suffix(".json")}})
    assert image.exists() and image.with_suffix(".json").exists()
    assert not list(outside.iterdir())


@pytest.mark.parametrize("reference", ["absolute", "root_relative", "drive_relative"])
def test_absolute_image_reference_is_rejected_before_moving(tmp_path: Path, monkeypatch, reference: str) -> None:
    import json
    from tools.clean.blurry import main

    dataset, output = tmp_path / "ds", tmp_path / "out"
    image = _sample(dataset / "same.png", "red")
    annotation = image.with_suffix(".json")
    data = json.loads(annotation.read_text(encoding="utf-8"))
    if reference == "root_relative":
        data["imagePath"] = image.as_posix()[len(image.drive):]
    elif reference == "drive_relative":
        data["imagePath"] = image.drive + image.as_posix()[len(image.drive):].lstrip("/")
    else:
        data["imagePath"] = str(image)
    annotation.write_text(json.dumps(data), encoding="utf-8")
    before = annotation.read_bytes()
    _fake_imagelab(monkeypatch)
    monkeypatch.setattr("sys.argv", [
        "blurry", "--dataset-path", str(dataset), "--crop-path", str(tmp_path / "crops"),
        "--output-path", str(output), "--cleanup-crop",
    ])

    assert main() == 1
    assert image.exists() and annotation.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("failure", ["copy", "unlink"])
def test_cross_volume_failure_cleans_only_its_partial_target(tmp_path: Path, monkeypatch, failure: str) -> None:
    import errno
    import os
    from tools.clean import blurry

    dataset, output = tmp_path / "ds", tmp_path / "out"
    image = _sample(dataset / "same.png")
    before = image.read_bytes()
    real_unlink = os.unlink

    def cross_volume(*args, **kwargs):
        raise OSError(errno.EXDEV, "模拟跨卷")

    def partial_copy(source, destination, *args, **kwargs):
        if hasattr(destination, "write"):
            destination.write(b"partial")
        else:
            Path(destination).write_bytes(b"partial")
        raise OSError("模拟复制中途失败")

    def locked_unlink(path, *args, **kwargs):
        if Path(path) == image:
            raise OSError("模拟源文件被占用")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "rename", cross_volume)
    monkeypatch.setattr(os, "link", cross_volume)
    if failure == "copy":
        monkeypatch.setattr(blurry.shutil, "copy2", partial_copy)
        monkeypatch.setattr(blurry.shutil, "copyfile", partial_copy)
        monkeypatch.setattr(blurry.shutil, "copyfileobj", partial_copy)
    else:
        monkeypatch.setattr(os, "unlink", locked_unlink)

    with pytest.raises(OSError, match="模拟"):
        _move_bad_images(dataset, output, {image}, {image: {image.with_suffix(".json")}})
    assert image.read_bytes() == before
    assert image.with_suffix(".json").exists()
    assert not list(output.rglob("*.png"))
    assert not list(output.rglob("*.json"))
