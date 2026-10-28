"""中文路径与增量划分的回归测试。

覆盖：
- ``jpg_to_png`` 直用 ``cv2.imread/imwrite``，中文路径下静默丢图（图片与标注
  一起丢），退出码仍为 0；
- ``labelme_to_yolo`` 增量重跑时全批次重排 train/val，旧划分的图片副本残留在
  另一个 split 目录里，造成 train/val 泄漏。
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.convert.jpg_to_png import process_single_file
from tools.convert.labelme_to_yolo import process_batch


def _counters() -> dict:
    return {"image": 0, "json": 0}


def _locks():
    import threading

    return threading.Lock(), threading.Lock()


# ---------------------------------------------------------------------------
# 缺陷 9：中文路径丢图
# ---------------------------------------------------------------------------


def test_jpg_to_png_handles_chinese_input_path(tmp_path: Path) -> None:
    """输入目录/文件名含中文时仍必须转换成功。"""
    in_dir = tmp_path / "输入目录"
    in_dir.mkdir()
    src = in_dir / "测试图.jpg"
    Image.new("RGB", (40, 30), color="red").save(src)
    out_dir = tmp_path / "out"
    dst = out_dir / "测试图.png"

    print_lock, counter_lock = _locks()
    counters = _counters()
    ok = process_single_file(
        src, dst,
        png_compression=9,
        print_lock=print_lock,
        counter_lock=counter_lock,
        counters=counters,
    )

    assert ok is True
    assert dst.exists(), "中文路径写盘失败，图片丢失"
    assert counters["image"] == 1


def test_jpg_to_png_handles_chinese_output_dir(tmp_path: Path) -> None:
    """输出目录含中文时也必须写盘成功。"""
    src = tmp_path / "plain.jpg"
    Image.new("RGB", (40, 30), color="red").save(src)
    out_dir = tmp_path / "输出目录"
    dst = out_dir / "plain.png"

    print_lock, counter_lock = _locks()
    counters = _counters()
    ok = process_single_file(
        src, dst,
        png_compression=9,
        print_lock=print_lock,
        counter_lock=counter_lock,
        counters=counters,
    )

    assert ok is True
    assert dst.exists(), "中文输出目录写盘失败"


def test_jpg_to_png_copies_json_annotation(tmp_path: Path) -> None:
    """中文路径下同名 JSON 也要一起迁移（守护）。"""
    in_dir = tmp_path / "输入目录"
    in_dir.mkdir()
    src = in_dir / "图.jpg"
    Image.new("RGB", (40, 30), color="red").save(src)
    (in_dir / "图.json").write_text(
        json.dumps({"shapes": [], "imagePath": "图.jpg", "imageWidth": 40, "imageHeight": 30}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    dst = out_dir / "图.png"

    print_lock, counter_lock = _locks()
    counters = _counters()
    process_single_file(
        src, dst,
        png_compression=9,
        print_lock=print_lock,
        counter_lock=counter_lock,
        counters=counters,
    )

    assert (out_dir / "图.json").exists(), "中文路径下标注没跟着迁移"
    assert counters["json"] == 1


# ---------------------------------------------------------------------------
# 缺陷 12：增量重跑导致 train/val 泄漏
# ---------------------------------------------------------------------------


def _write_one(dir_path: Path, stem: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color="white").save(dir_path / f"{stem}.png")
    data = {
        "shapes": [
            {"label": "phone", "shape_type": "rectangle", "points": [[10, 10], [50, 50]]}
        ],
        "imageWidth": 100,
        "imageHeight": 80,
    }
    json_path = dir_path / f"{stem}.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return json_path


def test_incremental_rerun_does_not_leak_between_splits(tmp_path: Path) -> None:
    """增量重跑后，同一张图不能同时存在于 images/train 和 images/val。"""
    source = tmp_path / "source"
    json_files = [_write_one(source, f"img{i:02d}") for i in range(20)]
    output = tmp_path / "output"

    process_batch("0001", json_files, output, seed=3407,
                  class_names=["phone"], force=True)
    # 增加一张后增量重跑
    json_files.append(_write_one(source, "img20"))
    process_batch("0001", json_files, output, seed=3407,
                  class_names=["phone"], force=False)

    train_names = {p.name for p in (output / "images" / "train").glob("*.png")}
    val_names = {p.name for p in (output / "images" / "val").glob("*.png")}
    overlap = train_names & val_names
    assert not overlap, f"train/val 泄漏：{sorted(overlap)}"


def test_split_lists_match_directories_after_rerun(tmp_path: Path) -> None:
    """列表文件必须与实际目录内容一致，不能多出残留文件。"""
    source = tmp_path / "source"
    json_files = [_write_one(source, f"img{i:02d}") for i in range(20)]
    output = tmp_path / "output"

    process_batch("0001", json_files, output, seed=3407,
                  class_names=["phone"], force=True)
    json_files.append(_write_one(source, "img20"))
    process_batch("0001", json_files, output, seed=3407,
                  class_names=["phone"], force=False)

    for split in ("train", "val"):
        listed = {
            Path(line.strip()).name
            for line in (output / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        on_disk = {p.name for p in (output / "images" / split).glob("*.png")}
        assert listed == on_disk, (
            f"{split}: 列表 {len(listed)} 个、目录 {len(on_disk)} 个，"
            f"多出 {sorted(on_disk - listed)}"
        )
