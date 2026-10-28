"""转换与核心层的三条回归测试。

覆盖：
- 未注册标签导致「全部标注被丢弃」时，不能把该图写成空标签当负样本；
- ``copy_dataset`` 不能把硬链接展开成实拷（178 GB 数据集留底会撑爆磁盘）；
- ``find_image_for_json`` 的扩展名优先级必须稳定，不能随进程变化。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from tools.convert.labelme_to_yolo import process_batch
from tools.core.copy_dataset import copy_dataset
from tools.core.labelme import find_image_for_json


def _write_annotation(json_path: Path, image_path: Path, labels: list[str]) -> None:
    """写一张白图 + 含指定标签的 LabelMe JSON。"""
    Image.new("RGB", (100, 80), color="white").save(image_path)
    data = {
        "shapes": [
            {
                "label": label,
                "shape_type": "rectangle",
                "points": [[10, 10], [50, 50]],
            }
            for label in labels
        ],
        "imageWidth": 100,
        "imageHeight": 80,
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# 缺陷 4：全部标注被丢弃的图不能变成负样本
# ---------------------------------------------------------------------------


def test_image_with_all_labels_dropped_is_not_written_as_negative(
    tmp_path: Path,
) -> None:
    """所有标注都未注册时，该图不该进入训练集当负样本。

    写空 txt 会让 ultralytics 判 ``ne=1``，图仍进训练集并被当成「这里没有目标」，
    等于主动教模型错误的负样本。
    """
    source = tmp_path / "source"
    source.mkdir()
    json_path = source / "a.json"
    _write_annotation(json_path, source / "a.png", ["helmet"])
    output = tmp_path / "output"

    dropped = process_batch(
        "0001", [json_path], output, seed=3407,
        class_names=["phone"], force=True,
    )

    assert "helmet" in dropped, "未注册标签必须上报"
    all_labels = list((output / "labels").rglob("*.txt"))
    all_images = list((output / "images").rglob("*.png"))
    assert not all_labels, f"不该写出标签文件，实际 {[p.name for p in all_labels]}"
    assert not all_images, f"不该复制图片，实际 {[p.name for p in all_images]}"


def test_image_with_some_labels_kept_still_written(tmp_path: Path) -> None:
    """部分标签保留时，图与标签都要正常写出（守护：不能矫枉过正）。"""
    source = tmp_path / "source"
    source.mkdir()
    json_path = source / "a.json"
    _write_annotation(json_path, source / "a.png", ["phone", "helmet"])
    output = tmp_path / "output"

    process_batch(
        "0001", [json_path], output, seed=3407,
        class_names=["phone"], force=True,
    )

    label_files = list((output / "labels").rglob("*.txt"))
    assert len(label_files) == 1
    assert label_files[0].read_text(encoding="utf-8").strip() != ""
    assert len(list((output / "images").rglob("*.png"))) == 1


def test_image_with_no_shapes_still_written_as_negative(tmp_path: Path) -> None:
    """本来就没有标注的图是真实负样本，仍应写出（守护）。"""
    source = tmp_path / "source"
    source.mkdir()
    json_path = source / "a.json"
    _write_annotation(json_path, source / "a.png", [])
    output = tmp_path / "output"

    process_batch("0001", [json_path], output, seed=3407, class_names=["phone"], force=True)

    assert len(list((output / "labels").rglob("*.txt"))) == 1
    assert len(list((output / "images").rglob("*.png"))) == 1


# ---------------------------------------------------------------------------
# 缺陷 10：copy_dataset 不能展开硬链接
# ---------------------------------------------------------------------------


def test_copy_dataset_preserves_hardlinks(tmp_path: Path) -> None:
    """硬链接数据集留底时不能展开成实拷，否则物理占用翻倍。"""
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (10, 10), color="white").save(src / "a.png")
    os.link(src / "a.png", src / "b.png")
    assert (src / "a.png").stat().st_nlink == 2
    dst = tmp_path / "dst"

    copy_dataset(src, dst)

    assert (dst / "a.png").stat().st_nlink >= 2, "硬链接被展开成实拷"
    assert (dst / "a.png").stat().st_ino == (dst / "b.png").stat().st_ino, (
        "复制后两个文件不再是同一 inode"
    )


# ---------------------------------------------------------------------------
# 缺陷 11：扩展名优先级必须确定
# ---------------------------------------------------------------------------


def test_find_image_for_json_is_stable_across_processes(tmp_path: Path) -> None:
    """同名多格式并存时，不同进程必须选中同一个文件。

    ``IMAGE_EXTENSIONS`` 若是 ``frozenset``，字符串哈希每进程随机化，迭代顺序
    随之变化 —— 同一份数据不同进程会选到不同图片，孤儿判定时对时错。
    """
    Image.new("RGB", (10, 10), color="white").save(tmp_path / "shot.jpg")
    Image.new("RGB", (10, 10), color="white").save(tmp_path / "shot.png")
    json_path = tmp_path / "shot.json"
    json_path.write_text(json.dumps({"shapes": []}), encoding="utf-8")

    script = (
        "import sys;"
        "from pathlib import Path;"
        "from tools.core.labelme import find_image_for_json;"
        f"print(find_image_for_json(Path(r'{json_path}')).suffix)"
    )
    repo_root = Path(__file__).resolve().parents[3]
    picks = set()
    for _ in range(8):
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
        assert out.returncode == 0, out.stderr
        picks.add(out.stdout.strip())

    assert len(picks) == 1, f"不同进程选中了不同图片：{sorted(picks)}"
