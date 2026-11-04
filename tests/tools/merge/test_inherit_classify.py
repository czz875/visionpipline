"""数据接续的**类别归类**路径测试。

``inherit_dataset.py`` 的 ``classify`` 决定一张图进哪个数字子目录
（0=phone / 1=cigarette / 2=face / 3=hand / 4=helmet / other）。这条路径此前
**零覆盖**——现有测试全部用 ``no_classify=True`` 绕开它。

归类错了的后果是静默的：图片进了错的子目录、YOLO 类别 id 整体错位，
ultralytics 会把越界 id 的图判为 corrupt 整张丢弃，且不报错。
"""

from __future__ import annotations

from pathlib import Path

from tools.core.constants import CLASS_PRIORITY, YOLO_CLASS_NAMES
from tools.merge.inherit_dataset import (
    CATEGORY_LABELS,
    CATEGORY_DIRS,
    OTHER_CATEGORY,
    classify,
    collect_existing_stems,
    read_labels,
)


def test_single_label_maps_to_its_own_category() -> None:
    """单个已知标签应进对应编号目录。"""
    for class_id, label in enumerate(YOLO_CLASS_NAMES):
        assert classify([label]) == str(class_id), f"{label} 应归入目录 {class_id}"


def test_unlabelled_image_goes_to_other() -> None:
    """没有任何标签的图进 other。"""
    assert classify([]) == OTHER_CATEGORY


def test_unknown_label_goes_to_other() -> None:
    """标签不在类别表里（如人、车）应进 other，而不是被硬塞进某个类别。"""
    assert classify(["person", "car"]) == OTHER_CATEGORY


def test_multiple_labels_follow_priority_order() -> None:
    """多标签时按 CLASSIFICATION_PRIORITY 取第一个命中的类别（返回目录号）。"""
    # CATEGORY_LABELS 是「目录号 -> 标签」映射，classify 返回的是目录号
    helmet_dir = next(d for d, lbl in CATEGORY_LABELS.items() if lbl == "helmet")
    cigarette_dir = next(d for d, lbl in CATEGORY_LABELS.items() if lbl == "cigarette")
    hand_dir = next(d for d, lbl in CATEGORY_LABELS.items() if lbl == "hand")

    assert classify(["face", "helmet"]) == helmet_dir
    assert classify(["phone", "cigarette"]) == cigarette_dir
    assert classify(["face", "hand"]) == hand_dir


def test_priority_order_is_respected_globally() -> None:
    """无论标签以什么顺序传入，结果都应相同（classify 内部按优先级遍历）。"""
    labels = list(YOLO_CLASS_NAMES)
    expected = classify(labels)

    for perm in ([labels[2], labels[0], labels[4]], [labels[4], labels[1]], list(reversed(labels))):
        assert classify(perm) == expected, f"标签顺序 {perm} 不应影响归类结果"


def test_classification_priority_covers_all_known_classes() -> None:
    """优先级表必须覆盖全部已知类别，不能有类别永远归不到自己。"""
    assert set(CLASS_PRIORITY) == set(YOLO_CLASS_NAMES)
    assert len(CLASS_PRIORITY) == len(set(CLASS_PRIORITY)), "优先级表里有重复项"


def test_category_dirs_include_other() -> None:
    """分类目录集合 = 各类别目录 + other。"""
    assert set(CATEGORY_DIRS) == {str(i) for i in range(len(YOLO_CLASS_NAMES))} | {OTHER_CATEGORY}


def test_read_labels_extracts_labels_from_shapes(tmp_path: Path) -> None:
    """从 JSON 的 shapes 里取出所有标签。"""
    import json as _json

    json_path = tmp_path / "a.json"
    json_path.write_text(
        _json.dumps(
            {
                "shapes": [
                    {"label": "face", "shape_type": "rectangle"},
                    {"label": "hand", "shape_type": "rectangle"},
                    {"label": "face", "shape_type": "rectangle"},
                    {"shape_type": "rectangle"},  # 缺 label 的坏 shape
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert read_labels(json_path) == ["face", "hand"]


def test_read_labels_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """JSON 不存在时返回空列表，不抛异常。"""
    assert read_labels(tmp_path / "nope.json") == []


def test_read_labels_tolerates_broken_json(tmp_path: Path) -> None:
    """坏 JSON 按无标签处理（会进 other），不能中断整批。"""
    json_path = tmp_path / "broken.json"
    json_path.write_text("{ 不是合法 JSON", encoding="utf-8")

    assert read_labels(json_path) == []


def test_collect_existing_stems_finds_images(tmp_path: Path) -> None:
    """去重扫描应收集到目标目录下的图片 stem。"""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "sub" / "c.png").write_bytes(b"x")
    (tmp_path / "d.txt").write_text("x", encoding="utf-8")

    stems = collect_existing_stems(tmp_path)

    assert {"a", "b", "c"} <= stems
    assert "d" not in stems, "非图片文件不应计入去重集合"


def test_collect_existing_stems_on_missing_dir_is_empty(tmp_path: Path) -> None:
    """目标目录不存在时返回空集合，不抛异常。"""
    assert collect_existing_stems(tmp_path / "nope") == set()
