"""配置中的 YOLO 类别表与实际数据集保持一致的回归测试。

``--classes`` 本身可指定，且 ``parameters.yolo_classes`` 能在项目配置里覆盖。
但默认值写死在 ``default.yaml``，与磁盘上的 ``datasets/yolo/*/data.yaml``
可能悄悄偏离：一旦不一致，转换会丢弃该批次的类别（如 helmet），
而 ``labelme_to_yolo`` 又会把 data.yaml 改写成新的类别表。

这里用守护测试锁住「默认配置的类别表 == 磁盘实际类别表」，
并确保重复出现的类别表只保留一处权威定义。
"""

from pathlib import Path

import pytest
import yaml

from tools.core.constants import YOLO_CLASS_NAMES
from tools.convert.labelme_to_yolo import DEFAULT_CLASS_NAMES
from tools.merge.inherit_dataset import CATEGORY_LABELS

REPO_ROOT = Path(__file__).resolve().parents[3]
CFG_DIR = REPO_ROOT / "tools" / "cfg"
YOLO_ROOT = REPO_ROOT / "datasets" / "yolo"


def _default_yolo_classes() -> list[str]:
    data = yaml.safe_load((CFG_DIR / "default.yaml").read_text(encoding="utf-8"))
    raw = data["parameters"]["yolo_classes"]
    return [c.strip() for c in raw.split(",") if c.strip()]


def test_code_constant_is_the_single_source_of_truth() -> None:
    """代码里的类别表必须直接引用 tools.core.constants.YOLO_CLASS_NAMES。

    历史上 ``labelme_to_yolo.py`` 写死了 4 类（缺 helmet）、``default.yaml``
    是 5 类，不传 ``--classes`` 直接跑就会把 helmet 标注全部丢弃。
    这里锁住「代码常量 == core 常量」，并要求继承映射与之对齐。
    """
    assert DEFAULT_CLASS_NAMES == YOLO_CLASS_NAMES, (
        "labelme_to_yolo 的 DEFAULT_CLASS_NAMES 必须直接引用 "
        "tools.core.constants.YOLO_CLASS_NAMES"
    )
    assert list(CATEGORY_LABELS.values()) == list(YOLO_CLASS_NAMES), (
        "inherit_dataset 的 CATEGORY_LABELS 顺序必须与 YOLO_CLASS_NAMES 一致"
    )


def test_default_yaml_matches_code_constant() -> None:
    """default.yaml 的 yolo_classes 必须与代码常量一致。"""
    assert _default_yolo_classes() == list(YOLO_CLASS_NAMES), (
        "default.yaml 的 yolo_classes 与 tools.core.constants.YOLO_CLASS_NAMES 不一致；"
        "不传 --classes 直接跑转换会丢弃缺失类别的标注"
    )


def _disk_yolo_classes() -> list[str]:
    """取磁盘上任一批次的 names 作为权威类别表。"""
    batches = sorted(p for p in YOLO_ROOT.iterdir() if p.is_dir())
    assert batches, "datasets/yolo 下没有批次"
    data = yaml.safe_load((batches[0] / "data.yaml").read_text(encoding="utf-8"))
    names = data.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


@pytest.mark.skipif(
    not YOLO_ROOT.is_dir(), reason="需要真实 datasets/yolo 才能比对"
)
def test_default_classes_match_dataset() -> None:
    """默认配置的类别表必须与磁盘数据集一致。"""
    assert _default_yolo_classes() == _disk_yolo_classes(), (
        "default.yaml 的 yolo_classes 与 datasets/yolo 的 names 不一致；"
        "跑 convert_to_yolo 会丢弃缺失类别的标注"
    )


@pytest.mark.skipif(
    not YOLO_ROOT.is_dir(), reason="需要真实 datasets/yolo 才能比对"
)
def test_all_batches_share_same_classes() -> None:
    """所有批次的类别表必须一致，否则跨批训练会错位。"""
    seen = {}
    for batch in sorted(p for p in YOLO_ROOT.iterdir() if p.is_dir()):
        data = yaml.safe_load((batch / "data.yaml").read_text(encoding="utf-8"))
        names = data.get("names") or []
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=lambda x: int(x))]
        seen[batch.name] = tuple(str(n) for n in names)
    distinct = set(seen.values())
    assert len(distinct) == 1, f"批次间类别表不一致：{distinct}"


def test_labelme_to_yolo_yaml_has_no_duplicate_class_table() -> None:
    """专项配置不应重复定义类别表，避免与 default.yaml 各写一份而偏离。

    ``labelme_to_yolo.yaml`` 的 ``--classes`` 应引用 ``${parameters.yolo_classes}``
    这一唯一来源，而不是在自身 ``parameters`` 里再写一份字面量。
    """
    data = yaml.safe_load((CFG_DIR / "labelme_to_yolo.yaml").read_text(encoding="utf-8"))
    own_params = data.get("parameters") or {}
    assert "yolo_classes" not in own_params, (
        "labelme_to_yolo.yaml 重复定义了 yolo_classes；"
        "应只引用 default.yaml 的 ${parameters.yolo_classes}"
    )
    command = data["stages"][0]["command"]
    assert "${parameters.yolo_classes}" in command, (
        "应引用 ${parameters.yolo_classes} 这一唯一来源"
    )
