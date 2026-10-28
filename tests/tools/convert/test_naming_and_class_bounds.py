"""同名图片覆盖与类别 id 越界的回归测试。

覆盖：
- ``--format yolo`` 导出与 ``yolo_to_labelme`` 平铺输出时，不同子目录的同名图片
  会互相覆盖，静默丢标注且报告数量虚高；
- 类别 id 超出类别表时静默丢弃标注，无任何提示。
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.convert.yolo_to_labelme import convert_yolo_to_labelme
from tools.core.labelme import detections_to_labelme_dict


def _make_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color="white").save(path)
    return path


# ---------------------------------------------------------------------------
# 缺陷 14：同名图片互相覆盖
# ---------------------------------------------------------------------------


def test_yolo_to_labelme_does_not_overwrite_same_stem(tmp_path: Path) -> None:
    """不同子目录的同名图片必须都保留，不能互相覆盖。"""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    # a/shot.png 与 b/shot.png 同名但不同图
    _make_image(images / "a" / "shot.png")
    _make_image(images / "b" / "shot.png")
    labels.mkdir()
    (labels / "shot.txt").write_text("0 0.3 0.3 0.4 0.4\n", encoding="utf-8")
    output = tmp_path / "output"

    generated = convert_yolo_to_labelme(images, labels, ["phone"], output)

    jsons = list(output.glob("*.json"))
    assert generated == 2, f"应生成 2 个 JSON，实际 {generated}"
    assert len(jsons) == 2, f"同名覆盖，只剩 {len(jsons)} 个：{[p.name for p in jsons]}"
    # imagePath 必须各不相同，否则 LabelMe 里仍会指向同一张图
    paths = {json.loads(p.read_text(encoding="utf-8"))["imagePath"] for p in jsons}
    assert len(paths) == 2, f"imagePath 重复：{paths}"


# ---------------------------------------------------------------------------
# 缺陷 15：类别 id 越界静默丢弃
# ---------------------------------------------------------------------------


def test_out_of_range_class_id_raises(tmp_path: Path) -> None:
    """类别 id 超出类别表时必须报错，不能静默丢掉标注。"""
    import supervision as sv

    detections = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
        class_id=np.array([7]),
        confidence=np.array([0.9]),
    )

    with pytest.raises(ValueError, match="类别"):
        detections_to_labelme_dict(
            detections,
            class_names=["phone", "face"],
            image_path="a.png",
            image_width=100,
            image_height=80,
        )


def test_valid_class_id_still_converts() -> None:
    """正常类别 id 仍正常转换（守护）。"""
    import supervision as sv

    detections = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
        class_id=np.array([1]),
        confidence=np.array([0.9]),
    )
    data = detections_to_labelme_dict(
        detections,
        class_names=["phone", "face"],
        image_path="a.png",
        image_width=100,
        image_height=80,
    )
    assert len(data["shapes"]) == 1
    assert data["shapes"][0]["label"] == "face"


def test_negative_class_id_raises() -> None:
    """负的类别 id 同样要报错。"""
    import supervision as sv

    detections = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
        class_id=np.array([-1]),
        confidence=np.array([0.9]),
    )
    with pytest.raises(ValueError, match="类别"):
        detections_to_labelme_dict(
            detections,
            class_names=["phone"],
            image_path="a.png",
            image_width=100,
            image_height=80,
        )
