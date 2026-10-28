"""tools.core 公共能力的回归测试。"""

from pathlib import Path

import pytest
from PIL import Image

from tools.core import (
    IMAGE_EXTENSIONS,
    detections_to_labelme_dict,
    find_image_for_json,
    find_json_for_image,
    labelme_dict_to_detections,
    list_images,
    list_labelme_files,
    load_labelme,
    rect_to_xyxy,
    save_labelme,
    unregistered_labels,
)


def _write_json(path: Path, shapes: list[dict], **extra) -> None:
    import json

    data = {"shapes": shapes}
    data.update(extra)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_labelme_dict_to_detections_clips_to_image_size() -> None:
    data = {
        "shapes": [
            {"label": "a", "shape_type": "rectangle", "points": [[-10, -10], [200, 200]]},
            {"label": "a", "shape_type": "rectangle", "points": [[50, 50], [150, 150]]},
        ]
    }

    detections = labelme_dict_to_detections(
        data, {"a": 0}, image_size=(100, 100)
    )

    assert len(detections) == 2
    assert detections.xyxy[0].tolist() == [0.0, 0.0, 100.0, 100.0]
    assert detections.xyxy[1].tolist() == [50.0, 50.0, 100.0, 100.0]


def test_labelme_dict_to_detections_drops_empty_boxes_after_clip() -> None:
    data = {
        "shapes": [
            {"label": "a", "shape_type": "rectangle", "points": [[200, 200], [300, 300]]},
        ]
    }

    assert len(labelme_dict_to_detections(data, {"a": 0}, image_size=(100, 100))) == 0


def test_labelme_dict_to_detections_skips_unregistered_and_non_rect() -> None:
    data = {
        "shapes": [
            {"label": "unknown", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]},
            {"label": "a", "shape_type": "polygon", "points": [[0, 0], [10, 10], [5, 5]]},
            {"label": "a", "shape_type": "rectangle", "points": [[0, 0]]},
            {"label": "a", "shape_type": "rectangle", "points": [[1, 1], [9, 9]]},
        ]
    }

    detections = labelme_dict_to_detections(data, {"a": 0})

    assert len(detections) == 1


def test_unregistered_labels_reports_only_missing_rect_labels() -> None:
    data = {
        "shapes": [
            {"label": "helmet", "shape_type": "rectangle", "points": [[0, 0], [1, 1]]},
            {"label": "helmet", "shape_type": "rectangle", "points": [[0, 0], [1, 1]]},
            {"label": "phone", "shape_type": "rectangle", "points": [[0, 0], [1, 1]]},
            {"label": "helmet", "shape_type": "polygon", "points": [[0, 0], [1, 1], [2, 2]]},
        ]
    }

    assert unregistered_labels(data, {"phone": 0}) == ["helmet"]


def test_rect_to_xyxy_normalizes_reversed_points() -> None:
    assert rect_to_xyxy([[50, 60], [10, 20]]) == (10, 20, 50, 60)


def test_list_images_honours_extension_set(tmp_path: Path) -> None:
    for name in ("a.png", "b.webp", "c.txt", "d.JPG"):
        (tmp_path / name).write_bytes(b"x")

    found = {p.name for p in list_images(tmp_path)}

    assert found == {"a.png", "b.webp", "d.JPG"}
    assert ".webp" in IMAGE_EXTENSIONS


def test_find_image_and_json_are_inverse(tmp_path: Path) -> None:
    image = tmp_path / "x.png"
    image.write_bytes(b"x")
    json_path = find_json_for_image(image)

    assert json_path.name == "x.json"
    assert find_image_for_json(json_path) == image


def test_labelme_roundtrip(tmp_path: Path) -> None:
    data = {"shapes": [{"label": "a"}], "imagePath": "x.png"}
    target = tmp_path / "x.json"

    save_labelme(data, target)

    assert load_labelme(target) == data
    assert list_labelme_files(tmp_path) == [target]


def test_detections_to_labelme_dict_rejects_out_of_range_class_ids() -> None:
    """类别 id 超出类别表时必须报错。

    原先静默 continue，标注无声消失（例如 --classes 少于磁盘 data.yaml 的
    类别数时），用户只看到框变少却没有任何提示。
    """
    import numpy as np
    import pytest
    import supervision as sv

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [1, 1, 5, 5]], dtype=np.float32),
        class_id=np.array([0, 5], dtype=int),
    )

    with pytest.raises(ValueError, match="类别"):
        detections_to_labelme_dict(
            detections,
            class_names=["a"],
            image_path="x.png",
            image_width=100,
            image_height=100,
        )


def test_detections_to_labelme_dict_accepts_in_range_class_ids() -> None:
    """类别 id 都在范围内时正常转换（守护）。"""
    import numpy as np
    import supervision as sv

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [1, 1, 5, 5]], dtype=np.float32),
        class_id=np.array([0, 1], dtype=int),
    )

    data = detections_to_labelme_dict(
        detections,
        class_names=["a", "b"],
        image_path="x.png",
        image_width=100,
        image_height=100,
    )

    assert len(data["shapes"]) == 2
    assert data["shapes"][0]["label"] == "a"
    assert data["shapes"][1]["label"] == "b"
    assert data["imageWidth"] == 100
