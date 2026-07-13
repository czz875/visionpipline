from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tools.annotate.reannotate_face_hand_onnx import (
    DEFAULT_END_BATCH,
    DEFAULT_MIN_FACE_RATIO,
    DEFAULT_START_BATCH,
    build_labelme_shapes,
    classify_face_box,
    mosaic_region,
    resolve_execution_providers,
    rewrite_labelme_dict,
)


def test_classify_face_box_marks_small_face() -> None:
    image_shape = (1000, 1000, 3)
    small_face = np.array([0, 0, 100, 100], dtype=np.float32)

    assert classify_face_box(small_face, image_shape, min_ratio=0.05) is False


def test_build_labelme_shapes_only_keeps_face_and_hand() -> None:
    boxes = np.array([[10, 10, 50, 50], [60, 60, 120, 120]], dtype=np.float32)
    labels = ["face", "hand"]

    shapes = build_labelme_shapes(boxes, labels)

    assert [shape["label"] for shape in shapes] == ["face", "hand"]


def test_mosaic_region_changes_pixels_inside_box() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for row in range(8, 24):
        for col in range(8, 24):
            image[row, col] = [(row - 8) * 8, (col - 8) * 8, (row + col) % 256]

    box = np.array([8, 8, 24, 24], dtype=np.float32)
    result = mosaic_region(image.copy(), box)

    assert np.any(result[8:24, 8:24] != image[8:24, 8:24])


def test_rewrite_labelme_dict_discards_old_phone_and_cigarette(tmp_path: Path) -> None:
    json_path = tmp_path / "sample.json"
    json_path.write_text(
        (
            '{"shapes":['
            '{"label":"phone","points":[[0,0],[1,1]],"shape_type":"rectangle"},'
            '{"label":"cigarette","points":[[1,1],[2,2]],"shape_type":"rectangle"}'
            '],"imagePath":"sample.png","imageHeight":100,"imageWidth":100}'
        ),
        encoding="utf-8",
    )

    updated = rewrite_labelme_dict(
        json_path=json_path,
        boxes=np.array([[2, 2, 10, 10]], dtype=np.float32),
        labels=["face"],
        image_shape=(100, 100, 3),
        image_name="sample.png",
    )

    assert [shape["label"] for shape in updated["shapes"]] == ["face"]
    assert updated["imagePath"] == "sample.png"


def test_resolve_execution_providers_prefers_cuda() -> None:
    providers = resolve_execution_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_execution_providers_falls_back_to_cpu() -> None:
    providers = resolve_execution_providers(["CPUExecutionProvider"])

    assert providers == ["CPUExecutionProvider"]


def test_default_batch_range_and_face_ratio() -> None:
    assert DEFAULT_START_BATCH == 22
    assert DEFAULT_END_BATCH == 23
    assert DEFAULT_MIN_FACE_RATIO == 0.01
