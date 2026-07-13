from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tools.annotate.reannotate_face_hand_onnx import (
    build_labelme_shapes,
    classify_face_box,
    mosaic_region,
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
