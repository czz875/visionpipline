from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tools.annotate.reannotate_face_hand_onnx import (
    DEFAULT_END_BATCH,
    DEFAULT_HAND_LABEL,
    DEFAULT_KEEP_EXISTING_HAND,
    DEFAULT_MIN_FACE_RATIO,
    DEFAULT_MIN_HAND_RATIO,
    DEFAULT_START_BATCH,
    blackout_region,
    build_labelme_shapes,
    classify_box_by_ratio,
    classify_face_box,
    collect_blackout_regions,
    extract_existing_label_boxes,
    get_available_execution_providers,
    resolve_execution_providers,
    rewrite_labelme_dict,
    subtract_box_regions,
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


def test_extract_existing_label_boxes_only_keeps_hand_rectangles(tmp_path: Path) -> None:
    json_path = tmp_path / "sample.json"
    json_path.write_text(
        (
            '{"shapes":['
            '{"label":"hand","points":[[10,20],[30,40]],"shape_type":"rectangle"},'
            '{"label":"phone","points":[[1,2],[3,4]],"shape_type":"rectangle"},'
            '{"label":"hand","points":[[5,6]],"shape_type":"rectangle"},'
            '{"label":"hand","points":[[50,60],[70,80]],"shape_type":"polygon"}'
            ']}'
        ),
        encoding="utf-8",
    )

    boxes = extract_existing_label_boxes(json_path, DEFAULT_HAND_LABEL)

    assert boxes.shape == (1, 4)
    assert boxes.tolist() == [[10.0, 20.0, 30.0, 40.0]]


def test_resolve_execution_providers_prefers_cuda() -> None:
    providers = resolve_execution_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_get_available_execution_providers_preloads_dlls() -> None:
    class FakeOrt:
        def __init__(self) -> None:
            self.preloaded = False

        def preload_dlls(self) -> None:
            self.preloaded = True

        def get_available_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    fake_ort = FakeOrt()

    providers = get_available_execution_providers(fake_ort)

    assert fake_ort.preloaded is True
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_execution_providers_falls_back_to_cpu() -> None:
    providers = resolve_execution_providers(["CPUExecutionProvider"])

    assert providers == ["CPUExecutionProvider"]


def test_default_batch_range_and_face_ratio() -> None:
    assert DEFAULT_START_BATCH == 23
    assert DEFAULT_END_BATCH == 37
    assert DEFAULT_MIN_FACE_RATIO == 0.01
    assert DEFAULT_MIN_HAND_RATIO == 0.01
    assert DEFAULT_KEEP_EXISTING_HAND is True


def test_classify_box_by_ratio_marks_small_hand() -> None:
    image_shape = (1000, 1000, 3)
    small_hand = np.array([0, 0, 80, 80], dtype=np.float32)

    assert classify_box_by_ratio(small_hand, image_shape, min_ratio=0.01) is False


def test_blackout_region_fills_box_with_zero() -> None:
    image = np.full((16, 16, 3), 255, dtype=np.uint8)

    result = blackout_region(image.copy(), np.array([4, 4, 12, 12], dtype=np.float32))

    assert np.all(result[4:12, 4:12] == 0)
    assert np.all(result[:4, :4] == 255)


def test_subtract_box_regions_returns_only_non_overlapping_rectangles() -> None:
    small_box = np.array([10, 10, 30, 30], dtype=np.float32)
    kept_box = np.array([20, 10, 40, 30], dtype=np.float32)

    pieces = subtract_box_regions(small_box, np.array([kept_box], dtype=np.float32))

    assert pieces.tolist() == [[10.0, 10.0, 20.0, 30.0]]


def test_collect_blackout_regions_protects_overlapping_large_faces() -> None:
    small_faces = np.array([[10, 10, 30, 30]], dtype=np.float32)
    kept_faces = np.array([[20, 10, 40, 30]], dtype=np.float32)

    regions = collect_blackout_regions(small_faces, kept_faces)

    assert regions.tolist() == [[10.0, 10.0, 20.0, 30.0]]

