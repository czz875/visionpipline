import json
from pathlib import Path

from PIL import Image

from tools.convert.labelme_to_yolo import process_batch


def _write_labelme_sample(json_path: Path, image_path: Path, label: str) -> None:
    Image.new("RGB", (100, 80), color="white").save(image_path)
    json_path.write_text(
        json.dumps(
            {
                "imageWidth": 100,
                "imageHeight": 80,
                "shapes": [
                    {
                        "label": label,
                        "shape_type": "rectangle",
                        "points": [[10, 20], [50, 60]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_process_batch_exports_yolo_class_ids_boxes_and_image_lists(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    phone_json = source / "phone.json"
    cigarette_json = source / "cigarette.json"
    _write_labelme_sample(phone_json, source / "phone.png", "phone")
    _write_labelme_sample(cigarette_json, source / "cigarette.png", "cigarette")
    output = tmp_path / "output"

    process_batch(
        "all",
        [phone_json, cigarette_json],
        output,
        seed=3407,
        class_names=["phone", "cigarette"],
        force=True,
    )

    for filename, class_id in (("phone", 0), ("cigarette", 1)):
        label_files = list((output / "labels").glob(f"*/{filename}.txt"))
        assert len(label_files) == 1
        assert label_files[0].read_text(encoding="utf-8").strip() == (
            f"{class_id} 0.30000 0.50000 0.40000 0.50000"
        )
        split_name = label_files[0].parent.name
        image_path = output / "images" / split_name / f"{filename}.png"
        assert image_path.is_file()
        assert not image_path.samefile(source / f"{filename}.png")
        assert f"images/{split_name}/{filename}.png" in (
            output / f"{split_name}.txt"
        ).read_text(encoding="utf-8")


def test_process_batch_creates_hardlinked_images_when_requested(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    sample_json = source / "sample.json"
    sample_image = source / "sample.png"
    _write_labelme_sample(sample_json, sample_image, "phone")
    output = tmp_path / "output"

    process_batch(
        "all",
        [sample_json],
        output,
        seed=3407,
        class_names=["phone"],
        force=True,
        hardlink_images=True,
    )

    output_images = list((output / "images").glob("*/*.png"))
    assert len(output_images) == 1
    assert output_images[0].samefile(sample_image)
