"""验证中文路径下确实删除小框并写回打码像素。"""
import json

import numpy as np
import pytest
from PIL import Image

from tools.annotate.parser import build_parser
from tools.annotate.runners.mosaic import run_mosaic


@pytest.mark.parametrize("extension", [".png", ".jpg"])
def test_mosaic_unicode_path_modifies_pixels_and_annotations(tmp_path, extension):
    folder = tmp_path / "中文路径 带空格"
    folder.mkdir()
    image_path = folder / ("图片" + extension)
    pixels = np.random.default_rng(7).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image_path)
    before = image_path.read_bytes()
    annotation = image_path.with_suffix(".json")
    annotation.write_text(json.dumps({
        "imagePath": image_path.name, "imageWidth": 100, "imageHeight": 100,
        "shapes": [{"label": "face", "shape_type": "rectangle", "points": [[10, 10], [15, 15]]}],
    }), encoding="utf-8")
    args = build_parser().parse_args([
        "--mosaic-existing", "--source", str(folder), "--mosaic-labels", "face,hand", "--apply",
    ])
    assert run_mosaic(args) == 0
    assert json.loads(annotation.read_text(encoding="utf-8"))["shapes"] == []
    assert image_path.read_bytes() != before
