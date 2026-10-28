"""参数边界与空输入的回归测试。

覆盖：
- ``--mosaic-block 0`` 触发 ZeroDivisionError（argparse 无取值校验）；
- 空源目录退出码为 0 且不产出 data.yaml，工作流误判成功。
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.annotate.ops import mosaic_region
from tools.convert.labelme_to_yolo import convert_to_yolo


# ---------------------------------------------------------------------------
# 缺陷 16：mosaic-block 0 除零
# ---------------------------------------------------------------------------


def test_mosaic_region_rejects_zero_block() -> None:
    """block_size 为 0 时必须给出明确错误，而不是 ZeroDivisionError。"""
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    box = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)

    with pytest.raises(ValueError, match="block"):
        mosaic_region(image, box, 0)


def test_mosaic_region_rejects_negative_block() -> None:
    """负数同样要拒绝。"""
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    box = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)

    with pytest.raises(ValueError, match="block"):
        mosaic_region(image, box, -4)


def test_mosaic_region_works_with_valid_block() -> None:
    """正常 block 值仍能打码（守护）。"""
    image = np.full((80, 100, 3), 200, dtype=np.uint8)
    box = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)

    out = mosaic_region(image, box, 8)
    assert out.shape == image.shape


def test_mosaic_block_cli_zero_is_rejected() -> None:
    """命令行传 0 应在参数解析阶段就被拒绝。"""
    from tools.annotate.parser import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mosaic-block", "0"])


# ---------------------------------------------------------------------------
# 缺陷 18：空源目录退出码 0
# ---------------------------------------------------------------------------


def test_empty_source_dir_is_an_error(tmp_path: Path) -> None:
    """源目录存在但没有 JSON 时必须报错，不能静默返回成功。"""
    source = tmp_path / "source"
    source.mkdir()
    (source / "0001").mkdir()
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="JSON"):
        convert_to_yolo(source, output, class_names=["phone"])


def test_missing_source_dir_raises(tmp_path: Path) -> None:
    """源目录不存在仍要报错（守护）。"""
    with pytest.raises(FileNotFoundError):
        convert_to_yolo(tmp_path / "nope", tmp_path / "out", class_names=["phone"])


def test_normal_source_still_converts(tmp_path: Path) -> None:
    """有数据的源目录正常转换（守护）。"""
    source = tmp_path / "source" / "0001"
    source.mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(source / "a.png")
    data = {
        "shapes": [
            {"label": "phone", "shape_type": "rectangle", "points": [[10, 10], [50, 50]]}
        ],
        "imageWidth": 100,
        "imageHeight": 80,
    }
    (source / "a.json").write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "output"

    convert_to_yolo(source.parent, output, class_names=["phone"])
    assert (output / "0001" / "data.yaml").is_file()
