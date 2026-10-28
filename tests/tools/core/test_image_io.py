"""中文路径图片读写的回归测试。

OpenCV 的 ``imread`` / ``imwrite`` 在 Windows 中文路径下会静默失败（返回
``None`` / ``False`` 且不抛异常），标注流程因此会丢图或产出「有 JSON 无图片」
的脏数据。``tools.core`` 的 ``read_image`` / ``write_image`` 必须能正确处理。
"""

from pathlib import Path

import numpy as np
import pytest

from tools.core import image_size, read_image, write_image


def _sample_image() -> np.ndarray:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 2] = 200  # 让内容可区分
    return image


def test_write_then_read_roundtrip_in_chinese_directory(tmp_path: Path) -> None:
    chinese_dir = tmp_path / "中文目录" / "批次一"
    chinese_dir.mkdir(parents=True)
    target = chinese_dir / "图片.png"

    write_image(target, _sample_image())

    assert target.is_file()
    loaded = read_image(target)
    assert loaded is not None
    assert loaded.shape == (8, 12, 3)
    assert int(loaded[0, 0, 2]) == 200


def test_read_image_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_image(tmp_path / "不存在.png") is None


def test_write_image_raises_on_unwritable_path(tmp_path: Path) -> None:
    """写盘失败必须抛错，不能像 cv2.imwrite 那样静默返回 False。"""
    target = tmp_path / "缺失目录" / "图片.png"

    with pytest.raises(OSError):
        write_image(target, _sample_image())


def test_image_size_reports_width_and_height(tmp_path: Path) -> None:
    target = tmp_path / "中文名.png"
    write_image(target, _sample_image())

    assert image_size(target) == (12, 8)


def test_image_size_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        image_size(tmp_path / "不存在.png")
