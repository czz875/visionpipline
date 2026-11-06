"""
tools/core/image_io.py

图片读写工具，兼容 Windows 中文路径。

OpenCV 的 ``cv2.imread`` / ``cv2.imwrite`` 在 Windows 上按本地编码解析路径，
遇到中文或非 ASCII 路径时分别返回 ``None`` / ``False``，且不抛异常。这会让
标注流程「静默丢图」或产出「有 JSON 无图片」的脏数据，因此项目内统一改用
按字节读写文件的 ``cv2.imdecode`` / ``cv2.imencode``。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools.core.file_io import atomic_write_bytes


def read_image(image_path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """按字节读取图片，兼容中文路径。

    Args:
        image_path: 图片路径。
        flags: ``cv2`` 的读取标志，默认按彩色读取。

    Returns:
        图像数组；读取失败时返回 ``None``（与 ``cv2.imread`` 语义一致）。
    """
    try:
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def write_image(
    image_path: Path | str,
    image: np.ndarray,
    params: list[int] | None = None,
) -> None:
    """原子写出图片，兼容中文路径，并隔离其他硬链接。

    编码或写盘失败时抛 ``OSError``。此处必须检查返回值：``cv2.imwrite`` 在中文
    路径下只返回 ``False`` 而不抛异常，会造成「图片没写出但 JSON 已写出」。

    Args:
        image_path: 目标图片路径，扩展名决定编码格式。
        image: 待写出的图像数组。
        params: 传给 ``cv2.imencode`` 的编码参数，如
            ``[cv2.IMWRITE_PNG_COMPRESSION, 9]``。

    Raises:
        OSError: 编码失败或写盘失败。
    """
    image_path = Path(image_path)
    suffix = image_path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image, params or [])
    if not success:
        raise OSError(f"图片编码失败：{image_path}")
    try:
        atomic_write_bytes(image_path, memoryview(encoded))
    except OSError as e:
        raise OSError(f"图片写盘失败：{image_path}（{e}）") from e


def image_size(image_path: Path | str) -> tuple[int, int]:
    """读取图片尺寸，兼容中文路径。

    Returns:
        ``(width, height)``。

    Raises:
        OSError: 图片无法读取或尺寸非法。
    """
    image = read_image(image_path)
    if image is None:
        raise OSError(f"无法读取图片：{image_path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise OSError(f"图片尺寸非法：{image_path}")
    return width, height
