"""
tools/core/labelme.py

LabelMe 格式相关工具函数。

包括：
- LabelMe JSON 的扫描 / 读取 / 写出；
- LabelMe JSON 与 ``supervision.Detections`` 之间的双向转换。
  （supervision 0.29.x 没有官方 labelme 适配，所以这层桥接放在 tools.core。）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from tools.core.constants import IMAGE_EXTENSIONS, LABELME_EXT

if TYPE_CHECKING:
    import supervision as sv


def list_labelme_files(folder: Path, recursive: bool = True) -> list[Path]:
    """收集目录下 LabelMe JSON 文件，按路径排序。

    Args:
        folder: 待扫描目录。
        recursive: 是否递归子目录。

    Returns:
        按路径排序后的 `.json` 文件列表。
    """
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(folder.glob(pattern))


def find_image_for_json(
    json_path: Path,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> Path | None:
    """在同目录下查找与 JSON 同名的图片文件。

    Args:
        json_path: LabelMe JSON 文件路径。
        extensions: 允许的图片扩展名，按传入顺序优先匹配。

    Returns:
        第一个匹配的图片路径；未找到返回 `None`。
    """
    for ext in extensions:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def find_json_for_image(image_path: Path) -> Path:
    """根据图片路径返回对应的 LabelMe JSON 路径。

    Args:
        image_path: 图片文件路径。

    Returns:
        与图片同目录、同文件名的 `.json` 路径。
    """
    return image_path.with_suffix(LABELME_EXT)


def load_labelme(json_path: Path) -> dict:
    """读取 LabelMe JSON 文件。

    Args:
        json_path: JSON 文件路径。

    Returns:
        JSON 反序列化后的字典。
    """
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_labelme(
    data: dict,
    json_path: Path,
    *,
    indent: int = 2,
) -> None:
    """写出 LabelMe JSON 文件。

    Args:
        data: 待写出的字典。
        json_path: 目标 JSON 文件路径。
        indent: JSON 缩进空格数。
    """
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# LabelMe ↔ supervision.Detections 桥接
# ---------------------------------------------------------------------------
#
# 为什么要放在 tools.core 而不是脚本里？
#   - supervision 0.29.x 缺失 `supervision.dataset.formats.labelme`
#     （见 AGENTS.md §6 已知约束），所以"labelme → 内部表示"必须自己写；
#   - 这层桥接会被 `tools/convert/labelme_to_yolo.py` 和
#     `tools/convert/yolo_to_labelme.py` 两个脚本共用，按 4.6 节原则
#     抽到 tools.core，避免重复。
# ---------------------------------------------------------------------------


def labelme_dict_to_detections(
    labelme_data: dict,
    class_name_to_id: dict[str, int],
) -> sv.Detections:
    """把单个 LabelMe JSON dict 转成 ``supervision.Detections``。

    规则：
    - 只保留 ``shape_type == "rectangle"`` 的形状；
    - 跳过未在 ``class_name_to_id`` 中注册的 label（避免出现未知类别）；
    - 自动把两点矩形归一化（允许 ``[[x1,y1],[x2,y2]]`` 或 ``[[x2,y2],[x1,y1]]``）。

    Args:
        labelme_data: ``load_labelme`` 读出的字典。
        class_name_to_id: 类别名 → YOLO 类别 id 的映射。

    Returns:
        ``supervision.Detections``；当无可用形状时返回 ``Detections.empty()``。
    """
    import supervision as sv

    xyxy_list: list[list[float]] = []
    class_id_list: list[int] = []

    for shape in labelme_data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape.get("label", "")
        if label not in class_name_to_id:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        xyxy_list.append([x_min, y_min, x_max, y_max])
        class_id_list.append(class_name_to_id[label])

    if not xyxy_list:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        class_id=np.array(class_id_list, dtype=int),
    )


def detections_to_labelme_dict(
    detections: sv.Detections,
    *,
    class_names: list[str],
    image_path: str,
    image_width: int,
    image_height: int,
) -> dict:
    """把 ``supervision.Detections`` 转成 LabelMe JSON dict（不写盘）。

    Args:
        detections: 待转换的检测结果。
        class_names: YOLO 类别 id → 类别名 的有序列表。
        image_path: 写入 ``imagePath`` 的图片名（通常只用 basename）。
        image_width: 图片宽度（像素）。
        image_height: 图片高度（像素）。

    Returns:
        可直接交给 ``save_labelme`` 写盘的字典。
    """
    shapes: list[dict] = []
    if len(detections) > 0:
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].tolist()
            class_id = int(detections.class_id[i])
            if class_id < 0 or class_id >= len(class_names):
                continue
            shapes.append({
                "label": class_names[class_id],
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            })
    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }
