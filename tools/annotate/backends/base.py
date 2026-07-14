"""
tools/annotate/backends/base.py

检测器后端的公共接口与类型别名。

各后端（YOLO / SAM / DETR）都实现 ``AutoLabeler.predict`` 返回 supervision 的
``sv.Detections``，供 ``tools/annotate/auto.py`` 以统一数据集方式编排导出。
纯 ONNX 后端不走 supervision 数据集式，直接返回 numpy 框（见 ``backends/onnx.py``）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

# 类型别名仅用于内部提示，不在运行时强制导入 supervision，避免 --help 阶段
# 触发昂贵依赖加载。
if TYPE_CHECKING:
    import supervision as sv

    DetectionsLike = sv.Detections
    DatasetLike = sv.DetectionDataset
else:
    DetectionsLike = object
    DatasetLike = object


class AutoLabeler(ABC):
    """自动标注器（supervision 数据集式）的统一接口。

    所有模型后端的标注器（YOLO、SAM3、后续 DETR 等）都只需实现
    ``predict(image_path) -> sv.Detections``，并暴露 ``classes`` 属性供数据集
    构建时使用。
    """

    classes: list[str]

    @abstractmethod
    def predict(self, image_path: Path) -> DetectionsLike:
        """对单张图片进行推理并返回 ``sv.Detections``。"""
        ...
