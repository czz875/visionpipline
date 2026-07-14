"""
tools/annotate/backends/detr.py

ultralytics DETR（RT-DETR）检测后端。

当前为预留占位：``auto.py`` 选择 ``--model-type detr`` 时会创建
``DETRLabeler`` 并在 ``predict`` 抛出未实现错误，待后续按 ultralytics
``RTDETR`` 接入。接口与其它后端保持一致（实现 ``AutoLabeler``）。
"""

from __future__ import annotations

from pathlib import Path

from tools.annotate.backends.base import AutoLabeler, DetectionsLike


class DETRLabeler(AutoLabeler):
    """基于 ultralytics RT-DETR 的自动标注器（预留，尚未实现）。"""

    def __init__(self, model_path: str, predict_kwargs: dict) -> None:
        self.model_path = model_path
        self.predict_kwargs = predict_kwargs
        self.classes = []

    def predict(self, image_path: Path) -> DetectionsLike:
        """执行 DETR 推理（尚未实现）。"""
        raise NotImplementedError("DETR 标注器尚未实现")
