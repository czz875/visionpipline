"""
tools/annotate/backends/detr.py

ultralytics DETR（RT-DETR）检测后端：``DETRLabeler`` 实现 ``AutoLabeler`` 接口，
返回 supervision ``sv.Detections``，供 ``auto.py`` 以统一数据集方式导出。

与 ``YOLOLabeler`` 几乎一致：``RTDETR.predict`` 的结果同样可用
``sv.Detections.from_ultralytics`` 转换。
"""

from __future__ import annotations

from pathlib import Path

from tools.annotate.backends.base import AutoLabeler, DetectionsLike


class DETRLabeler(AutoLabeler):
    """基于 ultralytics RT-DETR 的自动标注器。"""

    def __init__(self, model_path: str, predict_kwargs: dict) -> None:
        """加载 RT-DETR 模型并保存推理参数。

        Args:
            model_path: RT-DETR 权重路径或模型名称（如 ``rtdetr-l.pt``）。
            predict_kwargs: 传递给 ``model.predict()`` 的关键字参数。
        """
        from ultralytics import RTDETR  # noqa: WPS433 — 延迟导入重型依赖

        self.model = RTDETR(model_path)
        self.predict_kwargs = predict_kwargs
        self.classes = [
            self.model.names[i] for i in sorted(self.model.names.keys())
        ]

    def predict(self, image_path: Path) -> DetectionsLike:
        """对单张图片执行 RT-DETR 推理并转换为 ``sv.Detections``。"""
        import supervision as sv

        result = self.model.predict(source=str(image_path), **self.predict_kwargs)[0]
        return sv.Detections.from_ultralytics(result)
