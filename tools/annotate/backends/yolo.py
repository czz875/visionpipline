"""
tools/annotate/backends/yolo.py

ultralytics YOLO 检测后端：``YOLOLabeler`` 实现 ``AutoLabeler`` 接口，返回
supervision ``sv.Detections``，供 ``auto.py`` 以统一数据集方式导出。
"""

from __future__ import annotations

from pathlib import Path

from tools.annotate.backends.base import AutoLabeler, DetectionsLike


class YOLOLabeler(AutoLabeler):
    """基于 Ultralytics YOLO 的自动标注器。"""

    def __init__(self, model_path: str, predict_kwargs: dict) -> None:
        """加载 YOLO 模型并保存推理参数。

        Args:
            model_path: YOLO 权重路径或模型名称。
            predict_kwargs: 传递给 ``model.predict()`` 的关键字参数。
        """
        from ultralytics import YOLO  # noqa: WPS433 — 延迟导入重型依赖

        self.model = YOLO(model_path)
        self.predict_kwargs = predict_kwargs
        self.classes = [
            self.model.names[i] for i in sorted(self.model.names.keys())
        ]

    def predict(self, image_path: Path) -> DetectionsLike:
        """对单张图片执行 YOLO 推理并转换为 ``sv.Detections``。"""
        import supervision as sv

        result = self.model.predict(source=str(image_path), **self.predict_kwargs)[0]
        return sv.Detections.from_ultralytics(result)
