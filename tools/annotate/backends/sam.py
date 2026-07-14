"""
tools/annotate/backends/sam.py

ultralytics SAM 文本 prompt 后端，提供两种粒度：

- ``SAMTextDetector``：返回原图像素坐标系下的 ``xyxy`` numpy 框，供
  ``auto.py``（ONNX 模式）直接编排；
- ``SAM3Labeler``：实现 ``AutoLabeler`` 接口，返回 supervision ``sv.Detections``，
  供 ``auto.py`` 以统一数据集方式导出。

两者都基于 ``ultralytics.models.sam.SAM3SemanticPredictor``。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.annotate.backends.base import AutoLabeler, DetectionsLike


def extract_sam_boxes(results) -> np.ndarray:
    """从 SAM3 结果中提取矩形框（原图像素坐标系）。"""
    if not results:
        return np.empty((0, 4), dtype=np.float32)

    collected: list[np.ndarray] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            continue
        # SAM 跑在 GPU 时 xyxy 是 CUDA tensor，需先 .cpu() 搬到主机再转 numpy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        array = np.asarray(xyxy, dtype=np.float32)
        if array.size == 0:
            continue
        collected.append(array.reshape(-1, 4))

    if not collected:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(collected, axis=0).astype(np.float32)


class SAMTextDetector:
    """基于本地 SAM3 模型的文本 prompt 检测器（返回 numpy 框，不绑定具体类别）。"""

    def __init__(
        self,
        model_path: Path,
        label: str,
        conf: float,
        prompt: str,
        device: str | None = None,
        predictor_cls=None,
    ) -> None:
        self.model_path = model_path
        self.label = label
        self.conf = conf
        self.prompt = prompt
        overrides = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": str(model_path),
        }
        if device:
            overrides["device"] = device

        if predictor_cls is None:
            try:
                from ultralytics.models.sam import SAM3SemanticPredictor
            except ImportError as exc:
                raise ImportError(
                    "SAM3 文本 prompt 依赖 ultralytics.models.sam.SAM3SemanticPredictor，"
                    "如缺少 CLIP 依赖，请安装 git+https://github.com/ultralytics/CLIP.git。"
                ) from exc
            predictor_cls = SAM3SemanticPredictor

        self.predictor = predictor_cls(overrides=overrides)

    def predict(self, image_path: Path) -> tuple[np.ndarray, list[str]]:
        """按文本 prompt 执行 SAM3 分割，并返回外接矩形框。"""
        self.predictor.set_image(image_path)
        results = self.predictor(text=[self.prompt])
        boxes = extract_sam_boxes(results)
        labels = [self.label] * len(boxes)
        return boxes, labels


class SAM3Labeler(AutoLabeler):
    """基于本地 ultralytics SAM3 权重的开放词汇分割标注器（supervision 数据集式）。

    使用 ``weight/sam3.1_multiplex.pt``（ultralytics 格式）作为默认后端，
    通过文本提示（prompt）对图中所有匹配实例做实例分割。一次多提示调用即可
    拿到带 ``class_id``（= 提示索引）的 ``sv.Detections``。
    """

    def __init__(
        self,
        model_path: str,
        classes: list[str],
        device: str | None,
        conf_threshold: float = 0.25,
    ) -> None:
        """加载 SAM3 预测器。

        Args:
            model_path: 本地 SAM3 权重路径（ultralytics 格式，如 .pt）。
            classes: 文本提示列表，每个提示对应一个类别。
            device: 推理设备，如 ``cpu`` / ``0`` / ``cuda``；留空则自动选择。
            conf_threshold: 掩膜/框置信度阈值。
        """
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = {
            "conf": conf_threshold,
            "task": "segment",
            "mode": "predict",
            "model": str(model_path),
            "save": False,
            "verbose": False,
        }
        if device:
            overrides["device"] = device
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.classes = classes
        self.conf_threshold = conf_threshold

    def predict(self, image_path: Path) -> DetectionsLike:
        """对单张图片执行 SAM3 文本提示分割并转换为 ``sv.Detections``。

        一次传入全部提示，结果的 ``boxes.cls`` 即为提示索引，据此写入
        ``class_id``，无需对每个提示单独推理。
        """
        import supervision as sv

        self.predictor.set_image(str(image_path))
        results = self.predictor(text=self.classes)
        if not results:
            return sv.Detections.empty()
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return sv.Detections.empty()

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = (
            boxes.conf.cpu().numpy().astype(np.float32)
            if boxes.conf is not None
            else np.ones(len(xyxy), dtype=np.float32)
        )
        # 只保留落在提示类别范围内的结果（过滤异常 cls）。
        keep = (cls >= 0) & (cls < len(self.classes))
        xyxy, cls, conf = xyxy[keep], cls[keep], conf[keep]
        if len(xyxy) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=xyxy,
            class_id=cls,
            confidence=conf,
        )
