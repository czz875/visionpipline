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


def extract_sam_boxes(results) -> tuple[np.ndarray, np.ndarray]:
    """从 SAM3 结果中提取矩形框（原图像素坐标系）与类别索引（= 文本提示索引）。

    返回的 ``boxes`` 形状 ``(N, 4)``、``cls`` 形状 ``(N,)``，二者逐框对齐，
    ``cls`` 用于把每个框映射到 ``label`` 列表里的具体类别名。
    """
    if not results:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    box_list: list[np.ndarray] = []
    cls_list: list[np.ndarray] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy
        # SAM 跑在 GPU 时 xyxy / cls 是 CUDA tensor，需先 .cpu() 搬到主机再转 numpy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        array = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        cls = boxes.cls
        if hasattr(cls, "cpu"):
            cls = cls.cpu().numpy()
        cls = np.asarray(cls, dtype=np.int64).reshape(-1)
        if array.shape[0] == 0:
            continue
        box_list.append(array)
        cls_list.append(cls)

    if not box_list:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(box_list, axis=0).astype(np.float32),
        np.concatenate(cls_list, axis=0).astype(np.int64),
    )


class SAMTextDetector:
    """基于本地 SAM3 模型的文本 prompt 检测器（返回 numpy 框 + 逐框类别名）。

    支持同时传入多个文本 prompt（单值或列表），一次 ``set_image`` 后按 prompt 列表
    一次性推理，返回的标签按 ``boxes.cls``（提示索引）映射到 ``label`` 列表。
    """

    def __init__(
        self,
        model_path: Path,
        label: str | list[str],
        conf: float,
        prompt: str | list[str],
        device: str | None = None,
        predictor_cls=None,
    ) -> None:
        self.model_path = model_path
        # 单值或列表统一成列表，逐框按 cls 索引取对应类别名
        self.labels = [label] if isinstance(label, str) else list(label)
        self.prompts = [prompt] if isinstance(prompt, str) else list(prompt)
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
        """按文本 prompt 执行 SAM3 分割，返回外接矩形框与逐框类别名。

        多 prompt 时一次 ``set_image`` 后 ``text=self.prompts`` 调用，按返回的
        ``boxes.cls`` 索引到 ``self.labels`` 得到每个框的具体类别名。
        """
        self.predictor.set_image(image_path)
        results = self.predictor(text=self.prompts)
        boxes, cls = extract_sam_boxes(results)
        keep = (cls >= 0) & (cls < len(self.labels))
        boxes, cls = boxes[keep], cls[keep]
        labels = [self.labels[c] for c in cls]
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
