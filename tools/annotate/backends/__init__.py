"""
tools.annotate.backends
=======================

检测器后端子包，按模型类型分离：

- ``onnx``  —— 通用 ONNX 检测（``OnnxDetector`` + YOLO 风格解码适配）
- ``sam``   —— ultralytics SAM 文本 prompt（``SAMTextDetector`` / ``SAM3Labeler``）
- ``yolo``  —— ultralytics YOLO（``YOLOLabeler``）
- ``detr``  —— ultralytics DETR（``DETRLabeler``，RT-DETR 实现）

``AutoLabeler`` 抽象接口与类型别名见 ``base``。
"""

from __future__ import annotations

from tools.annotate.backends.base import (
    AutoLabeler,
    DatasetLike,
    DetectionsLike,
)
from tools.annotate.backends.detr import DETRLabeler
from tools.annotate.backends.onnx import (
    DecoderType,
    OnnxDetector,
    apply_nms,
    decode_yolo_outputs,
    get_available_execution_providers,
    preprocess_image,
    resolve_execution_providers,
    resolve_input_size,
    scale_boxes,
    xywh_to_xyxy,
)
from tools.annotate.backends.sam import (
    SAM3Labeler,
    SAMTextDetector,
    extract_sam_boxes,
)
from tools.annotate.backends.yolo import YOLOLabeler

__all__ = [
    "AutoLabeler",
    "DatasetLike",
    "DetectionsLike",
    # ONNX
    "OnnxDetector",
    "DecoderType",
    "apply_nms",
    "decode_yolo_outputs",
    "get_available_execution_providers",
    "preprocess_image",
    "resolve_execution_providers",
    "resolve_input_size",
    "scale_boxes",
    "xywh_to_xyxy",
    # SAM
    "SAMTextDetector",
    "SAM3Labeler",
    "extract_sam_boxes",
    # YOLO
    "YOLOLabeler",
    # DETR
    "DETRLabeler",
]
