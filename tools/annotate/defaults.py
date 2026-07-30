"""
tools/annotate/defaults.py

auto.py 及其 runner / parser 共享的默认参数常量。
"""

from __future__ import annotations

from pathlib import Path

# 项目标准标注输出目录
DEFAULT_OUTPUT_DIR = Path("datasets/01_annotated")

# supervision 后端
DEFAULT_MODEL_TYPE = "yolo"
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_DETR_MODEL = "rtdetr-l.pt"
DEFAULT_SAM3_MODEL = r"weight\sam3.1_multiplex.pt"
DEFAULT_SOURCE = "datasets/0024"
DEFAULT_OUTPUT = "datasets/autolabel"
DEFAULT_FORMAT = "labelme"
DEFAULT_CLASSES = ""
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = ""
DEFAULT_MIN_CONFIDENCE = None
DEFAULT_VERBOSE = False
DEFAULT_COPY_IMAGES = False

# ONNX / SAM 两路打标（--model-type onnx）
DEFAULT_ONNX_MODEL = r"weight\yolov5s-lmk.onnx"
DEFAULT_ONNX_LABEL = "face"
DEFAULT_ONNX_CONF = 0.25
DEFAULT_ONNX_MIN_RATIO = 0.01
DEFAULT_ONNX_TRANSPOSE = False
DEFAULT_ONNX_SCORE_INDICES = "4,15"   # 人脸模型：obj 分 * 关键点置信度
DEFAULT_ONNX_NORMALIZE = True
DEFAULT_SAM_MODEL = r"weight\sam3.1_multiplex.pt"
DEFAULT_SAM_PROMPT = "hand"
DEFAULT_SAM_LABEL = "hand"
DEFAULT_SAM_CONF = 0.25
DEFAULT_SAM_MIN_RATIO = 0.01

# ONNX 覆盖模式（--model-type onnx --reannotate）
DEFAULT_INPUT_ROOT = r"datasets\behavior"
DEFAULT_START_BATCH = 23
DEFAULT_END_BATCH = 37
DEFAULT_KEEP_MIN_RATIO = None
