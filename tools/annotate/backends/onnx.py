"""
tools/annotate/backends/onnx.py

ONNX 检测后端：通用 ONNX 推理封装 ``OnnxDetector``，以及 YOLO 风格输出的
预处理 / 解码 / provider 选择等适配函数。

不绑定任何具体类别（face / hand 等），通过 ``transpose`` / ``score_indices`` /
``decoder`` 适配不同 YOLO 风格输出；返回的是原图像素坐标系下的 ``xyxy`` numpy
框，供 ``auto.py``（ONNX 模式）等编排脚本复用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# =============================================================================
# 1. ONNX 运行环境（provider 选择）
# =============================================================================


def resolve_input_size(shape: list[int | str | None]) -> tuple[int, int]:
    """从 ONNX 输入 shape 里解析输入宽高。"""
    if len(shape) >= 4:
        input_h = int(shape[2] or 640)
        input_w = int(shape[3] or 640)
        return input_w, input_h
    return 640, 640


def resolve_execution_providers(available_providers: list[str]) -> list[str]:
    """优先使用 CUDA，不可用时回退到 CPU。"""
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_available_execution_providers(ort_module) -> list[str]:
    """先预加载 DLL，再获取当前环境可用的 provider。"""
    preload_dlls = getattr(ort_module, "preload_dlls", None)
    if callable(preload_dlls):
        try:
            preload_dlls()
        except Exception as exc:
            print(f"[警告] ONNX Runtime 预加载 CUDA DLL 失败：{exc}")
    return ort_module.get_available_providers()


# =============================================================================
# 2. 预处理 / 框解码
# =============================================================================


def preprocess_image(
    image: np.ndarray,
    input_size: tuple[int, int],
    *,
    normalize: bool,
) -> np.ndarray:
    """按模型输入尺寸直接缩放，并按模型要求决定是否归一化。"""
    input_w, input_h = input_size
    resized = cv2.resize(image, (input_w, input_h))
    resized = resized.astype(np.float32)
    if normalize:
        resized /= 255.0
    tensor = np.transpose(resized, (2, 0, 1))[None, ...]
    return tensor


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """把中心点格式的 xywh 转成 xyxy。"""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    centers_x = boxes[:, 0]
    centers_y = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    converted = np.stack(
        [
            centers_x - widths / 2.0,
            centers_y - heights / 2.0,
            centers_x + widths / 2.0,
            centers_y + heights / 2.0,
        ],
        axis=1,
    )
    return converted.astype(np.float32)


def scale_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int, int],
    input_size: tuple[int, int],
) -> np.ndarray:
    """把输入尺度上的框映射回原图尺度。"""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    image_h, image_w = image_shape[:2]
    input_w, input_h = input_size
    scaled = boxes.copy().astype(np.float32)
    scaled[:, [0, 2]] *= image_w / float(input_w)
    scaled[:, [1, 3]] *= image_h / float(input_h)
    return scaled


def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """对检测框做 NMS，返回保留索引。"""
    if len(boxes) == 0:
        return np.empty((0,), dtype=int)
    nms_boxes = [
        [
            float(box[0]),
            float(box[1]),
            float(box[2] - box[0]),
            float(box[3] - box[1]),
        ]
        for box in boxes
    ]
    kept = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=scores.astype(float).tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if len(kept) == 0:
        return np.empty((0,), dtype=int)
    kept_indices = np.array(kept).reshape(-1).astype(int)
    return kept_indices


def decode_yolo_outputs(
    outputs: list[np.ndarray],
    image_shape: tuple[int, int, int],
    input_size: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    *,
    transpose: bool = False,
    score_indices: tuple[int, ...] = (4,),
) -> np.ndarray:
    """解码通用 YOLO 风格输出（xywh + 多通道置信度相乘）。

    Args:
        transpose: 输出布局是否为 ``(C, N)``（需转置为 ``(N, C)``）。
        score_indices: 参与相乘得到最终分数的通道下标（如人脸模型
            为 ``(4, 15)``，即 obj 分乘关键点置信度）。
    """
    raw = outputs[0][0]
    rows = raw.T if transpose else raw
    scores = np.ones(rows.shape[0], dtype=float)
    for idx in score_indices:
        scores = scores * rows[:, idx]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32)
    boxes = xywh_to_xyxy(rows[keep, :4])
    scores = scores[keep]
    boxes = scale_boxes(boxes, image_shape, input_size)
    kept_indices = apply_nms(boxes, scores, iou_threshold)
    return boxes[kept_indices]


# 解码函数签名：decoder(outputs, image_shape, input_size, conf, iou) -> xyxy 数组
DecoderType = Callable[[list[np.ndarray], tuple[int, int, int], tuple[int, int], float, float], np.ndarray]


# =============================================================================
# 3. ONNX 检测器
# =============================================================================


class OnnxDetector:
    """通用 ONNX 检测器（不绑定具体类别）。

    通过 ``transpose`` / ``score_indices`` 适配不同 YOLO 风格输出；
    若模型输出非常规，可传入自定义 ``decoder``。
    """

    def __init__(
        self,
        model_path: Path,
        label: str,
        conf: float,
        *,
        normalize: bool = True,
        iou_threshold: float = 0.45,
        transpose: bool = False,
        score_indices: tuple[int, ...] = (4,),
        decoder: DecoderType | None = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.label = label
        self.conf = conf
        self.normalize = normalize
        self.iou_threshold = iou_threshold
        self.transpose = transpose
        self.score_indices = score_indices
        self.decoder = decoder

        self.providers = resolve_execution_providers(
            get_available_execution_providers(ort)
        )
        self.session = ort.InferenceSession(
            str(model_path),
            providers=self.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = resolve_input_size(self.session.get_inputs()[0].shape)
        print(
            f"[信息] {model_path.name} providers: "
            f"{','.join(self.session.get_providers())}"
        )

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """执行 ONNX 推理并返回框与标签。"""
        tensor = preprocess_image(
            image,
            self.input_size,
            normalize=self.normalize,
        )
        outputs = self.session.run(None, {self.input_name: tensor})
        if self.decoder is not None:
            boxes = self.decoder(
                outputs, image.shape, self.input_size, self.conf, self.iou_threshold
            )
        else:
            boxes = decode_yolo_outputs(
                outputs,
                image.shape,
                self.input_size,
                self.conf,
                self.iou_threshold,
                transpose=self.transpose,
                score_indices=self.score_indices,
            )
        labels = [self.label] * len(boxes)
        return boxes, labels
