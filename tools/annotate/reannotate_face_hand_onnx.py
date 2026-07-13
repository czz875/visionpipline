"""
tools/annotate/reannotate_face_hand_onnx.py
使用人脸 ONNX 覆盖 face，保留现有 JSON 的 hand，
并对过小框执行直接涂黑后删除的专项流程。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import find_json_for_image, list_images, load_labelme, save_labelme


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_FACE_MODEL = r"weight\yolov5s-lmk.onnx"
DEFAULT_SAM_HAND_MODEL = r"weight\sam3.1_multiplex.pt"
DEFAULT_SAM_HAND_PROMPT = "hand"
DEFAULT_HAND_MODEL = DEFAULT_SAM_HAND_MODEL
DEFAULT_INPUT_ROOT = r"datasets\behavior"
DEFAULT_START_BATCH = 23
DEFAULT_END_BATCH = 37
DEFAULT_MIN_FACE_RATIO = 0.01
DEFAULT_MIN_HAND_RATIO = 0.01
DEFAULT_FACE_CONF = 0.25
DEFAULT_HAND_CONF = 0.25
DEFAULT_FACE_IOU = 0.45
DEFAULT_HAND_IOU = 0.45
DEFAULT_DRY_RUN = True
DEFAULT_FACE_LABEL = "face"
DEFAULT_HAND_LABEL = "hand"
DEFAULT_KEEP_EXISTING_HAND = True


# =============================================================================
# 2. 数据结构
# =============================================================================


@dataclass
class ProcessResult:
    image: np.ndarray
    boxes: np.ndarray
    labels: list[str]
    face_blackout_count: int
    hand_blackout_count: int
    face_count: int
    hand_count: int


# =============================================================================
# 3. 纯函数
# =============================================================================


def classify_face_box(
    box: np.ndarray,
    image_shape: tuple[int, int, int],
    min_ratio: float,
) -> bool:
    """判断人脸框面积占比是否达到保留阈值。"""
    image_h, image_w = image_shape[:2]
    image_area = float(image_w * image_h)
    if image_area <= 0:
        return False
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return (box_area / image_area) >= min_ratio


def classify_box_by_ratio(
    box: np.ndarray,
    image_shape: tuple[int, int, int],
    min_ratio: float,
) -> bool:
    """按面积占比判断框是否达到保留阈值。"""
    return classify_face_box(box, image_shape, min_ratio)


def clip_box(box: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """把框裁剪到图像范围内。"""
    image_h, image_w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    clipped = np.array(
        [
            np.clip(x1, 0, image_w),
            np.clip(y1, 0, image_h),
            np.clip(x2, 0, image_w),
            np.clip(y2, 0, image_h),
        ],
        dtype=np.float32,
    )
    return clipped


def blackout_region(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """把框内区域直接填充为纯黑。"""
    clipped = clip_box(box, image.shape)
    x1, y1, x2, y2 = [int(round(v)) for v in clipped.tolist()]
    if x2 <= x1 or y2 <= y1:
        return image
    image[y1:y2, x1:x2] = 0
    return image


def subtract_box_regions(box: np.ndarray, keep_boxes: np.ndarray) -> np.ndarray:
    """从待删除小框中减去所有保留大框重叠区域。"""
    pieces = [clip_box(box, (10**9, 10**9, 3))]
    for keep_box in keep_boxes:
        next_pieces: list[np.ndarray] = []
        for piece in pieces:
            x1, y1, x2, y2 = [float(v) for v in piece.tolist()]
            kx1, ky1, kx2, ky2 = [float(v) for v in keep_box.tolist()]
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                next_pieces.append(piece)
                continue
            if y1 < iy1:
                next_pieces.append(np.array([x1, y1, x2, iy1], dtype=np.float32))
            if iy2 < y2:
                next_pieces.append(np.array([x1, iy2, x2, y2], dtype=np.float32))
            if x1 < ix1:
                next_pieces.append(np.array([x1, iy1, ix1, iy2], dtype=np.float32))
            if ix2 < x2:
                next_pieces.append(np.array([ix2, iy1, x2, iy2], dtype=np.float32))
        pieces = [piece for piece in next_pieces if piece[2] > piece[0] and piece[3] > piece[1]]
        if not pieces:
            return np.empty((0, 4), dtype=np.float32)
    return np.array(pieces, dtype=np.float32) if pieces else np.empty((0, 4), dtype=np.float32)


def collect_blackout_regions(
    removed_boxes: np.ndarray,
    kept_boxes: np.ndarray,
) -> np.ndarray:
    """收集所有待删除小框的非重叠残余区域。"""
    regions: list[np.ndarray] = []
    for removed_box in removed_boxes:
        for piece in subtract_box_regions(removed_box, kept_boxes):
            regions.append(piece)
    return np.array(regions, dtype=np.float32) if regions else np.empty((0, 4), dtype=np.float32)


def split_boxes_by_ratio(
    boxes: np.ndarray,
    image_shape: tuple[int, int, int],
    min_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按面积占比分成保留框与删除框。"""
    kept: list[np.ndarray] = []
    removed: list[np.ndarray] = []
    for box in boxes:
        if classify_box_by_ratio(box, image_shape, min_ratio):
            kept.append(box)
        else:
            removed.append(box)
    kept_array = np.array(kept, dtype=np.float32) if kept else np.empty((0, 4), dtype=np.float32)
    removed_array = np.array(removed, dtype=np.float32) if removed else np.empty((0, 4), dtype=np.float32)
    return kept_array, removed_array


def build_labelme_shapes(boxes: np.ndarray, labels: list[str]) -> list[dict]:
    """把矩形框与标签转成 LabelMe shapes。"""
    shapes: list[dict] = []
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        shapes.append({
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
        })
    return shapes


def rewrite_labelme_dict(
    json_path: Path,
    boxes: np.ndarray,
    labels: list[str],
    image_shape: tuple[int, int, int],
    image_name: str,
) -> dict:
    """用新的人脸与手部框整体覆盖 LabelMe。"""
    if json_path.exists():
        data = load_labelme(json_path)
    else:
        data = {}
    image_h, image_w = image_shape[:2]
    data["version"] = data.get("version", "5.3.1")
    data["flags"] = data.get("flags", {})
    data["shapes"] = build_labelme_shapes(boxes, labels)
    data["imagePath"] = image_name
    data["imageData"] = None
    data["imageHeight"] = int(image_h)
    data["imageWidth"] = int(image_w)
    return data


def extract_existing_label_boxes(json_path: Path, label: str) -> np.ndarray:
    """从现有 LabelMe JSON 中提取指定标签的矩形框。"""
    if not json_path.exists():
        return np.empty((0, 4), dtype=np.float32)
    data = load_labelme(json_path)
    boxes: list[list[float]] = []
    for shape in data.get("shapes", []):
        if shape.get("label") != label:
            continue
        if shape.get("shape_type") != "rectangle":
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        (x1, y1), (x2, y2) = points[:2]
        boxes.append([
            float(min(x1, x2)),
            float(min(y1, y2)),
            float(max(x1, x2)),
            float(max(y1, y2)),
        ])
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def extract_sam_boxes(results) -> np.ndarray:
    """从 SAM3 结果中提取矩形框。"""
    if not results:
        return np.empty((0, 4), dtype=np.float32)

    collected: list[np.ndarray] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            continue
        array = np.asarray(xyxy, dtype=np.float32)
        if array.size == 0:
            continue
        collected.append(array.reshape(-1, 4))

    if not collected:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(collected, axis=0).astype(np.float32)


# =============================================================================
# 4. 文件发现
# =============================================================================


def iter_batch_dirs(root: Path, start_batch: int, end_batch: int) -> list[Path]:
    """收集指定 batch 范围内存在的目录。"""
    batch_dirs: list[Path] = []
    for batch_id in range(start_batch, end_batch + 1):
        batch_dir = root / f"{batch_id:04d}"
        if batch_dir.is_dir():
            batch_dirs.append(batch_dir)
    return batch_dirs


def iter_image_files(root: Path, start_batch: int, end_batch: int) -> list[Path]:
    """收集指定 batch 范围内的全部图片。"""
    image_paths: list[Path] = []
    for batch_dir in iter_batch_dirs(root, start_batch, end_batch):
        image_paths.extend(list_images(batch_dir, recursive=True))
    return image_paths


# =============================================================================
# 5. ONNX 适配
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


def decode_face_outputs(
    outputs: list[np.ndarray],
    image_shape: tuple[int, int, int],
    input_size: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """解码人脸模型输出。"""
    rows = outputs[0][0]
    scores = rows[:, 4] * rows[:, 15]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32)
    boxes = xywh_to_xyxy(rows[keep, :4])
    scores = scores[keep]
    boxes = scale_boxes(boxes, image_shape, input_size)
    kept_indices = apply_nms(boxes, scores, iou_threshold)
    return boxes[kept_indices]


def decode_hand_outputs(
    outputs: list[np.ndarray],
    image_shape: tuple[int, int, int],
    input_size: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """解码手部模型输出。"""
    rows = outputs[0][0].transpose(1, 0)
    scores = rows[:, 4]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32)
    boxes = xywh_to_xyxy(rows[keep, :4])
    scores = scores[keep]
    boxes = scale_boxes(boxes, image_shape, input_size)
    kept_indices = apply_nms(boxes, scores, iou_threshold)
    return boxes[kept_indices]


class OnnxDetector:
    """ONNX 检测器。"""

    def __init__(self, model_path: Path, label: str, conf: float) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.label = label
        self.conf = conf
        self.providers = resolve_execution_providers(
            get_available_execution_providers(ort)
        )
        self.session = ort.InferenceSession(
            str(model_path),
            providers=self.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = resolve_input_size(self.session.get_inputs()[0].shape)
        self.normalize = label == DEFAULT_FACE_LABEL
        self.iou_threshold = (
            DEFAULT_FACE_IOU if label == DEFAULT_FACE_LABEL else DEFAULT_HAND_IOU
        )
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
        if self.label == DEFAULT_FACE_LABEL:
            boxes = decode_face_outputs(
                outputs=outputs,
                image_shape=image.shape,
                input_size=self.input_size,
                conf_threshold=self.conf,
                iou_threshold=self.iou_threshold,
            )
        else:
            boxes = decode_hand_outputs(
                outputs=outputs,
                image_shape=image.shape,
                input_size=self.input_size,
                conf_threshold=self.conf,
                iou_threshold=self.iou_threshold,
            )
        labels = [self.label] * len(boxes)
        return boxes, labels


class SAMTextDetector:
    """基于本地 SAM3 模型的文本 prompt 检测器。"""

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


def combine_boxes(face_boxes: list[np.ndarray], hand_boxes: np.ndarray) -> np.ndarray:
    """把保留的人脸框与手部框合并成统一数组。"""
    parts: list[np.ndarray] = []
    if face_boxes:
        parts.append(np.array(face_boxes, dtype=np.float32))
    if len(hand_boxes):
        parts.append(hand_boxes.astype(np.float32))
    if not parts:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def sanitize_boxes(boxes: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """裁剪并过滤无效框。"""
    sanitized: list[np.ndarray] = []
    for box in boxes:
        clipped = clip_box(box, image_shape)
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        sanitized.append(clipped)
    if not sanitized:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(sanitized, dtype=np.float32)


def process_image_file(
    image_path: Path,
    face_detector: OnnxDetector,
    hand_detector: SAMTextDetector | None,
    min_face_ratio: float,
    min_hand_ratio: float,
    dry_run: bool,
    keep_existing_hand: bool,
) -> ProcessResult:
    """对单张图片执行 face / hand 重标注。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")

    json_path = find_json_for_image(image_path)
    face_boxes, _ = face_detector.predict(image)
    if keep_existing_hand:
        hand_boxes = extract_existing_label_boxes(json_path, DEFAULT_HAND_LABEL)
    else:
        if hand_detector is None:
            raise ValueError("未提供手部检测器。")
        hand_boxes, _ = hand_detector.predict(image_path)
    face_boxes = sanitize_boxes(face_boxes, image.shape)
    hand_boxes = sanitize_boxes(hand_boxes, image.shape)

    kept_faces, removed_faces = split_boxes_by_ratio(
        face_boxes, image.shape, min_face_ratio
    )
    kept_hands, removed_hands = split_boxes_by_ratio(
        hand_boxes, image.shape, min_hand_ratio
    )

    face_blackouts = collect_blackout_regions(removed_faces, kept_faces)
    hand_blackouts = collect_blackout_regions(removed_hands, kept_hands)

    if not dry_run:
        for box in face_blackouts:
            image = blackout_region(image, box)
        for box in hand_blackouts:
            image = blackout_region(image, box)

    final_boxes = combine_boxes(list(kept_faces), kept_hands)
    final_labels = [DEFAULT_FACE_LABEL] * len(kept_faces)
    final_labels.extend([DEFAULT_HAND_LABEL] * len(kept_hands))
    return ProcessResult(
        image=image,
        boxes=final_boxes,
        labels=final_labels,
        face_blackout_count=len(face_blackouts),
        hand_blackout_count=len(hand_blackouts),
        face_count=len(kept_faces),
        hand_count=len(kept_hands),
    )


# =============================================================================
# 7. 运行逻辑
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(
        description="用人脸 ONNX 覆盖 face，并保留现有 JSON 中的 hand 进行小框涂黑过滤。",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path(DEFAULT_FACE_MODEL),
        help=f"人脸 ONNX 模型路径（默认：{DEFAULT_FACE_MODEL}）。",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path(DEFAULT_HAND_MODEL),
        help=f"保留兼容性的手部模型参数（默认：{DEFAULT_HAND_MODEL}）。",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(DEFAULT_INPUT_ROOT),
        help=f"behavior 根目录（默认：{DEFAULT_INPUT_ROOT}）。",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=DEFAULT_START_BATCH,
        help=f"起始 batch 编号（默认：{DEFAULT_START_BATCH}）。",
    )
    parser.add_argument(
        "--end-batch",
        type=int,
        default=DEFAULT_END_BATCH,
        help=f"结束 batch 编号（默认：{DEFAULT_END_BATCH}）。",
    )
    parser.add_argument(
        "--min-face-ratio",
        type=float,
        default=DEFAULT_MIN_FACE_RATIO,
        help=f"保留人脸的最小面积占比（默认：{DEFAULT_MIN_FACE_RATIO}）。",
    )
    parser.add_argument(
        "--face-conf",
        type=float,
        default=DEFAULT_FACE_CONF,
        help=f"人脸置信度阈值（默认：{DEFAULT_FACE_CONF}）。",
    )
    parser.add_argument(
        "--hand-conf",
        type=float,
        default=DEFAULT_HAND_CONF,
        help=f"保留兼容性的手部阈值参数（默认：{DEFAULT_HAND_CONF}）。",
    )
    parser.add_argument(
        "--min-hand-ratio",
        type=float,
        default=DEFAULT_MIN_HAND_RATIO,
        help=f"保留手部的最小面积占比（默认：{DEFAULT_MIN_HAND_RATIO}）。",
    )
    parser.add_argument(
        "--keep-existing-hand",
        action="store_true",
        default=DEFAULT_KEEP_EXISTING_HAND,
        help="直接保留现有 JSON 里的 hand 矩形框，不调用 hand 模型。",
    )
    parser.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="真正写回图片与 JSON；默认仅 dry-run 统计。",
    )
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


def run(args: argparse.Namespace) -> int:
    """执行批量重标注。"""
    image_paths = iter_image_files(args.input_root, args.start_batch, args.end_batch)
    if not image_paths:
        print("[错误] 指定 batch 范围内没有可处理图片。", file=sys.stderr)
        return 1

    face_detector = OnnxDetector(args.face_model, DEFAULT_FACE_LABEL, args.face_conf)
    hand_detector = None
    if not args.keep_existing_hand:
        hand_detector = SAMTextDetector(
            model_path=args.hand_model,
            label=DEFAULT_HAND_LABEL,
            conf=args.hand_conf,
            prompt=args.hand_prompt,
        )

    total_face = 0
    total_hand = 0
    total_mosaic = 0
    total_json = 0

    for image_path in tqdm(image_paths, desc="重标注", unit="img"):
        result = process_image_file(
            image_path=image_path,
            face_detector=face_detector,
            hand_detector=hand_detector,
            min_face_ratio=args.min_face_ratio,
            min_hand_ratio=args.min_hand_ratio,
            dry_run=args.dry_run,
            keep_existing_hand=args.keep_existing_hand,
        )
        total_face += result.face_count
        total_hand += result.hand_count
        total_mosaic += result.face_blackout_count + result.hand_blackout_count
        total_json += 1

        if args.dry_run:
            continue

        json_path = find_json_for_image(image_path)
        data = rewrite_labelme_dict(
            json_path,
            result.boxes,
            result.labels,
            result.image.shape,
            image_path.name,
        )
        cv2.imwrite(str(image_path), result.image)
        save_labelme(data, json_path)

    mode = "预览" if args.dry_run else "完成"
    print(f"[{mode}] 图片数：{len(image_paths)}")
    print(f"[{mode}] 保留 face：{total_face}")
    print(f"[{mode}] 保留 hand：{total_hand}")
    print(f"[{mode}] 小框涂黑并删除：{total_mosaic}")
    print(f"[{mode}] 覆盖 JSON：{total_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
