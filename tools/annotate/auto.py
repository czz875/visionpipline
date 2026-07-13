"""
tools/annotate/auto.py
基于 supervision 的批量自动标注脚本，支持 YOLO / SAM3 / DETR(预留) 等多种
模型后端，输出统一数据集格式。

数据流：模型推理 -> sv.Detections -> 类别与置信度过滤 -> 累计为
sv.DetectionDataset -> 导出 YOLO / LabelMe / COCO。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

# 允许以 `python tools/annotate/auto.py` 直接运行；本脚本在写入
# sys.path 之后再延迟导入 supervision 与 ultralytics，避免在 --help 阶段
# 触发昂贵的依赖加载。
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_MODEL_TYPE = "yolo"
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_SAM3_MODEL = "facebook/sam3"
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


# =============================================================================
# 2. 常量与类型别名
# =============================================================================

from tools.core import IMAGE_EXTENSIONS as IMG_EXTS, list_images  # noqa: E402

# 脚本属于工具层，类型仅用于内部提示，不参与运行时检查。
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    import supervision as sv

    DetectionsLike = sv.Detections
    DatasetLike = sv.DetectionDataset
else:
    DetectionsLike = object
    DatasetLike = object


# =============================================================================
# 3. 本地导入辅助
# =============================================================================


def ensure_local_supervision_import() -> None:
    """将仓库 `src/` 插入 `sys.path` 前部，优先使用本地开发版 supervision。

    在脚本入口处调用一次即可，避免安装版覆盖仓库内的 `as_labelme`、
    `as_coco` 等较新接口。
    """
    repo_root = Path(__file__).resolve().parent.parent
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


# =============================================================================
# 4. 标注器抽象与实现
# =============================================================================


class AutoLabeler(ABC):
    """自动标注器的统一接口。

    所有模型后端的标注器（YOLO、SAM3、后续 DETR 等）都只需实现
    `predict(image_path) -> sv.Detections`，并暴露 `classes` 属性供数据集
    构建时使用。
    """

    classes: list[str]

    @abstractmethod
    def predict(self, image_path: Path) -> DetectionsLike:
        """对单张图片进行推理并返回 `sv.Detections`。"""
        ...


class YOLOLabeler(AutoLabeler):
    """基于 Ultralytics YOLO 的自动标注器。"""

    def __init__(self, model_path: str, predict_kwargs: dict) -> None:
        """加载 YOLO 模型并保存推理参数。

        Args:
            model_path: YOLO 权重路径或模型名称。
            predict_kwargs: 传递给 `model.predict()` 的关键字参数。
        """
        from ultralytics import YOLO  # noqa: WPS433 — 延迟导入重型依赖

        self.model = YOLO(model_path)
        self.predict_kwargs = predict_kwargs
        self.classes = [
            self.model.names[i] for i in sorted(self.model.names.keys())
        ]

    def predict(self, image_path: Path) -> DetectionsLike:
        """对单张图片执行 YOLO 推理并转换为 `sv.Detections`。"""
        import supervision as sv

        result = self.model.predict(source=str(image_path), **self.predict_kwargs)[0]
        return sv.Detections.from_ultralytics(result)


class SAM3Labeler(AutoLabeler):
    """基于 Hugging Face Transformers SAM3 的开放词汇分割标注器。

    使用 `facebook/sam3` 作为默认后端，通过文本提示（prompt）对图中所有
    匹配实例进行实例分割。输出 `sv.Detections` 包含 `mask`、`xyxy`、
    `confidence` 与 `class_id`（prompt 索引）。
    """

    def __init__(
        self,
        model_id: str,
        classes: list[str],
        device: str | None,
        conf_threshold: float = 0.5,
    ) -> None:
        """加载 SAM3 处理器与模型。

        Args:
            model_id: Hugging Face model id，例如 `facebook/sam3`。
            classes: 文本提示列表，每个提示对应一个类别。
            device: 推理设备，如 `cpu`、`cuda`；留空则自动选择。
            conf_threshold: 掩膜置信度阈值。
        """
        import torch  # noqa: WPS433 — 延迟导入重型依赖
        from transformers import Sam3Model, Sam3Processor  # noqa: WPS433

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.classes = classes
        self.conf_threshold = conf_threshold

    def predict(self, image_path: Path) -> DetectionsLike:
        """对单张图片执行 SAM3 文本提示分割并转换为 `sv.Detections`。

        为了明确每个实例对应的类别，这里对每个文本提示分别推理一次，
        将提示索引作为 `class_id` 写入结果。后续可优化为单次多提示推理。
        """
        import supervision as sv
        import torch  # noqa: WPS433 — 延迟导入重型依赖
        from PIL import Image  # noqa: WPS433

        image = Image.open(image_path).convert("RGB")
        target_size = [image.size[::-1]]

        all_masks: list[npt.NDArray[np.bool_]] = []
        all_boxes: list[npt.NDArray[np.number]] = []
        all_scores: list[float] = []
        all_class_ids: list[int] = []

        for class_id, prompt in enumerate(self.classes):
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.conf_threshold,
                mask_threshold=self.conf_threshold,
                target_sizes=target_size,
            )[0]

            if not result or "masks" not in result or len(result["masks"]) == 0:
                continue

            masks = result["masks"].cpu().numpy() > 0
            num_instances = masks.shape[0]
            all_masks.extend(masks)

            if "boxes" in result:
                boxes = result["boxes"].cpu().numpy()
            else:
                from supervision.detection.utils.converters import mask_to_xyxy

                boxes = mask_to_xyxy(masks)
            all_boxes.append(boxes)

            scores = (
                result["scores"].cpu().numpy()
                if "scores" in result
                else np.ones(num_instances, dtype=np.float32)
            )
            all_scores.extend(scores.tolist())
            all_class_ids.extend([class_id] * num_instances)

        if not all_masks:
            return sv.Detections.empty()

        xyxy = np.concatenate(all_boxes, axis=0).astype(np.float32)
        mask = np.stack(all_masks, axis=0)
        return sv.Detections(
            xyxy=xyxy,
            mask=mask,
            confidence=np.array(all_scores, dtype=np.float32),
            class_id=np.array(all_class_ids, dtype=int),
        )


# =============================================================================
# 5. 路径与文件发现
# =============================================================================

# `list_images` 与 `IMG_EXTS` 已从 `tools.core` 导入复用。


# =============================================================================
# 6. 类别解析
# =============================================================================


def parse_class_ids(
    arg: str | None, model_names: dict[int, str]
) -> set[int] | None:
    """将 `--classes` 字符串解析为类别 ID 集合。

    支持 `0,2,3` 与 `person,car` 两种形式；未知名称打印警告但不中断。
    返回 `None` 表示不过滤。
    """
    if not arg:
        return None
    items = [s.strip() for s in arg.split(",") if s.strip()]
    if not items:
        return None
    ids: set[int] = set()
    name_to_id = {name.lower(): idx for idx, name in model_names.items()}
    for it in items:
        if it.isdigit():
            ids.add(int(it))
            continue
        mapped = name_to_id.get(it.lower())
        if mapped is not None:
            ids.add(mapped)
        else:
            print(f"[警告] 未识别的类别：{it}", file=sys.stderr)
    return ids or None


def filter_by_class_ids(
    detections: DetectionsLike, keep_class_ids: set[int] | None
) -> DetectionsLike:
    """按类别 ID 集合筛选 `Detections`，未指定集合时原样返回。"""
    if keep_class_ids is None or len(detections) == 0:
        return detections
    mask = np.isin(detections.class_id, list(keep_class_ids))
    return detections[mask]


# =============================================================================
# 7. 置信度过滤
# =============================================================================


def filter_by_confidence(
    detections: DetectionsLike, min_confidence: float | None
) -> DetectionsLike:
    """按最小置信度筛选 `Detections`，未指定时原样返回。"""
    if min_confidence is None or len(detections) == 0:
        return detections
    if detections.confidence is None:
        return detections
    mask = detections.confidence >= min_confidence
    return detections[mask]


# =============================================================================
# 8. 数据集构建
# =============================================================================


def build_detection_dataset(
    labeler: AutoLabeler,
    images: list[Path],
    keep_class_ids: set[int] | None,
    min_confidence: float | None,
) -> DatasetLike:
    """逐张图推理并累积为 `sv.DetectionDataset`。

    无检测结果的图仍保留为空的 `Detections`，确保图像集合完整。
    """
    import supervision as sv

    annotations: dict[str, sv.Detections] = {}
    image_paths: list[str] = []
    for img_path in tqdm(images, desc="推理中"):
        detections = labeler.predict(img_path)
        detections = filter_by_class_ids(detections, keep_class_ids)
        detections = filter_by_confidence(detections, min_confidence)
        key = str(img_path.resolve())
        image_paths.append(key)
        # 即便为空也写入，保证导出时图像集合完整。
        annotations[key] = detections if len(detections) else sv.Detections.empty()
    return sv.DetectionDataset(
        classes=labeler.classes, images=image_paths, annotations=annotations
    )


# =============================================================================
# 9. 数据集导出
# =============================================================================


def _unique_stem(path: Path, used: set[str]) -> str:
    """为 `path` 生成不重复的 stem，冲突时逐层追加父目录名。

    例如 `datasets/15/capture_000000.png` 与 `datasets/16/capture_000000.png`
    会分别生成 `15_capture_000000` 与 `16_capture_000000`。
    """
    if path.stem not in used:
        return path.stem
    parts = [path.stem]
    for parent in path.parents:
        candidate = "_".join([parent.name] + parts)
        if candidate not in used:
            return candidate
        parts.insert(0, parent.name)
    i = 1
    while f"{path.stem}_{i}" in used:
        i += 1
    return f"{path.stem}_{i}"


def _prepare_labelme_dataset(
    dataset: DatasetLike, output_dir: Path
) -> DatasetLike:
    """将图片复制到 `output_dir` 并使用唯一 basename，返回新的 dataset。

    LabelMe 要求每张图片与其 `.json` 标注文件同名同目录。当不同子目录存在
    同名图片时，通过把父目录名拼入文件名来避免 JSON 输出冲突。
    """
    import supervision as sv

    used_stems: set[str] = set()
    new_image_paths: list[str] = []
    new_annotations: dict[str, sv.Detections] = {}
    for image_path, _image, detections in dataset:
        src = Path(image_path)
        stem = _unique_stem(src, used_stems)
        used_stems.add(stem)
        dst = output_dir / f"{stem}{src.suffix}"
        shutil.copy2(str(src), str(dst))
        new_path = str(dst.resolve())
        new_image_paths.append(new_path)
        new_annotations[new_path] = detections
    return sv.DetectionDataset(
        classes=dataset.classes,
        images=new_image_paths,
        annotations=new_annotations,
    )


def export_detection_dataset(
    dataset: DatasetLike, output: Path, fmt: str, copy_images: bool
) -> None:
    """根据 `fmt` 选择 YOLO / LabelMe / COCO 中的一种导出方式。

    LabelMe 的标准布局是图片与同名 `.json` 同目录，因此 labelme 模式下
    会先把图片复制到输出目录并赋予唯一 basename，再写出 JSON。
    其他格式仍按 `images/` 与标注目录分开的常规布局写出。
    """
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    if fmt == "yolo":
        if copy_images:
            images_dir.mkdir(parents=True, exist_ok=True)
        (output / "labels").mkdir(parents=True, exist_ok=True)
        dataset.as_yolo(
            images_directory_path=str(images_dir) if copy_images else None,
            annotations_directory_path=str(output / "labels"),
            data_yaml_path=str(output / "data.yaml"),
        )
    elif fmt == "labelme":
        # LabelMe 图片与 JSON 同目录，且要求 basename 唯一；通过
        # _prepare_labelme_dataset 复制图片并生成唯一文件名。
        if not copy_images:
            print(
                "[信息] LabelMe 格式需要图片与 JSON 同目录，已自动复制图片到输出目录。",
                file=sys.stderr,
            )
        labelme_dataset = _prepare_labelme_dataset(dataset, output)
        from supervision.dataset.formats.labelme import save_labelme_annotations

        save_labelme_annotations(labelme_dataset, str(output))
    elif fmt == "coco":
        if copy_images:
            images_dir.mkdir(parents=True, exist_ok=True)
        dataset.as_coco(
            images_directory_path=str(images_dir) if copy_images else None,
            annotations_path=str(output / "annotations.json"),
        )
    else:
        raise ValueError(f"不支持的导出格式：{fmt}")


# =============================================================================
# 10. 命令行参数
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """构造脚本的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="基于 supervision 的批量自动标注脚本，支持 YOLO / SAM3 / DETR(预留)",
    )
    parser.add_argument(
        "--model-type",
        choices=["yolo", "sam3", "detr"],
        default=DEFAULT_MODEL_TYPE,
        help="标注器类型，yolo 为检测框，sam3 为开放词汇分割，detr 仅预留接口",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"模型路径或名称；YOLO 默认 {DEFAULT_YOLO_MODEL}，SAM3 默认 {DEFAULT_SAM3_MODEL}",
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"输入图片目录，默认为 {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出目录，默认为 {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--format",
        choices=["yolo", "labelme", "coco"],
        default=DEFAULT_FORMAT,
        help=f"导出格式，默认为 {DEFAULT_FORMAT}",
    )
    parser.add_argument(
        "--classes", default=DEFAULT_CLASSES,
        help="YOLO 模式下用于按 id 或名称过滤类别；SAM3 模式下作为文本提示，"
             f"逗号分隔（如 person,car），每个提示对应一个 class_id；默认为 '{DEFAULT_CLASSES}'",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"推理置信度阈值；YOLO 传给 model.predict()，SAM3 作为 mask 阈值；默认为 {DEFAULT_CONF}",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"YOLO 模式下推理 NMS IoU 阈值，传递给 model.predict()；默认为 {DEFAULT_IOU}",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help=f"YOLO 模式下推理图像尺寸；默认为 {DEFAULT_IMGSZ}",
    )
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE,
        help="推理设备，如 cpu / 0 / cuda，留空为自动",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
        help="结果级最小置信度过滤，覆盖在模型阈值之上",
    )
    parser.add_argument(
        "--copy-images", action="store_true", default=DEFAULT_COPY_IMAGES,
        help="导出时同时复制图片到输出目录的 images/ 子目录",
    )
    return parser


# =============================================================================
# 11. 主入口与执行
# =============================================================================


def _parse_sam3_prompts(arg: str) -> list[str]:
    """将 `--classes` 字符串解析为 SAM3 文本提示列表，空字符串返回空列表。"""
    return [s.strip() for s in arg.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，解析参数并执行推理与导出链路。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    ensure_local_supervision_import()

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        print(f"[错误] 图片文件夹不存在：{source}", file=sys.stderr)
        return 1

    images = list_images(source)
    if not images:
        print(f"[错误] 文件夹内没有图片：{source}", file=sys.stderr)
        return 1
    print(f"[信息] 共发现 {len(images)} 张图片")

    if args.model_type == "yolo":
        model_path = args.model or DEFAULT_YOLO_MODEL
        print(f"[信息] 正在加载 YOLO 模型：{model_path}")
        predict_kwargs: dict = {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "verbose": DEFAULT_VERBOSE,
        }
        if args.device:
            predict_kwargs["device"] = args.device
        labeler: AutoLabeler = YOLOLabeler(model_path, predict_kwargs)
        keep_ids = parse_class_ids(args.classes, dict(enumerate(labeler.classes)))
        if keep_ids:
            keep_names = [labeler.classes[i] for i in sorted(keep_ids)]
            print(f"[信息] 仅保留类别：{keep_names}")
    elif args.model_type == "sam3":
        prompts = _parse_sam3_prompts(args.classes)
        if not prompts:
            print(
                "[错误] SAM3 模式需要文本提示，请通过 --classes 指定，"
                "例如 --classes person,car",
                file=sys.stderr,
            )
            return 1
        model_id = args.model or DEFAULT_SAM3_MODEL
        print(f"[信息] 正在加载 SAM3 模型：{model_id}")
        print(f"[信息] 文本提示：{prompts}")
        device = args.device or None
        labeler = SAM3Labeler(
            model_id=model_id,
            classes=prompts,
            device=device,
            conf_threshold=args.conf,
        )
        keep_ids = None
    elif args.model_type == "detr":
        print("[错误] DETR 标注器尚未实现", file=sys.stderr)
        return 1
    else:
        print(f"[错误] 不支持的 model-type：{args.model_type}", file=sys.stderr)
        return 1

    dataset = build_detection_dataset(
        labeler=labeler,
        images=images,
        keep_class_ids=keep_ids,
        min_confidence=args.min_confidence,
    )

    print(f"[信息] 当前导出格式：{args.format}")
    export_detection_dataset(
        dataset=dataset, output=output, fmt=args.format,
        copy_images=args.copy_images,
    )
    print(f"[完成] 输出目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
