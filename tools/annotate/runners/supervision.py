"""
tools/annotate/runners/supervision.py

--model-type yolo|sam3|detr 模式：使用 supervision 数据集式导出为
YOLO / LabelMe / COCO 格式。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shutil
import sys
from tqdm import tqdm

from tools.annotate.backends import (
    AutoLabeler,
    DETRLabeler,
    SAM3Labeler,
    YOLOLabeler,
)
from tools.core import (
    IMAGE_EXTENSIONS as IMG_EXTS,
    detections_to_labelme_dict,
    list_images,
    save_labelme,
)
from tools.annotate.defaults import (
    DEFAULT_DETR_MODEL,
    DEFAULT_SAM3_MODEL,
    DEFAULT_YOLO_MODEL,
)


def parse_class_ids(
    arg: str | None, model_names: dict[int, str]
) -> set[int] | None:
    """将 ``--classes`` 字符串解析为类别 ID 集合。

    支持 ``0,2,3`` 与 ``person,car`` 两种形式；未知名称打印警告但不中断。
    返回 ``None`` 表示不过滤。
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
    detections, keep_class_ids: set[int] | None
):
    """按类别 ID 集合筛选 ``Detections``，未指定集合时原样返回。"""
    if keep_class_ids is None or len(detections) == 0:
        return detections
    mask = np.isin(detections.class_id, list(keep_class_ids))
    return detections[mask]


def filter_by_confidence(
    detections, min_confidence: float | None
):
    """按最小置信度筛选 ``Detections``，未指定时原样返回。"""
    if min_confidence is None or len(detections) == 0:
        return detections
    if detections.confidence is None:
        return detections
    mask = detections.confidence >= min_confidence
    return detections[mask]


def build_detection_dataset(
    labeler: AutoLabeler,
    images: list[Path],
    keep_class_ids: set[int] | None,
    min_confidence: float | None,
):
    """逐张图推理并累积为 ``sv.DetectionDataset``。

    无检测结果的图仍保留为空的 ``Detections``，确保图像集合完整。
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


def _unique_stem(path: Path, used: set[str]) -> str:
    """为 ``path`` 生成不重复的 stem，冲突时逐层追加父目录名。

    例如 ``datasets/15/capture_000000.png`` 与 ``datasets/16/capture_000000.png``
    会分别生成 ``15_capture_000000`` 与 ``16_capture_000000``。
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


def _export_labelme(dataset, output_dir: Path) -> None:
    """将数据集导出为 LabelMe JSON（图片与同名 .json 同目录）。

    当前环境 supervision 缺少 ``dataset.formats.labelme`` 模块（见 AGENTS.md
    §6），因此用 ``tools.core`` 的 ``detections_to_labelme_dict`` +
    ``save_labelme`` 自行写出矩形框，不依赖缺失的官方接口。不同子目录的
    同名图片通过 ``_unique_stem`` 去重。
    """
    import cv2

    used_stems: set[str] = set()
    for image_path, image, detections in dataset:
        src = Path(image_path)
        stem = _unique_stem(src, used_stems)
        used_stems.add(stem)
        dst = output_dir / f"{stem}{src.suffix}"
        shutil.copy2(str(src), str(dst))

        if image is not None and getattr(image, "shape", None) is not None:
            height, width = image.shape[:2]
        else:
            loaded = cv2.imread(str(src))
            height, width = (loaded.shape[:2] if loaded is not None else (0, 0))

        # 将框夹回图像范围内，避免 SAM 给出轻微越界的负坐标。
        if width > 0 and height > 0 and len(detections) > 0:
            xyxy = detections.xyxy.copy()
            xyxy[:, 0] = np.clip(xyxy[:, 0], 0, width)
            xyxy[:, 1] = np.clip(xyxy[:, 1], 0, height)
            xyxy[:, 2] = np.clip(xyxy[:, 2], 0, width)
            xyxy[:, 3] = np.clip(xyxy[:, 3], 0, height)
            detections.xyxy = xyxy

        data = detections_to_labelme_dict(
            detections,
            class_names=dataset.classes,
            image_path=dst.name,
            image_width=width,
            image_height=height,
        )
        save_labelme(data, dst.with_suffix(".json"))


def export_detection_dataset(
    dataset, output: Path, fmt: str, copy_images: bool
) -> None:
    """根据 ``fmt`` 选择 YOLO / LabelMe / COCO 中的一种导出方式。

    LabelMe 的标准布局是图片与同名 ``.json`` 同目录，因此 labelme 模式下
    会先把图片复制到输出目录并赋予唯一 basename，再写出 JSON。
    其他格式仍按 ``images/`` 与标注目录分开的常规布局写出。
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
        # LabelMe 图片与 JSON 同目录，且要求 basename 唯一。当前环境
        # supervision 未提供 labelme 导出模块（见 AGENTS.md §6），改用
        # tools.core 的桥接函数自行写出。
        if not copy_images:
            print(
                "[信息] LabelMe 格式需要图片与 JSON 同目录，已自动复制图片到输出目录。",
                file=sys.stderr,
            )
        _export_labelme(dataset, output)
    elif fmt == "coco":
        if copy_images:
            images_dir.mkdir(parents=True, exist_ok=True)
        dataset.as_coco(
            images_directory_path=str(images_dir) if copy_images else None,
            annotations_path=str(output / "annotations.json"),
        )
    else:
        raise ValueError(f"不支持的导出格式：{fmt}")


def _build_labeler(args) -> AutoLabeler:
    """根据 ``--model-type`` 构造 supervision 后端标注器。"""
    predict_kwargs = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    if args.model_type == "yolo":
        model_path = args.model or DEFAULT_YOLO_MODEL
        return YOLOLabeler(model_path, predict_kwargs)
    if args.model_type == "detr":
        model_path = args.model or DEFAULT_DETR_MODEL
        return DETRLabeler(model_path, predict_kwargs)
    # sam3
    prompts = [s.strip() for s in args.classes.split(",") if s.strip()]
    if not prompts:
        print("[错误] SAM3 模式需要 --classes 指定文本提示。", file=sys.stderr)
        raise SystemExit(1)
    return SAM3Labeler(
        model_path=args.model or DEFAULT_SAM3_MODEL,
        classes=prompts,
        device=args.device or None,
        conf_threshold=args.conf,
    )


def run_supervision(args) -> int:
    """执行 supervision 后端（yolo/sam3/detr）数据集式导出。"""
    source = Path(args.source)
    if not source.is_dir():
        print(f"[错误] 输入目录不存在：{source}", file=sys.stderr)
        return 1

    images = list_images(source, recursive=args.recursive)
    if not images:
        print(f"[错误] 文件夹内没有图片：{source}", file=sys.stderr)
        return 1

    labeler = _build_labeler(args)
    keep_class_ids = parse_class_ids(args.classes, labeler.classes)
    dataset = build_detection_dataset(
        labeler,
        images,
        keep_class_ids,
        args.min_confidence,
    )
    export_detection_dataset(
        dataset,
        Path(args.output),
        args.format,
        args.copy_images,
    )
    return 0
