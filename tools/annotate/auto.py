"""
tools/annotate/auto.py
统一标注入口，支持两类后端：

- supervision 数据集式：``--model-type yolo|sam3|detr``，模型推理 ->
  sv.Detections -> 类别/置信度过滤 -> 累计为 sv.DetectionDataset ->
  导出 YOLO / LabelMe / COCO；
- ONNX 式（合并原 auto_onnx_sam.py / reannotate_onnx.py）：``--model-type onnx``，
  用 ONNX 检测器标一路（可选 SAM 文本 prompt 标第二路），按面积占比切分保留/
  删除框，小框打码（默认马赛克，重叠保护）后从标注中删除，输出 LabelMe；
  加 ``--reannotate`` 则改为覆盖指定类别并保留其它现有类别（原地覆盖）。

打码（马赛克/纯黑）统一抽到 ``tools/annotate/ops.apply_blackout``，本脚本只编排。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
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

# ----- ONNX / SAM 两路打标（--model-type onnx，覆盖原 auto_onnx_sam.py）-----
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

# ----- ONNX 覆盖模式（--model-type onnx --reannotate，覆盖原 reannotate_onnx.py）-----
DEFAULT_INPUT_ROOT = r"datasets\behavior"
DEFAULT_START_BATCH = 23
DEFAULT_END_BATCH = 37
DEFAULT_KEEP_MIN_RATIO = None


# =============================================================================
# 2. 常量与后端导入
# =============================================================================

from tools.core import (  # noqa: E402
    IMAGE_EXTENSIONS as IMG_EXTS,
    detections_to_labelme_dict,
    find_json_for_image,
    list_images,
    load_labelme,
    save_labelme,
)

# 检测器后端按模型类型分离，统一从 tools.annotate.backends 引入。
from tools.annotate.backends import (  # noqa: E402
    AutoLabeler,
    DatasetLike,
    DetectionsLike,
    SAM3Labeler,
    YOLOLabeler,
)
from tools.annotate.backends.onnx import OnnxDetector
from tools.annotate.backends.sam import SAMTextDetector
from tools.annotate.ops import (  # noqa: E402
    DEFAULT_MOSAIC_BLOCK,
    apply_blackout,
    concat_boxes,
    extract_existing_label_boxes,
    rewrite_labelme_dict,
    sanitize_boxes,
    split_boxes_by_ratio,
)


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
# 4. 标注器后端
# =============================================================================

# 各模型后端（YOLO / SAM3 / DETR）已拆分到 tools.annotate.backends，按模型类型
# 分文件维护；本脚本只做「打标编排」：类别/置信度过滤、累积数据集、导出。


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


def _export_labelme(dataset: DatasetLike, output_dir: Path) -> None:
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


# =============================================================================
# 10. ONNX 打标 / 覆盖流水线（合并原 auto_onnx_sam.py / reannotate_onnx.py）
# =============================================================================


def _parse_score_indices(text: str) -> tuple[int, ...]:
    """解析 ``"4,15"`` 形式的分通道下标为 int 元组。"""
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


@dataclass
class _BoxSource:
    """单路检测/保留来源的框与标签，已按面积占比切分为保留/删除两部分。"""
    label: str
    kept: np.ndarray
    removed: np.ndarray


def _build_source(
    boxes: np.ndarray,
    label: str,
    min_ratio: float,
    image_shape: tuple[int, int, int],
) -> _BoxSource:
    """把原始框清洗并按面积占比切成保留/删除两部分。"""
    boxes = sanitize_boxes(boxes, image_shape)
    kept, removed = split_boxes_by_ratio(boxes, image_shape, min_ratio)
    return _BoxSource(label=label, kept=kept, removed=removed)


def _resolve_keep_labels(
    json_path: Path,
    onnx_label: str,
    explicit: list[str] | None,
) -> list[str]:
    """解析覆盖模式下需要保留的现有类别。

    ``explicit`` 为 ``None`` 时自动保留 JSON 中除 ``onnx_label`` 外的所有矩形类别；
    否则使用显式指定的类别列表。
    """
    if explicit is not None:
        return explicit
    if not json_path.exists():
        return []
    data = load_labelme(json_path)
    labels: list[str] = []
    seen: set[str] = set()
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape.get("label", "")
        if label == onnx_label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _iter_batch_image_files(root: Path, start_batch: int, end_batch: int) -> list[Path]:
    """收集指定 batch 范围内存在的图片（按 ``0000``~``9999`` 子目录组织）。"""
    image_paths: list[Path] = []
    for batch_id in range(start_batch, end_batch + 1):
        batch_dir = root / f"{batch_id:04d}"
        if batch_dir.is_dir():
            image_paths.extend(list_images(batch_dir, recursive=True))
    return image_paths


def _run_onnx_annotation(
    image: np.ndarray,
    sources: list[_BoxSource],
    *,
    use_mosaic: bool,
    mosaic_block: int,
    dry_run: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """对多路来源执行：合并保留大框 -> 对每路小框打码（重叠保护）-> 汇总框与标签。

    所有保留大框共同构成「重叠保护区」，任一来源的小框只要与任一保留大框重叠
    都不打码。返回（可能打码后的图像, 最终框, 最终标签, 打码区域数）。
    """
    all_kept = concat_boxes([s.kept for s in sources])
    blackout_total = 0
    for source in sources:
        image, n = apply_blackout(
            image,
            source.removed,
            all_kept,
            use_mosaic=use_mosaic,
            mosaic_block=mosaic_block,
            apply=not dry_run,
        )
        blackout_total += n
    final_boxes = concat_boxes([s.kept for s in sources])
    final_labels = [label for s in sources for label in [s.label] * len(s.kept)]
    return image, final_boxes, final_labels, blackout_total


def run_onnx(args: argparse.Namespace) -> int:
    """执行 ONNX 打标 / 覆盖（``--model-type onnx``）。

    - 未加 ``--reannotate``：ONNX 一路（可选 SAM 第二路）标新框，写入输出目录；
    - 加 ``--reannotate``：ONNX 覆盖指定类别并保留其它类别，原地覆盖。
    """
    source = Path(args.source) if args.source else None
    output = Path(args.output) if args.output else None

    onnx_detector = OnnxDetector(
        args.onnx_model,
        args.onnx_label,
        args.onnx_conf,
        normalize=args.onnx_normalize,
        transpose=args.onnx_transpose,
        score_indices=_parse_score_indices(args.onnx_score_indices),
    )
    sam_detector = None
    if getattr(args, "sam_model", None):
        sam_detector = SAMTextDetector(
            model_path=args.sam_model,
            label=args.sam_label,
            conf=args.sam_conf,
            prompt=args.sam_prompt,
            device=args.device or None,
        )

    # 打码策略：标注模式默认马赛克；覆盖模式默认纯黑（与原脚本一致）。
    use_mosaic = args.mosaic if args.reannotate else (not args.blackout)

    # ---- 覆盖模式：仅在原地覆盖，不走输出目录 ----
    if args.reannotate:
        image_paths = _iter_batch_image_files(args.input_root, args.start_batch, args.end_batch)
        if not image_paths:
            print("[错误] 指定 batch 范围内没有可处理图片。", file=sys.stderr)
            return 1
        keep_labels = (
            [s.strip() for s in args.keep_labels.split(",") if s.strip()]
            if args.keep_labels
            else None
        )
        min_keep_ratio = (
            args.keep_min_ratio if args.keep_min_ratio is not None else args.onnx_min_ratio
        )

        total_onnx = total_keep = total_blackout = total_json = 0
        for image_path in tqdm(image_paths, desc="覆盖标注", unit="img"):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图片：{image_path}")
            json_path = find_json_for_image(image_path)
            onnx_boxes, _ = onnx_detector.predict(image)
            sources = [
                _build_source(onnx_boxes, args.onnx_label, args.onnx_min_ratio, image.shape)
            ]
            for label in _resolve_keep_labels(json_path, args.onnx_label, keep_labels):
                keep_boxes = extract_existing_label_boxes(json_path, label)
                sources.append(_build_source(keep_boxes, label, min_keep_ratio, image.shape))
            image, boxes, labels, n = _run_onnx_annotation(
                image,
                sources,
                use_mosaic=use_mosaic,
                mosaic_block=args.mosaic_block,
                dry_run=args.dry_run,
            )
            total_onnx += len(sources[0].kept)
            total_keep += sum(len(s.kept) for s in sources[1:])
            total_blackout += n
            total_json += 1
            if not args.dry_run:
                data = rewrite_labelme_dict(json_path, boxes, labels, image.shape, image_path.name)
                cv2.imwrite(str(image_path), image)
                save_labelme(data, json_path)

        mode = "预览" if args.dry_run else "完成"
        print(f"[{mode}] 图片数：{len(image_paths)}")
        print(f"[{mode}] 覆盖 {args.onnx_label}：{total_onnx}")
        print(f"[{mode}] 保留其它类别框：{total_keep}")
        print(f"[{mode}] 小框打码并删除：{total_blackout}")
        print(f"[{mode}] 覆盖 JSON：{total_json}")
        return 0

    # ---- 标注模式：写入输出目录 ----
    if source is None or not source.is_dir():
        print("[错误] 请通过 --source 指定输入图片目录。", file=sys.stderr)
        return 1
    images = list_images(source, recursive=args.recursive)
    if not images:
        print(f"[错误] 文件夹内没有图片：{source}", file=sys.stderr)
        return 1
    print(f"[信息] 共发现 {len(images)} 张图片")
    if not args.dry_run and output is not None:
        output.mkdir(parents=True, exist_ok=True)

    total_onnx = total_sam = total_removed = total_mosaic = 0
    for image_path in tqdm(images, desc="标注中", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片：{image_path}")
        onnx_boxes, _ = onnx_detector.predict(image)
        sources = [
            _build_source(onnx_boxes, args.onnx_label, args.onnx_min_ratio, image.shape)
        ]
        if sam_detector is not None:
            sam_boxes, _ = sam_detector.predict(image_path)
            sources.append(_build_source(sam_boxes, args.sam_label, args.sam_min_ratio, image.shape))
        image, boxes, labels, n = _run_onnx_annotation(
            image,
            sources,
            use_mosaic=use_mosaic,
            mosaic_block=args.mosaic_block,
            dry_run=args.dry_run,
        )
        total_onnx += len(sources[0].kept)
        total_sam += (len(sources[1].kept) if len(sources) > 1 else 0)
        total_removed += sum(len(s.removed) for s in sources)
        total_mosaic += n
        if not args.dry_run and output is not None:
            out_img = output / image_path.name
            out_json = out_img.with_suffix(".json")
            cv2.imwrite(str(out_img), image)
            data = rewrite_labelme_dict(out_json, boxes, labels, image.shape, image_path.name)
            save_labelme(data, out_json)

    mode = "预览" if args.dry_run else "完成"
    print(f"[{mode}] 图片数：{len(images)}")
    print(f"[{mode}] 保留 {args.onnx_label}：{total_onnx}")
    if sam_detector is not None:
        print(f"[{mode}] 保留 {args.sam_label}：{total_sam}")
    print(f"[{mode}] 小框(已删)：{total_removed}")
    print(
        f"[{mode}] 实际打码区域：{total_mosaic}"
        f"（重叠被保护跳过：{total_removed - total_mosaic}）"
    )
    if not args.dry_run and output is not None:
        print(f"[{mode}] 输出目录：{output}")
    else:
        print("[提示] 当前为预览模式，未写盘；确认无误请加 --apply。")
    return 0


# =============================================================================
# 11. 命令行参数
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """构造脚本的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="统一标注入口：supervision 后端（yolo/sam3/detr）或 "
                    "ONNX 后端（onnx 两路打标 / 覆盖，含打码）",
    )
    parser.add_argument(
        "--model-type",
        choices=["yolo", "sam3", "detr", "onnx"],
        default=DEFAULT_MODEL_TYPE,
        help="标注器类型；yolo/sam3/detr 走 supervision 数据集式，onnx 走 ONNX/SAM 两路打标或覆盖",
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
        help=f"导出格式（仅 supervision 后端），默认为 {DEFAULT_FORMAT}",
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

    # ----- ONNX 一路 -----
    parser.add_argument(
        "--onnx-model", type=Path, default=Path(DEFAULT_ONNX_MODEL),
        help=f"ONNX 模型路径（默认：{DEFAULT_ONNX_MODEL}）。",
    )
    parser.add_argument(
        "--onnx-label", default=DEFAULT_ONNX_LABEL,
        help=f"ONNX 检测器输出的类别名（默认：{DEFAULT_ONNX_LABEL}）。",
    )
    parser.add_argument(
        "--onnx-conf", type=float, default=DEFAULT_ONNX_CONF,
        help=f"ONNX 置信度阈值（默认：{DEFAULT_ONNX_CONF}）。",
    )
    parser.add_argument(
        "--onnx-min-ratio", type=float, default=DEFAULT_ONNX_MIN_RATIO,
        help=f"ONNX 保留框的最小面积占比；小于此值的框打码后删除（默认：{DEFAULT_ONNX_MIN_RATIO}）。",
    )
    parser.add_argument(
        "--onnx-transpose", action="store_true", default=DEFAULT_ONNX_TRANSPOSE,
        help="ONNX 输出布局为 (C, N)，需转置为 (N, C) 再解码（默认关闭）。",
    )
    parser.add_argument(
        "--onnx-score-indices", default=DEFAULT_ONNX_SCORE_INDICES,
        help=f"参与相乘得到分数的输出通道下标，逗号分隔（默认：{DEFAULT_ONNX_SCORE_INDICES}）。",
    )
    parser.add_argument(
        "--onnx-normalize", dest="onnx_normalize", action="store_true",
        default=DEFAULT_ONNX_NORMALIZE,
        help="预处理时对像素除以 255 归一化（默认开启）。",
    )
    parser.add_argument(
        "--no-onnx-normalize", dest="onnx_normalize", action="store_false",
        help="关闭像素归一化。",
    )

    # ----- SAM 第二路（可选）-----
    parser.add_argument(
        "--sam-model", type=Path, default=None,
        help=f"SAM 文本 prompt 模型路径；留空则只用 ONNX 一路（默认：{DEFAULT_SAM_MODEL}）。",
    )
    parser.add_argument(
        "--sam-prompt", default=DEFAULT_SAM_PROMPT,
        help=f"SAM 文本提示（默认：{DEFAULT_SAM_PROMPT}）。",
    )
    parser.add_argument(
        "--sam-label", default=DEFAULT_SAM_LABEL,
        help=f"SAM 检测器输出的类别名（默认：{DEFAULT_SAM_LABEL}）。",
    )
    parser.add_argument(
        "--sam-conf", type=float, default=DEFAULT_SAM_CONF,
        help=f"SAM 置信度阈值（默认：{DEFAULT_SAM_CONF}）。",
    )
    parser.add_argument(
        "--sam-min-ratio", type=float, default=DEFAULT_SAM_MIN_RATIO,
        help=f"SAM 保留框的最小面积占比（默认：{DEFAULT_SAM_MIN_RATIO}）。",
    )

    # ----- ONNX 覆盖模式（--reannotate）-----
    parser.add_argument(
        "--reannotate", action="store_true", default=False,
        help="覆盖模式：ONNX 覆盖 --onnx-label 旧框，保留其它现有类别，原地覆盖图片与 JSON。",
    )
    parser.add_argument(
        "--input-root", type=Path, default=Path(DEFAULT_INPUT_ROOT),
        help=f"覆盖模式按 batch 子目录组织的数据根目录（默认：{DEFAULT_INPUT_ROOT}）。",
    )
    parser.add_argument(
        "--start-batch", type=int, default=DEFAULT_START_BATCH,
        help=f"覆盖模式起始 batch 编号（默认：{DEFAULT_START_BATCH}）。",
    )
    parser.add_argument(
        "--end-batch", type=int, default=DEFAULT_END_BATCH,
        help=f"覆盖模式结束 batch 编号（默认：{DEFAULT_END_BATCH}）。",
    )
    parser.add_argument(
        "--keep-labels",
        default=None,
        help="覆盖模式保留的现有类别，逗号分隔（如 'hand,phone'）。留空则自动保留除 "
             "--onnx-label 外的所有现有类别。",
    )
    parser.add_argument(
        "--keep-min-ratio", type=float, default=DEFAULT_KEEP_MIN_RATIO,
        help="覆盖模式保留类别框的最小面积占比；默认与 --onnx-min-ratio 相同。",
    )

    # ----- 打码策略 -----
    parser.add_argument(
        "--mosaic", dest="mosaic", action="store_true", default=False,
        help="小框用马赛克打码（标注模式默认即马赛克，覆盖模式需显式开启）。",
    )
    parser.add_argument(
        "--blackout", dest="blackout", action="store_true", default=False,
        help="小框用纯黑打码（覆盖模式默认纯黑，标注模式需显式开启）。",
    )
    parser.add_argument(
        "--mosaic-block", type=int, default=DEFAULT_MOSAIC_BLOCK,
        help=f"马赛克块大小（像素），越大越糊（默认：{DEFAULT_MOSAIC_BLOCK}）。",
    )

    parser.add_argument(
        "--recursive", action="store_true", default=True,
        help="递归扫描子目录（默认开启）。",
    )
    parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="关闭递归，只扫描顶层目录。",
    )
    parser.add_argument(
        "--apply", dest="dry_run", action="store_false",
        help="真正写盘（图片打码 + 写 JSON）；默认仅统计预览（ONNX 模式）。",
    )
    parser.set_defaults(dry_run=False)
    return parser


# =============================================================================
# 12. 主入口与执行
# =============================================================================


def _parse_sam3_prompts(arg: str) -> list[str]:
    """将 `--classes` 字符串解析为 SAM3 文本提示列表，空字符串返回空列表。"""
    return [s.strip() for s in arg.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，解析参数并执行推理与导出链路。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ONNX 后端走独立流水线（含打码、原地覆盖等），不依赖 supervision。
    if args.model_type == "onnx":
        return run_onnx(args)

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
            model_path=model_id,
            classes=prompts,
            device=device,
            conf_threshold=args.conf,
        )
        keep_ids = None
    elif args.model_type == "detr":
        print("[错误] DETR 标注器尚未实现", file=sys.stderr)
        return 1
    elif args.model_type == "onnx":
        return run_onnx(args)
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
