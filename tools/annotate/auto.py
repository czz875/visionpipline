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
- 多检测器组合式：``--detectors-config <yaml>``，用 YAML 描述任意 N 路检测器
  （onnx / sam / yolo / detr，可混搭、同类型可多路，如两个 YOLO / 两个 ONNX），
  逐路归一成「框 + 逐框标签」后复用上面的打码合并链路，输出 LabelMe。

打码（马赛克/纯黑）统一抽到 ``tools/annotate/ops.apply_blackout``，本脚本只编排。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    DETRLabeler,
    SAM3Labeler,
    YOLOLabeler,
)
from tools.annotate.backends.onnx import OnnxDetector
from tools.annotate.backends.sam import SAMTextDetector
from tools.annotate.ops import (  # noqa: E402
    DEFAULT_MOSAIC_BLOCK,
    apply_blackout,
    classify_box_by_ratio,
    clip_box,
    concat_boxes,
    extract_existing_label_boxes,
    rewrite_labelme_dict,
)


# =============================================================================
# 3. 本地导入辅助
# =============================================================================


def ensure_local_supervision_import() -> None:
    """将仓库 `src/` 插入 `sys.path` 前部，优先使用本地开发版 supervision。

    在脚本入口处调用一次即可，避免安装版覆盖仓库内的 `as_labelme`、
    `as_coco` 等较新接口。
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
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
    """单路检测/保留来源的框与标签，已按面积占比切分为保留/删除两部分。

    ``kept_labels`` 与 ``kept`` 逐框对齐，支持同一路检测器输出多个类别
    （如一个 YOLO 模型同时给出 face / phone / hand）。删除的小框最终会被
    打码删掉，无需保留标签。
    """
    kept: np.ndarray
    removed: np.ndarray
    kept_labels: list[str]


def _build_source(
    boxes: np.ndarray,
    label: str | list[str],
    min_ratio: float,
    image_shape: tuple[int, int, int],
) -> _BoxSource:
    """把原始框清洗并按面积占比切成保留/删除两部分，保留框携带逐框标签。

    ``label`` 传单个字符串时对所有框统一标注（ONNX / SAM 单类别路）；
    传列表时与 ``boxes`` 逐框对齐（YOLO 多类别路）。清洗（裁剪 + 丢弃退化框）
    与按面积占比切分在同一次遍历完成，确保保留框与标签始终对齐。
    """
    labels = [label] * len(boxes) if isinstance(label, str) else list(label)
    kept_boxes: list[np.ndarray] = []
    kept_labels: list[str] = []
    removed_boxes: list[np.ndarray] = []
    for box, lab in zip(boxes, labels):
        clipped = clip_box(box, image_shape)
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        if classify_box_by_ratio(clipped, image_shape, min_ratio):
            kept_boxes.append(clipped)
            kept_labels.append(lab)
        else:
            removed_boxes.append(clipped)
    kept = np.array(kept_boxes, dtype=np.float32) if kept_boxes else np.empty((0, 4), dtype=np.float32)
    removed = np.array(removed_boxes, dtype=np.float32) if removed_boxes else np.empty((0, 4), dtype=np.float32)
    return _BoxSource(kept=kept, removed=removed, kept_labels=kept_labels)


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


def _run_multi_source_annotation(
    image: np.ndarray,
    sources: list[_BoxSource],
    *,
    use_mosaic: bool,
    mosaic_block: int,
    dry_run: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """对多路来源执行：合并保留大框 -> 对每路小框打码（重叠保护）-> 汇总框与标签。

    所有保留大框共同构成「重叠保护区」，任一来源的小框只要与任一保留大框重叠
    都不打码。适用于任意 N 路检测器组合。返回（可能打码后的图像, 最终框,
    最终标签, 打码区域数）。
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
    final_labels = [label for s in sources for label in s.kept_labels]
    return image, final_boxes, final_labels, blackout_total


def _write_annotation_result(
    image: np.ndarray,
    image_path: Path,
    image_name: str,
    boxes: np.ndarray,
    labels: list[str],
    output: Path | None,
) -> None:
    """把单张标注结果（打码后图像 + 框/标签）写入输出目录的图片与同名 JSON。

    ``output`` 为 ``None`` 时不写盘（调用方负责处理覆盖模式等的原地写盘）。
    """
    if output is None:
        return
    out_img = output / image_path.name
    out_json = out_img.with_suffix(".json")
    cv2.imwrite(str(out_img), image)
    data = rewrite_labelme_dict(out_json, boxes, labels, image.shape, image_name)
    save_labelme(data, out_json)


def _iter_annotated_images(
    image_paths: list[Path],
    build_sources: Callable[[np.ndarray, Path], list["_BoxSource"]],
    *,
    use_mosaic: bool,
    mosaic_block: int,
    dry_run: bool,
    output: Path | None,
):
    """通用标注循环：推理建源 -> 多源打码合并 -> 写盘，逐张 yield 统计要素。

    ``build_sources(image, image_path)`` 由调用方提供，负责把各检测器输出归一成
    ``_BoxSource`` 列表（ONNX 一路、SAM 第二路、多检测器组合的差异都封在这里）。
    循环主体因此在所有「标注模式」间复用，避免重复遍历/写盘逻辑。
    每个元素 yield ``(image_path, sources, boxes, labels, blackout_count)``。
    """
    for image_path in tqdm(image_paths, desc="标注中", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片：{image_path}")
        sources = build_sources(image, image_path)
        image, boxes, labels, n = _run_multi_source_annotation(
            image,
            sources,
            use_mosaic=use_mosaic,
            mosaic_block=mosaic_block,
            dry_run=dry_run,
        )
        if not dry_run:
            _write_annotation_result(
                image, image_path, image_path.name, boxes, labels, output
            )
        yield image_path, sources, boxes, labels, n


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
            image, boxes, labels, n = _run_multi_source_annotation(
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

    def _onnx_build_sources(image: np.ndarray, image_path: Path) -> list[_BoxSource]:
        """ONNX 一路（可选 SAM 第二路）归一成 _BoxSource 列表。"""
        onnx_boxes, _ = onnx_detector.predict(image)
        srcs = [
            _build_source(onnx_boxes, args.onnx_label, args.onnx_min_ratio, image.shape)
        ]
        if sam_detector is not None:
            sam_boxes, _ = sam_detector.predict(image_path)
            srcs.append(_build_source(sam_boxes, args.sam_label, args.sam_min_ratio, image.shape))
        return srcs

    total_onnx = total_sam = total_removed = total_mosaic = 0
    for _path, sources, _boxes, _labels, n in _iter_annotated_images(
        images,
        _onnx_build_sources,
        use_mosaic=use_mosaic,
        mosaic_block=args.mosaic_block,
        dry_run=args.dry_run,
        output=output,
    ):
        total_onnx += len(sources[0].kept)
        total_sam += (len(sources[1].kept) if len(sources) > 1 else 0)
        total_removed += sum(len(s.removed) for s in sources)
        total_mosaic += n

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
# 10.5 多检测器组合（--detectors-config，任意 N 路混搭）
# =============================================================================


@dataclass
class _Detector:
    """一路检测器的运行期封装：名称 + 最小面积占比 + 归一化的检测函数。

    ``detect(image, image_path) -> (xyxy 框, 逐框标签)`` 屏蔽各后端输入差异
    （ONNX 吃 ndarray、SAM/YOLO 吃 path），供多检测器链路统一调用。
    """
    name: str
    min_ratio: float
    detect: Callable[[np.ndarray, Path], tuple[np.ndarray, list[str]]]


def _coerce_score_indices(value) -> tuple[int, ...]:
    """把 YAML 里的 ``score_indices``（列表或 "4,15" 字符串）统一成 int 元组。"""
    if isinstance(value, str):
        return _parse_score_indices(value)
    return tuple(int(v) for v in value)


def _sv_detect(
    labeler: AutoLabeler,
    image_path: Path,
    keep_names: set[str],
    rename: str | None,
) -> tuple[np.ndarray, list[str]]:
    """跑一次 supervision 后端（YOLO / DETR）推理，抽出 xyxy 与逐框类别名。

    ``keep_names`` 非空时只保留其中的类别；``rename`` 非空时把保留下来的框
    统一改名为该标签（便于把多类别归并成一个打码/保留类别）。
    """
    detections = labeler.predict(image_path)
    if len(detections) == 0 or detections.class_id is None:
        return np.empty((0, 4), dtype=np.float32), []
    xyxy = detections.xyxy.astype(np.float32)
    names = [labeler.classes[int(cid)] for cid in detections.class_id]
    boxes_out: list[np.ndarray] = []
    labels_out: list[str] = []
    for box, name in zip(xyxy, names):
        if keep_names and name not in keep_names:
            continue
        boxes_out.append(box)
        labels_out.append(rename or name)
    if not boxes_out:
        return np.empty((0, 4), dtype=np.float32), []
    return np.array(boxes_out, dtype=np.float32), labels_out


def _build_detectors(
    detector_cfgs: list[dict[str, object]],
    *,
    default_min_ratio: float,
    default_device: str | None,
) -> list[_Detector]:
    """按 YAML 配置逐项构造检测器（模型只加载一次），返回统一封装列表。

    支持 ``onnx`` / ``sam`` / ``yolo`` / ``detr`` 混搭且同类型可多路。
    每项通用键：``type`` / ``model`` / ``min_ratio`` / ``conf`` / ``name``。
    """
    detectors: list[_Detector] = []
    for idx, cfg in enumerate(detector_cfgs):
        dtype = str(cfg.get("type", "")).lower()
        min_ratio = float(cfg.get("min_ratio", default_min_ratio))
        name = str(cfg.get("name") or f"{dtype}#{idx}")
        device = cfg.get("device") or default_device

        if dtype == "onnx":
            backend = OnnxDetector(
                Path(cfg["model"]),
                str(cfg.get("label", "object")),
                float(cfg.get("conf", DEFAULT_ONNX_CONF)),
                normalize=bool(cfg.get("normalize", DEFAULT_ONNX_NORMALIZE)),
                transpose=bool(cfg.get("transpose", DEFAULT_ONNX_TRANSPOSE)),
                score_indices=_coerce_score_indices(
                    cfg.get("score_indices", DEFAULT_ONNX_SCORE_INDICES)
                ),
                iou_threshold=float(cfg.get("iou", DEFAULT_IOU)),
            )

            def detect(image, _path, backend=backend):
                return backend.predict(image)

        elif dtype == "sam":
            backend = SAMTextDetector(
                model_path=Path(cfg["model"]),
                label=str(cfg.get("label", cfg.get("prompt", "object"))),
                conf=float(cfg.get("conf", DEFAULT_SAM_CONF)),
                prompt=str(cfg["prompt"]),
                device=device or None,
            )

            def detect(_image, image_path, backend=backend):
                return backend.predict(image_path)

        elif dtype in ("yolo", "detr"):
            predict_kwargs: dict[str, object] = {
                "conf": float(cfg.get("conf", DEFAULT_CONF)),
                "iou": float(cfg.get("iou", DEFAULT_IOU)),
                "imgsz": int(cfg.get("imgsz", DEFAULT_IMGSZ)),
                "verbose": DEFAULT_VERBOSE,
            }
            if device:
                predict_kwargs["device"] = device
            backend: AutoLabeler = (
                YOLOLabeler(str(cfg["model"]), predict_kwargs)
                if dtype == "yolo"
                else DETRLabeler(str(cfg["model"]), predict_kwargs)
            )
            keep_names = {str(c) for c in cfg.get("classes", [])}
            rename = cfg.get("label")

            def detect(_image, image_path, backend=backend, keep=keep_names, rename=rename):
                return _sv_detect(backend, image_path, keep, rename)

        else:
            raise ValueError(f"未知检测器类型：{dtype}")

        detectors.append(_Detector(name=name, min_ratio=min_ratio, detect=detect))
    return detectors


def run_detectors(args: argparse.Namespace) -> int:
    """执行多检测器组合标注（``--detectors-config <yaml>``）。

    从 YAML 读取任意 N 路检测器与全局项，逐路归一成「框 + 逐框标签」后复用
    ``_run_multi_source_annotation`` 的打码合并链路，输出 LabelMe。全局项
    （source/output/recursive/打码策略/最小面积占比等）可写在 YAML 里，
    未写则回退到 CLI 默认。是否写盘沿用 ``--apply``（默认预览）。
    """
    import yaml

    cfg_path = Path(args.detectors_config)
    if not cfg_path.is_file():
        print(f"[错误] 检测器配置不存在：{cfg_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    detector_cfgs = cfg.get("detectors") or []
    if not detector_cfgs:
        print("[错误] 配置里 detectors 为空。", file=sys.stderr)
        return 1

    # 全局项：优先取 YAML，其次回退到 CLI 默认。
    source = Path(cfg.get("source", args.source))
    output = Path(cfg.get("output", args.output))
    recursive = bool(cfg.get("recursive", args.recursive))
    use_mosaic = not bool(cfg.get("blackout", False))  # 默认马赛克，blackout: true 用纯黑
    mosaic_block = int(cfg.get("mosaic_block", args.mosaic_block))
    default_min_ratio = float(cfg.get("min_ratio", DEFAULT_ONNX_MIN_RATIO))
    device = cfg.get("device", args.device) or None

    if not source.is_dir():
        print(f"[错误] 输入目录不存在：{source}", file=sys.stderr)
        return 1
    images = list_images(source, recursive=recursive)
    if not images:
        print(f"[错误] 文件夹内没有图片：{source}", file=sys.stderr)
        return 1
    print(f"[信息] 共发现 {len(images)} 张图片，检测器 {len(detector_cfgs)} 路")

    detectors = _build_detectors(
        detector_cfgs, default_min_ratio=default_min_ratio, default_device=device
    )
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    per_kept: dict[str, int] = {d.name: 0 for d in detectors}
    total_removed = total_mosaic = 0

    def _det_build_sources(image: np.ndarray, image_path: Path) -> list[_BoxSource]:
        """各路检测器归一化为 _BoxSource，并累计每路保留框数。"""
        srcs: list[_BoxSource] = []
        for detector in detectors:
            boxes, labels = detector.detect(image, image_path)
            source_boxes = _build_source(boxes, labels, detector.min_ratio, image.shape)
            per_kept[detector.name] += len(source_boxes.kept)
            srcs.append(source_boxes)
        return srcs

    for _path, sources, _boxes, _labels, n in _iter_annotated_images(
        images,
        _det_build_sources,
        use_mosaic=use_mosaic,
        mosaic_block=mosaic_block,
        dry_run=args.dry_run,
        output=output,
    ):
        total_removed += sum(len(s.removed) for s in sources)
        total_mosaic += n

    mode = "预览" if args.dry_run else "完成"
    print(f"[{mode}] 图片数：{len(images)}")
    for name, count in per_kept.items():
        print(f"[{mode}] 保留 {name}：{count}")
    print(f"[{mode}] 小框(已删)：{total_removed}")
    print(
        f"[{mode}] 实际打码区域：{total_mosaic}"
        f"（重叠被保护跳过：{total_removed - total_mosaic}）"
    )
    if not args.dry_run:
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
        "--detectors-config",
        default=None,
        help="多检测器组合的 YAML 配置路径（建议放 src/，不入 git）。指定后走「任意 N 路"
             "混搭」打标链路（onnx/sam/yolo 可混搭、同类型可多路），忽略 --model-type。",
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

    # 多检测器组合（任意 N 路混搭）优先：走 YAML 配置驱动的打标链路。
    if args.detectors_config:
        return run_detectors(args)

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

    # YOLO / DETR 共用的推理参数（sam3 走文本 prompt，不在此列）。
    predict_kwargs: dict[str, object] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": DEFAULT_VERBOSE,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    if args.model_type == "yolo":
        model_path = args.model or DEFAULT_YOLO_MODEL
        print(f"[信息] 正在加载 YOLO 模型：{model_path}")
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
        model_path = args.model or DEFAULT_DETR_MODEL
        print(f"[信息] 正在加载 RT-DETR 模型：{model_path}")
        labeler = DETRLabeler(model_path, predict_kwargs)
        keep_ids = parse_class_ids(args.classes, dict(enumerate(labeler.classes)))
        if keep_ids:
            keep_names = [labeler.classes[i] for i in sorted(keep_ids)]
            print(f"[信息] 仅保留类别：{keep_names}")
    else:
        # onnx 已在函数开头提前分发，这里只处理未知类型。
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
