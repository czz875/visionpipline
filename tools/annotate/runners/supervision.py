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
    read_image,
    save_labelme,
)
from tools.annotate.defaults import (
    DEFAULT_DETR_MODEL,
    DEFAULT_SAM3_MODEL,
    DEFAULT_YOLO_MODEL,
)


def parse_class_ids(
    arg: str | None, model_names: dict[int, str] | list[str]
) -> set[int] | None:
    """将 ``--classes`` 字符串解析为类别 ID 集合。

    支持 ``0,2,3`` 与 ``person,car`` 两种形式。

    Args:
        arg: ``--classes`` 参数原文。
        model_names: 类别映射。既接受 ``{id: name}``（模型自带 ``names``），
            也接受 ``list[str]``（各后端 ``AutoLabeler.classes`` 的形态）。

    Returns:
        类别 ID 集合；``None`` 表示不过滤。

    Raises:
        ValueError: 存在无法识别的类别名。静默忽略会让 ``--classes fac``
            退化成「不过滤」，把不该标注的类别一并写进结果，而用户以为
            过滤已生效 —— 与 ultralytics 原生语义相反。
    """
    if not arg:
        return None
    items = [s.strip() for s in arg.split(",") if s.strip()]
    if not items:
        return None
    if isinstance(model_names, dict):
        pairs = list(model_names.items())
    else:
        pairs = list(enumerate(model_names))
    ids: set[int] = set()
    name_to_id = {str(name).lower(): int(idx) for idx, name in pairs}
    unknown: list[str] = []
    for it in items:
        if it.isdigit():
            ids.add(int(it))
            continue
        mapped = name_to_id.get(it.lower())
        if mapped is not None:
            ids.add(mapped)
        else:
            unknown.append(it)
    if unknown:
        available = ", ".join(str(name) for _, name in pairs)
        raise ValueError(
            f"未识别的类别：{'、'.join(unknown)}。可用类别：{available}"
        )
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
    单张图推理失败（损坏、截断、无法解码）时跳过该图并计数，不中断整批：
    ultralytics 对坏图会返回空结果列表，直接取 ``[0]`` 会抛 IndexError，
    让已推理成功的图片全部白跑。
    """
    import supervision as sv

    annotations: dict[str, sv.Detections] = {}
    image_paths: list[str] = []
    failed: list[tuple[Path, str]] = []
    for img_path in tqdm(images, desc="推理中"):
        try:
            detections = labeler.predict(img_path)
        except Exception as e:  # 单图失败不应终止整批
            failed.append((img_path, f"{type(e).__name__}: {e}"))
            continue
        detections = filter_by_class_ids(detections, keep_class_ids)
        detections = filter_by_confidence(detections, min_confidence)
        key = str(img_path.resolve())
        image_paths.append(key)
        # 即便为空也写入，保证导出时图像集合完整。
        annotations[key] = detections if len(detections) else sv.Detections.empty()

    if failed:
        print(
            f"[警告] {len(failed)} 张图片推理失败，已跳过：",
            file=sys.stderr,
        )
        for path, reason in failed[:5]:
            print(f"    {path.name}: {reason}", file=sys.stderr)
        if len(failed) > 5:
            print(f"    ... 另有 {len(failed) - 5} 张", file=sys.stderr)

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
    used_stems: set[str] = set()
    skipped = 0
    for image_path, image, detections in dataset:
        src = Path(image_path)
        stem = _unique_stem(src, used_stems)
        used_stems.add(stem)
        dst = output_dir / f"{stem}{src.suffix}"
        shutil.copy2(str(src), str(dst))

        if image is not None and getattr(image, "shape", None) is not None:
            height, width = image.shape[:2]
        else:
            loaded = read_image(src)
            if loaded is None:
                print(
                    f"[警告] 无法读取图片尺寸，跳过该图：{src}",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            height, width = loaded.shape[:2]

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

    if skipped:
        print(
            f"[警告] {skipped} 张图片无法读取尺寸，未导出对应 JSON。",
            file=sys.stderr,
        )


def _flatten_yolo_labels(labels_dir: Path, output: Path) -> None:
    """把 ``labels/`` 下的标签平铺到 ``output`` 顶层。

    ``yolo_to_labelme.py`` 按 ``<--labels>/<图片 stem>.txt`` 找标签；supervision
    把标签写在 ``<output>/labels/`` 里，差一层目录会让回转拿到 0 个框却仍报告成功。
    """
    if not labels_dir.is_dir():
        return
    for txt in labels_dir.glob("*.txt"):
        shutil.move(str(txt), str(output / txt.name))
    try:
        labels_dir.rmdir()
    except OSError:
        pass  # 仍有子目录或残留文件时保留，不强行删除


def _rewrite_yolo_data_yaml(output: Path, class_names: list[str]) -> None:
    """用可训练格式重写 data.yaml。

    supervision 的 ``save_data_yaml`` 只写 ``nc`` / ``names``，缺 ``train`` / ``val``
    时 ultralytics 直接报 ``'train:' key missing``。这里补上 ``path`` 与
    指向图片目录的 ``train`` / ``val``。
    """
    images_dir = output / "images"
    has_images = images_dir.is_dir() and any(images_dir.iterdir())
    lines = [f"path: {output.resolve().as_posix()}"]
    if has_images:
        lines.append("train: images")
        lines.append("val: images")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names: [" + ", ".join(class_names) + "]")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_yolo_basename_collisions(dataset) -> None:
    """检测 YOLO 导出会发生的同名覆盖。

    supervision 的 ``as_yolo`` 按图片 basename 命名标签文件，不同子目录的同名
    图片会互相覆盖、静默丢标注。这里提前检测并报错，让调用方显式处理，而不是
    拿到一份少了标注的数据集却看到「成功」。

    Raises:
        ValueError: 存在同名图片。
    """
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for image_path in dataset.image_paths:
        path = Path(image_path)
        name = path.name
        if name in seen and seen[name] != str(path):
            collisions.append(f"{seen[name]} 与 {path}")
        seen.setdefault(name, str(path))
    if collisions:
        sample = "\n    ".join(collisions[:5])
        more = f"\n    ... 另有 {len(collisions) - 5} 组" if len(collisions) > 5 else ""
        raise ValueError(
            f"YOLO 导出按文件名平铺，存在 {len(collisions)} 组同名图片会互相覆盖：\n"
            f"    {sample}{more}\n"
            f"请改用 --format labelme（已按子目录去重），或先重命名同名图片。"
        )


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
        _check_yolo_basename_collisions(dataset)
        if copy_images:
            images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = output / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        dataset.as_yolo(
            images_directory_path=str(images_dir) if copy_images else None,
            annotations_directory_path=str(labels_dir),
            data_yaml_path=str(output / "data.yaml"),
        )
        _flatten_yolo_labels(labels_dir, output)
        _rewrite_yolo_data_yaml(output, dataset.classes)
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
    try:
        keep_class_ids = parse_class_ids(args.classes, labeler.classes)
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if getattr(args, "dry_run", False):
        # 与 onnx / mosaic / multi 保持一致：默认只预览，加 --apply 才写盘。
        names = (
            ", ".join(labeler.classes[i] for i in sorted(keep_class_ids))
            if keep_class_ids is not None
            else "全部类别"
        )
        print(f"[预览] 待标注 {len(images)} 张图片，类别：{names}")
        print(f"[预览] 输出目录：{args.output}（格式 {args.format}）")
        print("[预览] 未写盘；确认无误请加 --apply。")
        return 0

    dataset = build_detection_dataset(
        labeler,
        images,
        keep_class_ids,
        args.min_confidence,
    )
    if len(dataset) == 0:
        print("[错误] 所有图片推理失败，未产出任何标注。", file=sys.stderr)
        return 1
    export_detection_dataset(
        dataset,
        Path(args.output),
        args.format,
        args.copy_images,
    )
    return 0
