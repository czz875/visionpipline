r"""
tools/convert/yolo_to_labelme.py
把 YOLO txt 标注批量转回 LabelMe JSON。

实现要点（按 AGENTS.md §4.3 优先调用官方库）：
- YOLO txt → ``supervision.Detections``：
  ``supervision.dataset.formats.yolo.yolo_annotations_to_detections``；
- ``Detections`` → LabelMe 字典：``tools.core.labelme.detections_to_labelme_dict``。
  （supervision 0.29.x 没有官方 labelme 适配，桥接放在 tools.core。）
- 自身只负责：扫图、构造 YOLO 行、调用转换、写盘。

典型用法：

    .conda\python.exe tools\convert\yolo_to_labelme.py ^
        --images datasets/02_remaining_70 ^
        --labels datasets/04_self_annotated_labels ^
        --classes datasets/03_yolo/data.yaml ^
        --output datasets/05_labelme_final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml
import supervision as sv
from PIL import Image
from supervision.dataset.formats.yolo import yolo_annotations_to_detections

from tools.core import (
    detections_to_labelme_dict,
    list_images,
    save_labelme,
)


# =============================================================================
# 1. 类别解析
# =============================================================================


def parse_classes(classes_arg: str) -> list[str]:
    """解析类别参数：逗号分隔字符串，或从 YAML 的 names 字段读取。

    ``classes_arg`` 形似 YAML 路径（以 ``.yaml`` / ``.yml`` 结尾）但文件不存在时
    直接抛 ``FileNotFoundError``：若静默当成类名字符串，会把路径本身写成类别名，
    只保留 class_id 0，其余类别的标注全部丢失。

    Raises:
        FileNotFoundError: 传入的是 YAML 路径但文件不存在。
        ValueError: YAML 内没有可用的 ``names`` 字段。
    """
    looks_like_yaml = Path(classes_arg).suffix.lower() in (".yaml", ".yml")
    path = Path(classes_arg)
    if looks_like_yaml and not path.is_file():
        raise FileNotFoundError(f"类别 YAML 不存在：{classes_arg}")
    if path.is_file() and looks_like_yaml:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=lambda x: int(x))]
        names = [str(n) for n in names]
        if not names:
            raise ValueError(f"类别 YAML 中没有 names 字段：{classes_arg}")
        return names

    return [c.strip() for c in classes_arg.split(",") if c.strip()]


# =============================================================================
# 3. 转换核心（YOLO → LabelMe，借 supervision 之力）
# =============================================================================


def build_labelme_data(
    image_path: Path,
    label_path: Path,
    class_names: list[str],
) -> dict:
    """根据图片与 YOLO txt 生成 LabelMe 字典。

    流程：PIL 读图拿尺寸 → ``yolo_annotations_to_detections`` 拿 Detections
    → ``detections_to_labelme_dict`` 转 LabelMe dict。
    """
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    if label_path.exists():
        lines = [
            line for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if lines:
            detections = yolo_annotations_to_detections(
                lines=lines,
                resolution_wh=(image_width, image_height),
                with_masks=False,
            )
        else:
            detections = sv.Detections.empty()
    else:
        detections = sv.Detections.empty()

    return detections_to_labelme_dict(
        detections,
        class_names=class_names,
        image_path=image_path.name,
        image_width=image_width,
        image_height=image_height,
    )


def _unique_stem(image_path: Path, images_dir: Path, used: set[str]) -> str:
    """为图片生成在输出目录内唯一的 stem。

    输入目录可能含子目录，不同子目录里的同名图片若直接按 ``stem`` 平铺输出，
    后写的会覆盖先写的，静默丢标注且计数虚高。冲突时用父目录名做前缀，
    与 ``supervision`` 导出的 ``_unique_stem`` 思路一致。
    """
    if image_path.stem not in used:
        return image_path.stem
    try:
        rel_parent = image_path.parent.relative_to(images_dir)
        prefix = "_".join(rel_parent.parts) if rel_parent.parts else ""
    except ValueError:
        prefix = image_path.parent.name
    if prefix:
        candidate = f"{prefix}_{image_path.stem}"
        if candidate not in used:
            return candidate
        i = 1
        while f"{candidate}_{i}" in used:
            i += 1
        return f"{candidate}_{i}"
    i = 1
    while f"{image_path.stem}_{i}" in used:
        i += 1
    return f"{image_path.stem}_{i}"


def convert_yolo_to_labelme(
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    output_dir: Path,
) -> int:
    """批量转换 YOLO 标注为 LabelMe JSON。

    标签目录或单张图的标签文件缺失时抛 ``FileNotFoundError``：静默当成空标注
    会写出 shapes=0 的 JSON 并报告成功，下游拿到的是「全部无框」的数据。

    空 txt（0 行）是合法的负样本，照常写成空 shapes。

    不同子目录的同名图片通过 ``_unique_stem`` 去重，不会互相覆盖。

    Returns:
        成功生成的 JSON 数量。
    """
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"标签目录不存在：{labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(images_dir)
    if not images:
        print(f"[警告] 在 {images_dir} 中未找到图片。")
        return 0

    generated = 0
    used_stems: set[str] = set()
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(
                f"图片 {image_path.name} 的标签文件不存在：{label_path}"
            )
        stem = _unique_stem(image_path, images_dir, used_stems)
        used_stems.add(stem)
        data = build_labelme_data(image_path, label_path, class_names)
        # imagePath 必须与输出文件名一致，否则 LabelMe 打开后指向另一张图。
        data["imagePath"] = f"{stem}{image_path.suffix}"
        save_labelme(data, output_dir / f"{stem}.json")
        generated += 1

    return generated


# =============================================================================
# 4. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 YOLO txt 标注批量转换为 LabelMe JSON。",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="图片目录。",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="YOLO labels 目录（与图片同名 .txt）。",
    )
    parser.add_argument(
        "--classes",
        required=True,
        help="类别名称，逗号分隔；或传入 data.yaml 路径自动解析 names。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="LabelMe JSON 输出目录。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        class_names = parse_classes(args.classes)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"[错误] {e}")
        return 1
    if not class_names:
        print("[错误] 未解析到类别名称。")
        return 1

    try:
        count = convert_yolo_to_labelme(
            args.images.resolve(),
            args.labels.resolve(),
            class_names,
            args.output.resolve(),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}")
        return 1
    print(f"已生成 {count} 个 LabelMe JSON 到 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
