"""
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
# 1. 默认参数
# =============================================================================

DEFAULT_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")


# =============================================================================
# 2. 类别解析
# =============================================================================


def parse_classes(classes_arg: str) -> list[str]:
    """解析类别参数：逗号分隔字符串，或从 YAML 的 names 字段读取。"""
    path = Path(classes_arg)
    if path.is_file() and path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return list(names)

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


def convert_yolo_to_labelme(
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    output_dir: Path,
) -> int:
    """批量转换 YOLO 标注为 LabelMe JSON。

    Returns:
        成功生成的 JSON 数量。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(images_dir)
    if not images:
        print(f"[警告] 在 {images_dir} 中未找到图片。")
        return 0

    generated = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        data = build_labelme_data(image_path, label_path, class_names)
        save_labelme(data, output_dir / f"{image_path.stem}.json")
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

    class_names = parse_classes(args.classes)
    if not class_names:
        print("[错误] 未解析到类别名称。")
        return 1

    count = convert_yolo_to_labelme(
        args.images.resolve(),
        args.labels.resolve(),
        class_names,
        args.output.resolve(),
    )
    print(f"已生成 {count} 个 LabelMe JSON 到 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
