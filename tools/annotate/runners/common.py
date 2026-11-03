"""
tools/annotate/runners/common.py

标注 runner 的公共基础设施：BoxSource 数据类、多源合并打码、通用标注循环、
批量文件发现等。被 onnx / mosaic / multi 等 runner 复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import sys

import numpy as np
from tqdm import tqdm

from tools.core import (
    BATCH_DIR_RE,
    list_images,
    load_labelme,
    read_image,
    save_labelme,
    write_image,
)
from tools.annotate.ops import (
    apply_blackout,
    classify_box_by_ratio,
    clip_box,
    concat_boxes,
    extract_existing_label_boxes,
    rewrite_labelme_dict,
)


# =============================================================================
# BoxSource：单路检测结果的归一化表示
# =============================================================================


@dataclass
class BoxSource:
    """单路检测/保留来源的框与标签，已按面积占比切分为保留/删除两部分。

    ``kept_labels`` 与 ``kept`` 逐框对齐，支持同一路检测器输出多个类别
    （如一个 YOLO 模型同时给出 face / phone / hand）。删除的小框最终会被
    打码删掉，无需保留标签。
    """

    kept: np.ndarray
    removed: np.ndarray
    kept_labels: list[str]


def build_source(
    boxes: np.ndarray,
    label: str | list[str],
    min_ratio: float,
    image_shape: tuple[int, int, int],
) -> BoxSource:
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
    return BoxSource(kept=kept, removed=removed, kept_labels=kept_labels)


# =============================================================================
# 多源合并 / 写盘 / 循环
# =============================================================================


def run_multi_source_annotation(
    image: np.ndarray,
    sources: list[BoxSource],
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


def write_annotation_result(
    image: np.ndarray,
    image_path: Path,
    image_name: str,
    boxes: np.ndarray,
    labels: list[str],
    output: Path | None,
    *,
    source: Path | None = None,
) -> None:
    """把单张标注结果（打码后图像 + 框/标签）写入输出目录的图片与同名 JSON。

    ``output`` 为 ``None`` 时不写盘（调用方负责处理覆盖模式等的原地写盘）。
    传入 ``source`` 时，会在输出目录下保留相对于 source 的子目录结构。

    Raises:
        OSError: 图片写盘失败。此时**不写** JSON，避免留下「有 JSON 无图」
            的脏数据；由调用方决定是否跳过该图继续。
    """
    if output is None:
        return
    if source is not None:
        try:
            rel = image_path.relative_to(source)
            out_img = output / rel
        except ValueError:
            out_img = output / image_path.name
    else:
        out_img = output / image_path.name
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_img.with_suffix(".json")
    write_image(out_img, image)
    data = rewrite_labelme_dict(out_json, boxes, labels, image.shape, image_name)
    save_labelme(data, out_json)


def iter_annotated_images(
    image_paths: list[Path],
    build_sources: Callable[[np.ndarray, Path], list[BoxSource]],
    *,
    use_mosaic: bool,
    mosaic_block: int,
    dry_run: bool,
    output: Path | None,
    source: Path | None = None,
    failures: list[Path] | None = None,
):
    """通用标注循环：推理建源 -> 多源打码合并 -> 写盘，逐张 yield 统计要素。

    ``build_sources(image, image_path)`` 由调用方提供，负责把各检测器输出归一成
    ``BoxSource`` 列表（ONNX 一路、SAM 第二路、多检测器组合的差异都封在这里）。
    循环主体因此在所有「标注模式」间复用，避免重复遍历/写盘逻辑。
    每个元素 yield ``(image_path, sources, boxes, labels, blackout_count)``。

    单张图读取、推理或写盘失败时跳过该图继续，不中断整批：磁盘满、路径过长、
    文件被占用都可能让写盘抛 OSError，一张图失败不该让整批已处理的结果白跑。
    失败路径追加到调用方传入的 ``failures`` 列表，调用方可据此决定返回码。
    """
    if failures is None:
        failures = []
    for image_path in tqdm(image_paths, desc="标注中", unit="img"):
        image = read_image(image_path)
        if image is None:
            print(f"[警告] 无法读取图片，跳过：{image_path}", file=sys.stderr)
            failures.append(image_path)
            continue
        try:
            sources = build_sources(image, image_path)
        except Exception as e:  # 单图推理失败不该终止整批
            print(
                f"[警告] 推理失败，跳过该图：{image_path}（{type(e).__name__}: {e}）",
                file=sys.stderr,
            )
            failures.append(image_path)
            continue
        image, boxes, labels, n = run_multi_source_annotation(
            image,
            sources,
            use_mosaic=use_mosaic,
            mosaic_block=mosaic_block,
            dry_run=dry_run,
        )
        if not dry_run:
            try:
                write_annotation_result(
                    image, image_path, image_path.name, boxes, labels, output,
                    source=source,
                )
            except OSError as e:
                print(f"[警告] 写盘失败，跳过该图：{e}", file=sys.stderr)
                failures.append(image_path)
                continue
        yield image_path, sources, boxes, labels, n


# =============================================================================
# 工具函数
# =============================================================================


def parse_score_indices(text: str) -> tuple[int, ...]:
    """解析 ``"4,15"`` 形式的分通道下标为 int 元组。"""
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


def resolve_keep_labels(
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


def iter_batch_image_files(
    root: Path,
    start_batch: int | None,
    end_batch: int | None,
) -> list[Path]:
    """收集指定 batch 范围内存在的图片（按 ``0000``~``9999`` 子目录组织）。

    ``start_batch`` / ``end_batch`` 传 ``None`` 表示不限范围，改由 ``root`` 下
    实际存在的 4 位数字批次目录决定——这样新增批次不会被静默漏掉。
    """
    if start_batch is None and end_batch is None:
        batch_dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and BATCH_DIR_RE.match(p.name)
        ) if root.is_dir() else []
    else:
        start = 0 if start_batch is None else start_batch
        end = 9999 if end_batch is None else end_batch
        batch_dirs = [root / f"{batch_id:04d}" for batch_id in range(start, end + 1)]

    image_paths: list[Path] = []
    for batch_dir in batch_dirs:
        if batch_dir.is_dir():
            image_paths.extend(list_images(batch_dir, recursive=True))
    return image_paths
