"""
tools/annotate/runners/multi.py

--detectors-config 模式：用 YAML 描述任意 N 路检测器（onnx/sam/yolo/detr，
可混搭、同类型可多路），逐路归一成「框 + 逐框标签」后输出 LabelMe。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sys
from tqdm import tqdm

from tools.annotate.backends import (
    AutoLabeler,
    DETRLabeler,
    YOLOLabeler,
)
from tools.annotate.backends.onnx import OnnxDetector
from tools.annotate.backends.sam import SAMTextDetector
from tools.core import list_images
from tools.annotate.defaults import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_ONNX_CONF,
    DEFAULT_ONNX_NORMALIZE,
    DEFAULT_ONNX_SCORE_INDICES,
    DEFAULT_ONNX_TRANSPOSE,
    DEFAULT_SAM_CONF,
    DEFAULT_VERBOSE,
)
from tools.annotate.runners.common import (
    BoxSource,
    build_source,
    iter_annotated_images,
    parse_score_indices,
)


# =============================================================================
# 检测器封装
# =============================================================================


@dataclass
class Detector:
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
        return parse_score_indices(value)
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
) -> list[Detector]:
    """按 YAML 配置逐项构造检测器（模型只加载一次），返回统一封装列表。

    支持 ``onnx`` / ``sam`` / ``yolo`` / ``detr`` 混搭且同类型可多路。
    每项通用键：``type`` / ``model`` / ``min_ratio`` / ``conf`` / ``name``。
    """
    detectors: list[Detector] = []
    for idx, cfg in enumerate(detector_cfgs):
        dtype = str(cfg.get("type", "")).lower()
        min_ratio = float(cfg.get("min_ratio", default_min_ratio))
        name = str(cfg.get("name") or f"{dtype}#{idx}")
        device = cfg.get("device") or default_device

        if dtype == "onnx":
            backend = OnnxDetector(
                Path(str(cfg["model"])),
                str(cfg.get("label", "object")),
                float(cfg.get("conf", DEFAULT_ONNX_CONF)),
                normalize=bool(cfg.get("normalize", DEFAULT_ONNX_NORMALIZE)),
                transpose=bool(cfg.get("transpose", DEFAULT_ONNX_TRANSPOSE)),
                score_indices=_coerce_score_indices(
                    cfg.get("score_indices", DEFAULT_ONNX_SCORE_INDICES)
                ),
                iou_threshold=float(cfg.get("iou", DEFAULT_IOU)),
            )

            def detect(
                image: np.ndarray,
                _path: Path,
                backend: OnnxDetector = backend,
            ) -> tuple[np.ndarray, list[str]]:
                return backend.predict(image)

        elif dtype == "sam":
            backend = SAMTextDetector(
                model_path=Path(str(cfg["model"])),
                label=str(cfg.get("label", cfg.get("prompt", "object"))),
                conf=float(cfg.get("conf", DEFAULT_SAM_CONF)),
                prompt=str(cfg["prompt"]),
                device=device or None,
            )

            def detect(
                _image: np.ndarray,
                image_path: Path,
                backend: SAMTextDetector = backend,
            ) -> tuple[np.ndarray, list[str]]:
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
            keep_names: set[str] = {str(c) for c in cfg.get("classes", [])}
            rename = cfg.get("label")

            def detect(
                _image: np.ndarray,
                image_path: Path,
                backend: AutoLabeler = backend,
                keep: set[str] = keep_names,
                rename: str | None = rename,
            ) -> tuple[np.ndarray, list[str]]:
                return _sv_detect(backend, image_path, keep, rename)

        else:
            raise ValueError(f"未知检测器类型：{dtype}")

        detectors.append(Detector(name=name, min_ratio=min_ratio, detect=detect))
    return detectors


# =============================================================================
# 入口
# =============================================================================


def run_multi(args) -> int:
    """执行多检测器组合标注（``--detectors-config <yaml>``）。

    从 YAML 读取任意 N 路检测器与全局项，逐路归一成「框 + 逐框标签」后复用
    ``run_multi_source_annotation`` 的打码合并链路，输出 LabelMe。全局项
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
    default_min_ratio = float(cfg.get("min_ratio", args.onnx_min_ratio))
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

    def _det_build_sources(image: np.ndarray, image_path: Path) -> list[BoxSource]:
        """各路检测器归一化为 BoxSource，并累计每路保留框数。"""
        srcs: list[BoxSource] = []
        for detector in detectors:
            boxes, labels = detector.detect(image, image_path)
            source_boxes = build_source(boxes, labels, detector.min_ratio, image.shape)
            per_kept[detector.name] += len(source_boxes.kept)
            srcs.append(source_boxes)
        return srcs

    for _path, sources, _boxes, _labels, n in iter_annotated_images(
        images,
        _det_build_sources,
        use_mosaic=use_mosaic,
        mosaic_block=mosaic_block,
        dry_run=args.dry_run,
        output=output,
        source=source,
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
