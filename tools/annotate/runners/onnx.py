"""
tools/annotate/runners/onnx.py

--model-type onnx 模式：ONNX 一路（可选 SAM 文本 prompt 第二路）打标，或
加 --reannotate 覆盖指定类别并保留其它类别。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import sys
from tqdm import tqdm

from tools.annotate.backends.onnx import OnnxDetector
from tools.annotate.backends.sam import SAMTextDetector
from tools.annotate.defaults import DEFAULT_OUTPUT_DIR
from tools.core import (
    build_batch_stage_dir,
    find_json_for_image,
    list_images,
    save_labelme,
)
from tools.annotate.ops import extract_existing_label_boxes, rewrite_labelme_dict
from tools.annotate.runners.common import (
    BoxSource,
    build_source,
    iter_annotated_images,
    iter_batch_image_files,
    parse_score_indices,
    resolve_keep_labels,
    run_multi_source_annotation,
)


def run_onnx(args) -> int:
    """执行 ONNX 打标 / 覆盖（``--model-type onnx``）。

    - 未加 ``--reannotate``：ONNX 一路（可选 SAM 第二路）标新框，写入输出目录；
    - 加 ``--reannotate``：ONNX 覆盖指定类别并保留其它类别，原地覆盖。
    """
    source = Path(args.source) if args.source else None
    output = Path(args.output) if args.output else None

    images: list[Path] = []
    if not args.reannotate:
        if source is None or not source.is_dir():
            print("[错误] 请通过 --source 指定输入图片目录。", file=sys.stderr)
            return 1
        images = list_images(source, recursive=args.recursive)
        if not images:
            print(f"[错误] 文件夹内没有图片：{source}", file=sys.stderr)
            return 1
        print(f"[信息] 共发现 {len(images)} 张图片")

    onnx_detector = OnnxDetector(
        args.onnx_model,
        args.onnx_label,
        args.onnx_conf,
        normalize=args.onnx_normalize,
        transpose=args.onnx_transpose,
        score_indices=parse_score_indices(args.onnx_score_indices),
    )
    sam_detector = None
    if getattr(args, "sam_model", None):
        # SAM 支持多个文本 prompt（逗号分隔），每个 prompt 对应一个类别名
        sam_prompts = [s.strip() for s in args.sam_prompt.split(",") if s.strip()]
        sam_labels = [s.strip() for s in args.sam_label.split(",") if s.strip()]
        sam_detector = SAMTextDetector(
            model_path=args.sam_model,
            label=sam_labels,
            conf=args.sam_conf,
            prompt=sam_prompts,
            device=args.device or None,
        )

    # 打码策略：标注模式默认马赛克；覆盖模式默认纯黑（与原脚本一致）。
    use_mosaic = args.mosaic if args.reannotate else (not args.blackout)

    # ---- 覆盖模式：仅在原地覆盖，不走输出目录 ----
    if args.reannotate:
        image_paths = iter_batch_image_files(args.input_root, args.start_batch, args.end_batch)
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
                print(f"[警告] 无法读取图片，跳过：{image_path}", file=sys.stderr)
                continue
            json_path = find_json_for_image(image_path)
            onnx_boxes, _ = onnx_detector.predict(image)
            sources = [
                build_source(onnx_boxes, args.onnx_label, args.onnx_min_ratio, image.shape)
            ]
            for label in resolve_keep_labels(json_path, args.onnx_label, keep_labels):
                keep_boxes = extract_existing_label_boxes(json_path, label)
                sources.append(build_source(keep_boxes, label, min_keep_ratio, image.shape))
            image, boxes, labels, n = run_multi_source_annotation(
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
    if not args.dry_run and output is not None:
        if output.resolve() == DEFAULT_OUTPUT_DIR.resolve():
            output = build_batch_stage_dir(output)
        else:
            output.mkdir(parents=True, exist_ok=True)
        args.output = output

    # --no-blackout：比例阈值视为 0，保留全部框（小框不打码、不删除），供后续合并。
    eff_onnx_ratio = 0.0 if args.no_blackout else args.onnx_min_ratio
    eff_sam_ratio = 0.0 if args.no_blackout else args.sam_min_ratio

    def _onnx_build_sources(image: np.ndarray, image_path: Path) -> list[BoxSource]:
        """ONNX 一路（可选 SAM 第二路）归一成 BoxSource 列表。"""
        onnx_boxes, _ = onnx_detector.predict(image)
        srcs = [
            build_source(onnx_boxes, args.onnx_label, eff_onnx_ratio, image.shape)
        ]
        if sam_detector is not None:
            sam_boxes, sam_labels = sam_detector.predict(image_path)
            srcs.append(build_source(sam_boxes, sam_labels, eff_sam_ratio, image.shape))
        return srcs

    total_onnx = total_sam = total_removed = total_mosaic = 0
    for _path, sources, _boxes, _labels, n in iter_annotated_images(
        images,
        _onnx_build_sources,
        use_mosaic=use_mosaic,
        mosaic_block=args.mosaic_block,
        dry_run=args.dry_run,
        output=output,
        source=source,
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
