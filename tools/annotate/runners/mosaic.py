"""
tools/annotate/runners/mosaic.py

--mosaic-existing 模式：对已有 LabelMe 标注的小框打码并删除（原地写回）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import sys
from tqdm import tqdm

from tools.core import find_json_for_image, list_images, load_labelme, rect_to_xyxy, save_labelme
from tools.annotate.ops import rewrite_labelme_dict
from tools.annotate.runners.common import (
    BoxSource,
    build_source,
    run_multi_source_annotation,
)


def run_mosaic(args) -> int:
    """对 ``--source`` 目录下已有 LabelMe 标注的小框打码并删除（原地写回）。

    用于「检测（``--no-blackout``）→ 合并（``merge.py``）→ 打码（本步）」
    流程的最后一步：读取每张图的 JSON，按面积占比把小框打成马赛克（或纯黑）
    并从标注中删除，大框保留（共同构成重叠保护区）；默认预览，加 ``--apply`` 才写盘。
    """
    source = Path(args.source) if args.source else None
    if source is None or not source.is_dir():
        print("[错误] 请通过 --source 指定已标注目录（图片 + JSON 同目录）。", file=sys.stderr)
        return 1
    images = list_images(source, recursive=args.recursive)
    if not images:
        print(f"[错误] 文件夹内没有图片：{source}", file=sys.stderr)
        return 1
    print(f"[信息] 共发现 {len(images)} 张图片")

    use_mosaic = not args.blackout
    min_ratio = args.mosaic_min_ratio if args.mosaic_min_ratio is not None else args.onnx_min_ratio
    mosaic_label_set = None
    if args.mosaic_labels:
        mosaic_label_set = {s.strip() for s in args.mosaic_labels.split(",") if s.strip()}

    total_kept = total_removed = total_mosaic = 0
    for image_path in tqdm(images, desc="打码中", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[警告] 无法读取图片，跳过：{image_path}", file=sys.stderr)
            continue
        json_path = find_json_for_image(image_path)
        if not json_path or not json_path.exists():
            continue
        data = load_labelme(json_path)
        # 按标签分组收集矩形框
        by_label: dict[str, list] = {}
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue
            pts = shape.get("points", [])
            if len(pts) < 2:
                continue
            by_label.setdefault(shape["label"], []).append(rect_to_xyxy(pts))
        sources: list[BoxSource] = []
        for label, boxes in by_label.items():
            boxes_arr = np.array(boxes, dtype=np.float32)
            if mosaic_label_set is not None and label not in mosaic_label_set:
                # 非打码标签：全部保留，既不打码也作为重叠保护区
                sources.append(build_source(boxes_arr, label, 0.0, image.shape))
            else:
                sources.append(build_source(boxes_arr, label, min_ratio, image.shape))
        image, boxes, labels, n = run_multi_source_annotation(
            image,
            sources,
            use_mosaic=use_mosaic,
            mosaic_block=args.mosaic_block,
            dry_run=args.dry_run,
        )
        total_kept += len(boxes)
        total_removed += sum(len(s.removed) for s in sources)
        total_mosaic += n
        if not args.dry_run:
            data = rewrite_labelme_dict(json_path, boxes, labels, image.shape, image_path.name)
            cv2.imwrite(str(image_path), image)
            save_labelme(data, json_path)

    mode = "预览" if args.dry_run else "完成"
    print(f"[{mode}] 图片数：{len(images)}")
    print(f"[{mode}] 保留框：{total_kept}")
    print(f"[{mode}] 小框(已删)：{total_removed}")
    print(
        f"[{mode}] 实际打码区域：{total_mosaic}"
        f"（重叠被保护跳过：{total_removed - total_mosaic}）"
    )
    if args.dry_run:
        print("[提示] 当前为预览模式，未写盘；确认无误请加 --apply。")
    return 0
