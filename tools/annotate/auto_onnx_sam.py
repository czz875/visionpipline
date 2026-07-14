"""
tools/annotate/auto_onnx_sam.py
新工作流：用 ONNX 检测器标注一类目标 + 用 SAM 文本 prompt 检测器标注另一类目标，
合并成一份 LabelMe 数据集；随后面积小于图片指定比例的框打马赛克后从标注中删除，
但小框与保留大框的重叠区域不做马赛克（重叠保护）。

类别与模型完全由参数决定（不再写死 face / hand）：
- ONNX 一路：``--onnx-model / --onnx-label / --onnx-conf / --onnx-min-ratio``
  以及适配不同模型输出的 ``--onnx-transpose / --onnx-score-indices / --onnx-normalize``；
- SAM 一路：``--sam-model / --sam-prompt / --sam-label / --sam-conf / --sam-min-ratio``。

复用 tools/annotate/backends 中的检测器后端（OnnxDetector / SAMTextDetector）与
tools/annotate/ops 中的框几何、打码、LabelMe IO，本脚本只负责编排两路结果并打码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.annotate.backends.onnx import OnnxDetector
from tools.annotate.backends.sam import SAMTextDetector
from tools.annotate.ops import (
    collect_blackout_regions,
    concat_boxes,
    mosaic_region,
    rewrite_labelme_dict,
    sanitize_boxes,
    split_boxes_by_ratio,
)
from tools.core import list_images, save_labelme


# =============================================================================
# 1. 默认参数（集中在文件顶部，argparse / 回退值统一引用）
# =============================================================================

# 默认 ONNX 模型（示例：人脸 yolov5s-lmk.onnx）及其输出解码配套参数。
DEFAULT_ONNX_MODEL = r"weight\yolov5s-lmk.onnx"
DEFAULT_ONNX_LABEL = "face"
DEFAULT_ONNX_CONF = 0.25
DEFAULT_ONNX_MIN_RATIO = 0.01
DEFAULT_ONNX_TRANSPOSE = False
DEFAULT_ONNX_SCORE_INDICES = "4,15"   # 人脸模型：obj 分 * 关键点置信度
DEFAULT_ONNX_NORMALIZE = True

# 默认 SAM 文本 prompt 检测（示例：hand）。
DEFAULT_SAM_MODEL = r"weight\sam3.1_multiplex.pt"
DEFAULT_SAM_PROMPT = "hand"
DEFAULT_SAM_LABEL = "hand"
DEFAULT_SAM_CONF = 0.25
DEFAULT_SAM_MIN_RATIO = 0.01

DEFAULT_SOURCE = "datasets/raw"
DEFAULT_OUTPUT = "datasets/01_annotated_onnx_sam"
DEFAULT_MOSAIC_BLOCK = 16          # 马赛克块大小（像素），越大越糊
DEFAULT_DEVICE = ""
DEFAULT_RECURSIVE = True
DEFAULT_APPLY = False


# =============================================================================
# 2. 单图处理
# =============================================================================


def process_image(
    image_path: Path,
    onnx_detector: OnnxDetector,
    sam_detector: SAMTextDetector,
    onnx_label: str,
    sam_label: str,
    min_ratio_onnx: float,
    min_ratio_sam: float,
    mosaic_block: int,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, int]:
    """对单张图做 ONNX 标一路 + SAM 标一路，并按规则打码后删小框。

    返回统计字典；非 dry_run 时把（打码后的）图片与 LabelMe JSON 写到
    输出目录。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")

    # 1) 两个检测器各出框（坐标系均为原图像素）。
    onnx_boxes, _ = onnx_detector.predict(image)
    sam_boxes, _ = sam_detector.predict(image_path)
    onnx_boxes = sanitize_boxes(onnx_boxes, image.shape)
    sam_boxes = sanitize_boxes(sam_boxes, image.shape)

    # 2) 按各自面积占比切分为「保留大框」与「待删小框」。
    kept_onnx, removed_onnx = split_boxes_by_ratio(onnx_boxes, image.shape, min_ratio_onnx)
    kept_sam, removed_sam = split_boxes_by_ratio(sam_boxes, image.shape, min_ratio_sam)

    # 3) 保留大框（>=比例）作为「重叠保护区」。
    all_kept = concat_boxes([kept_onnx, kept_sam])

    # 4) 待删小框 = 两路所有 <比例 的框；扣减与保留大框的重叠区域。
    removed_all = concat_boxes([removed_onnx, removed_sam])
    mosaic_regions = collect_blackout_regions(removed_all, all_kept)

    # 5) 在图片副本上打马赛克（仅残余碎片区域）。
    if not dry_run:
        for box in mosaic_regions:
            image = mosaic_region(image, box, mosaic_block)

    # 6) 最终保留的框 = 全部大框（两路）。
    final_boxes = all_kept
    final_labels = [onnx_label] * len(kept_onnx) + [sam_label] * len(kept_sam)

    stats = {
        "onnx_count": len(kept_onnx),
        "sam_count": len(kept_sam),
        "removed_count": len(removed_onnx) + len(removed_sam),
        "mosaic_count": len(mosaic_regions),
    }

    if not dry_run:
        out_img = output_dir / image_path.name
        out_json = out_img.with_suffix(".json")
        cv2.imwrite(str(out_img), image)
        data = rewrite_labelme_dict(
            out_json,
            final_boxes,
            final_labels,
            image.shape,
            image_path.name,
        )
        save_labelme(data, out_json)

    return stats


# =============================================================================
# 3. 命令行参数
# =============================================================================


def _parse_score_indices(text: str) -> tuple[int, ...]:
    """解析 ``"4,15"`` 形式的分通道下标为 int 元组。"""
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="ONNX 标一路 + SAM 文本 prompt 标一路，输出 LabelMe；"
        "面积小于阈值的小框打马赛克后删除（与保留大框重叠部分不打码）。",
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"输入图片目录（默认：{DEFAULT_SOURCE}）。",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出 LabelMe 目录（图片与同名 .json 同目录；默认：{DEFAULT_OUTPUT}）。",
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

    # ----- SAM 一路 -----
    parser.add_argument(
        "--sam-model", type=Path, default=Path(DEFAULT_SAM_MODEL),
        help=f"SAM 文本 prompt 模型路径（默认：{DEFAULT_SAM_MODEL}）。",
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

    parser.add_argument(
        "--mosaic-block", type=int, default=DEFAULT_MOSAIC_BLOCK,
        help=f"马赛克块大小（像素），越大越糊（默认：{DEFAULT_MOSAIC_BLOCK}）。",
    )
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE,
        help="SAM 推理设备，如 cpu / 0 / cuda，留空自动选择。",
    )
    parser.add_argument(
        "--recursive", action="store_true", default=DEFAULT_RECURSIVE,
        help="递归扫描子目录（默认开启）。",
    )
    parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="关闭递归，只扫描顶层目录。",
    )
    parser.add_argument(
        "--apply", dest="dry_run", action="store_false",
        help="真正写盘（图片打码 + 写 JSON）；默认仅统计预览。",
    )
    parser.set_defaults(dry_run=not DEFAULT_APPLY)
    return parser


# =============================================================================
# 4. 主入口
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        print(f"[错误] 图片文件夹不存在：{source}", file=sys.stderr)
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
        score_indices=_parse_score_indices(args.onnx_score_indices),
    )
    sam_detector = SAMTextDetector(
        model_path=args.sam_model,
        label=args.sam_label,
        conf=args.sam_conf,
        prompt=args.sam_prompt,
        device=args.device or None,
    )

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    total_onnx = total_sam = total_removed = total_mosaic = 0
    for image_path in tqdm(images, desc="标注中", unit="img"):
        stats = process_image(
            image_path=image_path,
            onnx_detector=onnx_detector,
            sam_detector=sam_detector,
            onnx_label=args.onnx_label,
            sam_label=args.sam_label,
            min_ratio_onnx=args.onnx_min_ratio,
            min_ratio_sam=args.sam_min_ratio,
            mosaic_block=args.mosaic_block,
            output_dir=output,
            dry_run=args.dry_run,
        )
        total_onnx += stats["onnx_count"]
        total_sam += stats["sam_count"]
        total_removed += stats["removed_count"]
        total_mosaic += stats["mosaic_count"]

    mode = "预览" if args.dry_run else "完成"
    print(f"[{mode}] 图片数：{len(images)}")
    print(f"[{mode}] 保留 {args.onnx_label}：{total_onnx}")
    print(f"[{mode}] 保留 {args.sam_label}：{total_sam}")
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


if __name__ == "__main__":
    raise SystemExit(main())
