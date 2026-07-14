"""
tools/annotate/auto_hand_face.py
新工作流：用 SAM 标注 hand + 用 yolov5s-lmk.onnx 标注 face，输出 LabelMe 格式；
随后面积小于图片 1% 的小框（face / hand 都算）打马赛克后从标注中删除，
但小框与保留大框（>=1%）的重叠区域不做马赛克。

复用 tools/annotate/reannotate_face_hand_onnx.py 中已验证的
OnnxDetector / SAMTextDetector / 框几何与重叠扣减逻辑，本脚本只负责：
- 把两个检测器的输出合并成一个 LabelMe 数据集；
- 用「马赛克」替代「纯黑填充」实现小框打码（重叠保护逻辑原样复用）。
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

# 复用已验证的人脸/手势检测与框几何逻辑，避免重复造轮子。
from tools.annotate.reannotate_face_hand_onnx import (  # noqa: E402
    OnnxDetector,
    SAMTextDetector,
    clip_box,
    collect_blackout_regions,
    rewrite_labelme_dict,
    sanitize_boxes,
    split_boxes_by_ratio,
)
from tools.core import list_images, save_labelme  # noqa: E402


# =============================================================================
# 1. 默认参数（集中在文件顶部，argparse / 回退值统一引用）
# =============================================================================

DEFAULT_FACE_MODEL = r"weight\yolov5s-lmk.onnx"
DEFAULT_HAND_MODEL = r"weight\sam3.1_multiplex.pt"
DEFAULT_HAND_PROMPT = "hand"
DEFAULT_SOURCE = "datasets/raw"
DEFAULT_OUTPUT = "datasets/01_annotated_hand_face"
DEFAULT_MIN_RATIO = 0.01           # 面积占比 < 1% 的框视为小框（打码后删除）
DEFAULT_FACE_CONF = 0.25
DEFAULT_HAND_CONF = 0.25
DEFAULT_MOSAIC_BLOCK = 16          # 马赛克块大小（像素），越大越糊
DEFAULT_DEVICE = ""
DEFAULT_RECURSIVE = True
DEFAULT_APPLY = False
DEFAULT_FACE_LABEL = "face"
DEFAULT_HAND_LABEL = "hand"


# =============================================================================
# 2. 马赛克（核心新增能力）
# =============================================================================


def mosaic_region(
    image: np.ndarray,
    box: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """把框内区域打成马赛克（先缩小再最近邻放大，保留边界硬块感）。

    只处理框内像素；框外与重叠部分由调用方通过 ``box`` 预先扣减好，
    本函数不关心重叠逻辑。
    """
    clipped = clip_box(box, image.shape)
    x1, y1, x2, y2 = (int(round(v)) for v in clipped.tolist())
    if x2 <= x1 or y2 <= y1:
        return image
    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return image
    # 缩小到 (w//block, h//block)，再放大回原尺寸，得到马赛克块。
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    return image


# =============================================================================
# 3. 单图处理
# =============================================================================


def process_image(
    image_path: Path,
    face_detector: OnnxDetector,
    hand_detector: SAMTextDetector,
    min_ratio: float,
    mosaic_block: int,
    dry_run: bool,
) -> dict[str, int]:
    """对单张图做 SAM 标 hand + ONNX 标 face，并按规则打码后删小框。

    返回统计字典；非 dry_run 时把（打码后的）图片与 LabelMe JSON 写到
    输出目录。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")

    # 1) 两个检测器各出框（坐标系均为原图像素）。
    face_boxes, _ = face_detector.predict(image)
    hand_boxes, _ = hand_detector.predict(image_path)
    face_boxes = sanitize_boxes(face_boxes, image.shape)
    hand_boxes = sanitize_boxes(hand_boxes, image.shape)

    # 2) 按 1% 面积占比切分为「保留大框」与「待删小框」。
    kept_faces, removed_faces = split_boxes_by_ratio(
        face_boxes, image.shape, min_ratio
    )
    kept_hands, removed_hands = split_boxes_by_ratio(
        hand_boxes, image.shape, min_ratio
    )

    # 3) 保留大框（>=1%）作为「重叠保护区」：所有保留框都算大框。
    kept_parts: list[np.ndarray] = []
    if len(kept_faces):
        kept_parts.append(kept_faces)
    if len(kept_hands):
        kept_parts.append(kept_hands)
    all_kept = (
        np.concatenate(kept_parts, axis=0)
        if kept_parts
        else np.empty((0, 4), dtype=np.float32)
    )

    # 4) 待删小框 = 所有 <1% 的框；扣减与保留大框的重叠区域，
    #    得到「真正需要打码」的残余碎片（重叠部分不打码）。
    removed_all_parts: list[np.ndarray] = []
    if len(removed_faces):
        removed_all_parts.append(removed_faces)
    if len(removed_hands):
        removed_all_parts.append(removed_hands)
    removed_all = (
        np.concatenate(removed_all_parts, axis=0)
        if removed_all_parts
        else np.empty((0, 4), dtype=np.float32)
    )
    mosaic_regions = collect_blackout_regions(removed_all, all_kept)

    # 5) 在图片副本上打马赛克（仅残余碎片区域）。
    if not dry_run:
        for box in mosaic_regions:
            image = mosaic_region(image, box, mosaic_block)

    # 6) 最终保留的框 = 全部大框（face + hand）。
    final_boxes = all_kept
    final_labels = [DEFAULT_FACE_LABEL] * len(kept_faces) + [
        DEFAULT_HAND_LABEL
    ] * len(kept_hands)

    stats = {
        "face_count": len(kept_faces),
        "hand_count": len(kept_hands),
        "removed_count": len(removed_faces) + len(removed_hands),
        "mosaic_count": len(mosaic_regions),
    }

    if not dry_run:
        out_img = output_dir_for() / image_path.name
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


# 输出目录在 main 里赋值，供 process_image 写出图片使用。
_output_dir: Path | None = None


def output_dir_for() -> Path:
    """返回当前图片应写出的输出目录（由 main 初始化）。"""
    assert _output_dir is not None, "输出目录尚未初始化"
    return _output_dir


# =============================================================================
# 4. 命令行参数
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="SAM 标 hand + yolov5s-lmk.onnx 标 face，输出 LabelMe；"
        "面积<1%的小框打马赛克后删除（与保留大框重叠部分不打码）。",
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"输入图片目录（默认：{DEFAULT_SOURCE}）。",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出 LabelMe 目录（图片与同名 .json 同目录；默认：{DEFAULT_OUTPUT}）。",
    )
    parser.add_argument(
        "--face-model", type=Path, default=Path(DEFAULT_FACE_MODEL),
        help=f"人脸 ONNX 模型路径（默认：{DEFAULT_FACE_MODEL}）。",
    )
    parser.add_argument(
        "--hand-model", type=Path, default=Path(DEFAULT_HAND_MODEL),
        help=f"SAM 手势模型路径（默认：{DEFAULT_HAND_MODEL}）。",
    )
    parser.add_argument(
        "--hand-prompt", default=DEFAULT_HAND_PROMPT,
        help=f"SAM 文本提示（默认：{DEFAULT_HAND_PROMPT}）。",
    )
    parser.add_argument(
        "--min-ratio", type=float, default=DEFAULT_MIN_RATIO,
        help=f"保留框的最小面积占比；小于此值的框打码后删除（默认：{DEFAULT_MIN_RATIO}）。",
    )
    parser.add_argument(
        "--face-conf", type=float, default=DEFAULT_FACE_CONF,
        help=f"人脸置信度阈值（默认：{DEFAULT_FACE_CONF}）。",
    )
    parser.add_argument(
        "--hand-conf", type=float, default=DEFAULT_HAND_CONF,
        help=f"SAM 手势置信度阈值（默认：{DEFAULT_HAND_CONF}）。",
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
# 5. 主入口
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

    # ONNX 人脸检测器（label="face" 触发其归一化 / IoU 默认配置）。
    face_detector = OnnxDetector(args.face_model, DEFAULT_FACE_LABEL, args.face_conf)
    # SAM 文本提示检测器（label="hand"）。
    hand_detector = SAMTextDetector(
        model_path=args.hand_model,
        label=DEFAULT_HAND_LABEL,
        conf=args.hand_conf,
        prompt=args.hand_prompt,
        device=args.device or None,
    )

    global _output_dir
    _output_dir = output
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    total_face = total_hand = total_removed = total_mosaic = 0
    for image_path in tqdm(images, desc="标注中", unit="img"):
        stats = process_image(
            image_path=image_path,
            face_detector=face_detector,
            hand_detector=hand_detector,
            min_ratio=args.min_ratio,
            mosaic_block=args.mosaic_block,
            dry_run=args.dry_run,
        )
        total_face += stats["face_count"]
        total_hand += stats["hand_count"]
        total_removed += stats["removed_count"]
        total_mosaic += stats["mosaic_count"]

    mode = "预览" if args.dry_run else "完成"
    print(f"[{mode}] 图片数：{len(images)}")
    print(f"[{mode}] 保留 face：{total_face}")
    print(f"[{mode}] 保留 hand：{total_hand}")
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
