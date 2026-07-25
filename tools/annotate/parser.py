"""
tools/annotate/parser.py

auto.py 的命令行参数解析器。把所有 --model-type / --detectors-config /
ONNX / SAM / 覆盖 / 打码策略等参数集中在这里定义，避免 auto.py 被参数
声明撑得过长。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.annotate.defaults import (
    DEFAULT_CLASSES,
    DEFAULT_CONF,
    DEFAULT_COPY_IMAGES,
    DEFAULT_DETR_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_FORMAT,
    DEFAULT_IMGSZ,
    DEFAULT_INPUT_ROOT,
    DEFAULT_IOU,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_ONNX_CONF,
    DEFAULT_ONNX_LABEL,
    DEFAULT_ONNX_MIN_RATIO,
    DEFAULT_ONNX_MODEL,
    DEFAULT_ONNX_NORMALIZE,
    DEFAULT_ONNX_SCORE_INDICES,
    DEFAULT_ONNX_TRANSPOSE,
    DEFAULT_SAM_CONF,
    DEFAULT_SAM_LABEL,
    DEFAULT_SAM_MIN_RATIO,
    DEFAULT_SAM_MODEL,
    DEFAULT_SAM_PROMPT,
    DEFAULT_SOURCE,
    DEFAULT_START_BATCH,
    DEFAULT_END_BATCH,
    DEFAULT_KEEP_MIN_RATIO,
    DEFAULT_YOLO_MODEL,
    DEFAULT_SAM3_MODEL,
    DEFAULT_VERBOSE,
)
from tools.annotate.ops import DEFAULT_MOSAIC_BLOCK


def build_parser() -> argparse.ArgumentParser:
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
        "--output", default=None,
        help="输出目录；未指定时自动生成时间戳子目录。",
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
        "--no-blackout", dest="no_blackout", action="store_true", default=False,
        help="只检测、不对小框打码：比例阈值视为 0，保留全部框供后续合并；"
             "配合 --apply 写出未打码的图片与完整标注。用于「检测→合并→打码」第一步。",
    )
    parser.add_argument(
        "--mosaic-existing", dest="mosaic_existing", action="store_true", default=False,
        help="对已有 LabelMe 标注的小框打码并删除（--source 指向已标注目录，原地写回）；"
             "配合「--no-blackout 检测 → merge.py 合并」即「检测→合并→打码」最后一步。",
    )
    parser.add_argument(
        "--mosaic-min-ratio", type=float, default=None,
        help="--mosaic-existing 的小框面积占比阈值；默认与 --onnx-min-ratio 相同。",
    )
    parser.add_argument(
        "--mosaic-labels",
        default=None,
        help="--mosaic-existing 时只对这些类别的框执行小框打码，逗号分隔（如 'face,hand'）。"
             "留空则对所有类别生效。",
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
    # 默认 dry_run=True（仅统计预览），加 --apply 才写盘；与上面 help 文案一致。
    # 若写成 False，则不加 --apply 也会直接写盘，与"默认仅统计预览"预期相反。
    parser.set_defaults(dry_run=True)
    return parser
