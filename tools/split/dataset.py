"""
按给定比例把 LabelMe 数据集随机拆分为两部分（A + B）。

常用于“随机清洗选取高质量数据 30%，剩余 70% 进入下一轮训练/自标注”的流水线。
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import (
    build_timestamped_output_dir,
    find_json_for_image,
    list_images,
)

DEFAULT_OUTPUT_DIR = Path("datasets/split")


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="按图片数量比例把 LabelMe 数据集随机拆分为 A、B 两份。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入的 LabelMe 数据集目录。",
    )
    parser.add_argument(
        "--output-a",
        type=Path,
        default=None,
        help="输出目录 A；未指定时自动生成时间戳子目录。",
    )
    parser.add_argument(
        "--output-b",
        type=Path,
        default=None,
        help="输出目录 B；未指定时自动生成时间戳子目录。",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="A 份占总图片数量的比例（默认：%(default)s）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现（默认：%(default)s）。",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="是否复制图片与 JSON（默认复制）。",
    )
    return parser


def _copy_image_json(
    image_path: Path,
    json_path: Path | None,
    output_dir: Path,
) -> None:
    """把单张图片及其 LabelMe JSON（如存在）复制到输出目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_dir / image_path.name)
    if json_path is not None and json_path.exists():
        shutil.copy2(json_path, output_dir / json_path.name)


def split_dataset(
    input_dir: Path,
    output_a: Path,
    output_b: Path,
    ratio: float,
    seed: int,
    copy: bool = True,
) -> tuple[int, int]:
    """按图片比例拆分数据集。

    Args:
        input_dir: 输入 LabelMe 数据集目录。
        output_a: A 份输出目录。
        output_b: B 份输出目录。
        ratio: A 份图片数量比例。
        seed: 随机种子。
        copy: 是否复制文件。

    Returns:
        (A 份图片数, B 份图片数)。
    """
    images = list_images(input_dir)
    if not images:
        print(f"[警告] 在 {input_dir} 中未找到图片。")
        return 0, 0

    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)

    count_a = max(1, int(len(shuffled) * ratio)) if 0 < ratio < 1 else len(shuffled)
    selected = set(shuffled[:count_a])

    copied_a = copied_b = 0
    for image_path in shuffled:
        json_path = find_json_for_image(image_path)
        if image_path in selected:
            if copy:
                _copy_image_json(image_path, json_path, output_a)
            copied_a += 1
        else:
            if copy:
                _copy_image_json(image_path, json_path, output_b)
            copied_b += 1

    return copied_a, copied_b


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    if not (0 < args.ratio < 1):
        print("[错误] --ratio 必须在 (0, 1) 之间。")
        return 1

    output_a = args.output_a
    output_b = args.output_b
    if output_a is None or output_b is None:
        base = Path(DEFAULT_OUTPUT_DIR)
        base.mkdir(parents=True, exist_ok=True)
        if output_a is None:
            output_a = build_timestamped_output_dir(base, "split_a")
        if output_b is None:
            output_b = build_timestamped_output_dir(base, "split_b")
    else:
        output_a = Path(output_a)
        output_b = Path(output_b)
        output_a.mkdir(parents=True, exist_ok=True)
        output_b.mkdir(parents=True, exist_ok=True)

    count_a, count_b = split_dataset(
        args.input.resolve(),
        output_a.resolve(),
        output_b.resolve(),
        args.ratio,
        args.seed,
        args.copy,
    )

    print(f"A 份：{count_a} 张 -> {output_a}")
    print(f"B 份：{count_b} 张 -> {output_b}")
    print(f"OUTPUT_PATH:{output_a.resolve()}")
    print(f"OUTPUT_PATH_A:{output_a.resolve()}")
    print(f"OUTPUT_PATH_B:{output_b.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
