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
DEFAULT_RATIO = 0.3
DEFAULT_SEED = 42


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
        default=DEFAULT_RATIO,
        help="A 份占总图片数量的比例（默认：%(default)s）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="随机种子，保证可复现（默认：%(default)s）。",
    )
    parser.add_argument(
        "--copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否复制图片与 JSON（默认复制，可用 --no-copy 关闭）。",
    )
    return parser


def _copy_image_json(
    image_path: Path,
    json_path: Path | None,
    output_dir: Path,
    source_root: Path,
) -> None:
    """把单张图片及其 LabelMe JSON（如存在）复制到输出目录。

    按相对 ``source_root`` 的路径镜像子目录结构，避免不同子目录下的同名
    图片 / JSON 被平铺到同一层后互相覆盖。
    """
    rel_dir = image_path.parent.relative_to(source_root)
    out_dir = output_dir / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_dir / image_path.name)
    if json_path is not None and json_path.exists():
        shutil.copy2(json_path, out_dir / json_path.name)


def _collect_input_images(input_dir: Path, ratio: float) -> list[Path]:
    """在创建输出目录前校验输入，图片只扫描一次。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    if not 0 < ratio < 1:
        raise ValueError("ratio 必须在 (0, 1) 之间。")
    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"在 {input_dir} 中未找到图片。")
    return images


def _validate_output_paths(input_dir: Path, outputs: tuple[Path, ...]) -> None:
    """预检全部输出及其父路径，不创建目录、不删除旧数据。"""
    paths = (input_dir, *outputs)
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"输入与 A/B 输出目录不能重叠：{left}、{right}")
    for output in outputs:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"输出路径非空或不是目录，请指定新的目录：{output}")
        for parent in output.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"输出的父路径不是目录：{parent}")


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

    Raises:
        FileNotFoundError: 输入目录不存在或没有图片。
        ValueError: 比例非法，或写入目录重叠、非空。
    """
    input_dir = input_dir.resolve()
    images = _collect_input_images(input_dir, ratio)
    return _split_images(
        images, input_dir, output_a.resolve(), output_b.resolve(), ratio, seed, copy,
    )


def _split_images(
    images: list[Path], input_dir: Path, output_a: Path, output_b: Path,
    ratio: float, seed: int, copy: bool,
) -> tuple[int, int]:
    """对已扫描的图片分组，写入前统一检查输出范围。"""
    if copy:
        _validate_output_paths(input_dir, (output_a, output_b))
        output_a.mkdir(parents=True, exist_ok=True)
        output_b.mkdir(parents=True, exist_ok=True)

    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)

    count_a = max(1, int(len(shuffled) * ratio))
    selected = set(shuffled[:count_a])

    copied_a = copied_b = 0
    for image_path in shuffled:
        json_path = find_json_for_image(image_path)
        if image_path in selected:
            if copy:
                _copy_image_json(image_path, json_path, output_a, input_dir)
            copied_a += 1
        else:
            if copy:
                _copy_image_json(image_path, json_path, output_b, input_dir)
            copied_b += 1

    return copied_a, copied_b


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        input_dir = args.input.resolve()
        images = _collect_input_images(input_dir, args.ratio)
        output_a = args.output_a.resolve() if args.output_a is not None else None
        output_b = args.output_b.resolve() if args.output_b is not None else None
        if args.copy:
            provided = tuple(p for p in (output_a, output_b) if p is not None)
            _validate_output_paths(input_dir, provided)
            if output_a is None or output_b is None:
                base = Path(DEFAULT_OUTPUT_DIR).resolve()
                for root in (input_dir, *provided):
                    if root == base or root in base.parents:
                        raise ValueError(f"默认输出目录与输入或另一输出重叠：{base}、{root}")
        if output_a is None:
            output_a = build_timestamped_output_dir(DEFAULT_OUTPUT_DIR, "split_a").resolve()
        if output_b is None:
            output_b = build_timestamped_output_dir(DEFAULT_OUTPUT_DIR, "split_b").resolve()
        count_a, count_b = _split_images(
            images, input_dir, output_a, output_b, args.ratio, args.seed, args.copy,
        )
    except (OSError, ValueError) as exc:
        print(f"[错误] {exc}")
        return 1

    print(f"A 份：{count_a} 张 -> {output_a}")
    print(f"B 份：{count_b} 张 -> {output_b}")
    print(f"OUTPUT_PATH:{output_a.resolve()}")
    print(f"OUTPUT_PATH_A:{output_a.resolve()}")
    print(f"OUTPUT_PATH_B:{output_b.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
