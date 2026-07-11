"""
tools/convert/labelme_to_yolo.py
把 LabelMe 标注批量转换为 YOLO 格式，支持多 batch 划分与增量更新。

实现要点（按 AGENTS.md §4.3 优先调用官方库）：
- 标签解析与归一化：``tools.core.labelme.labelme_dict_to_detections``（sv.Detections）；
- 写出 YOLO txt：``supervision.dataset.formats.yolo.detections_to_yolo_annotations``；
- 写 data.yaml：``supervision.dataset.formats.yolo.save_data_yaml``；
- 自身只负责：batch 发现、train/val 划分、图片复制、增量跳过。

典型用法：

    .conda\python.exe tools\convert\labelme_to_yolo.py ^
        --src datasets/behavior ^
        --out datasets/yolo

    .conda\python.exe tools\convert\labelme_to_yolo.py --help
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from supervision.dataset.formats.yolo import (
    detections_to_yolo_annotations,
    save_data_yaml,
)

from tools.core import labelme_dict_to_detections, load_labelme


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SRC_DIR = DEFAULT_PROJECT_ROOT / "datasets" / "behavior"
DEFAULT_OUT_DIR = DEFAULT_PROJECT_ROOT / "datasets" / "yolo"
DEFAULT_CLASS_NAMES: tuple[str, ...] = ("phone", "cigarette", "face", "hand")
DEFAULT_SPLITS: tuple[str, ...] = ("train", "val")
DEFAULT_RATIOS: tuple[float, ...] = (0.9, 0.1)
DEFAULT_SEED = 3407
DEFAULT_FORCE = False
DEFAULT_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")

# 批次子目录命名：4 位数字（如 0001/、0002/）。不在此格式内的子目录会被忽略。
BATCH_DIR_RE = re.compile(r"^\d{4}$")


# =============================================================================
# 2. batch 发现与划分
# =============================================================================


def collect_json_files(src_dir: Path) -> list[Path]:
    return sorted(src_dir.glob("*.json"))


def discover_batches(src_dir: Path) -> dict[str, list[Path]]:
    """自动发现批次结构。

    规则：src_dir 下的 4 位数字子目录（如 0001/、0002/）视为一个 batch，
    递归读取其下所有 .json；不在该格式的子目录会被忽略。
    若找不到任何 4 位数字子目录，则回退为平铺模式：把 src_dir 下所有 .json
    归到单一虚拟 batch "all"，保持与原脚本兼容。
    返回：{batch_name: [json_paths]}，按 batch 名称排序。
    """
    if not src_dir.is_dir():
        return {}
    batch_dirs = sorted(
        d for d in src_dir.iterdir()
        if d.is_dir() and BATCH_DIR_RE.match(d.name)
    )
    if not batch_dirs:
        return {"all": sorted(src_dir.glob("*.json"))}
    return {b.name: sorted(b.rglob("*.json")) for b in batch_dirs}


def split_files(
    files: list[Path],
    seed: int = DEFAULT_SEED,
    ratios: Iterable[float] = DEFAULT_RATIOS,
) -> list[list[Path]]:
    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    ratios = list(ratios)
    counts = [int(total * r) for r in ratios]
    counts[-1] = total - sum(counts[:-1])
    if counts[-1] < 0:
        counts[-1] = 0

    split_sets: list[list[Path]] = []
    start = 0
    for count in counts:
        end = start + count
        split_sets.append(shuffled[start:end])
        start = end
    return split_sets


# =============================================================================
# 3. 转换核心（LabelMe → YOLO，借 supervision 之力）
# =============================================================================


def find_image_path(
    json_path: Path,
    *,
    suffixes: Iterable[str] = DEFAULT_IMAGE_SUFFIXES,
) -> Path:
    for suffix in suffixes:
        image_path = json_path.with_suffix(suffix)
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Image not found for {json_path}")


def is_up_to_date(src: Path, dst: Path) -> bool:
    """判断 dst 是否已存在且不比 src 旧（即 src 未被修改过）。"""
    return dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime


def convert_labelme_to_yolo(
    json_path: Path,
    output_path: Path,
    *,
    class_names: list[str],
) -> None:
    """把单个 LabelMe JSON 转成 YOLO txt（核心走 supervision）。"""
    data = load_labelme(json_path)
    image_w = int(data.get("imageWidth", 1))
    image_h = int(data.get("imageHeight", 1))
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    detections = labelme_dict_to_detections(data, class_name_to_id=class_name_to_id)
    if len(detections) == 0:
        lines: list[str] = []
    else:
        # supervision 要求 image_shape = (H, W, C)
        lines = detections_to_yolo_annotations(
            detections, image_shape=(image_h, image_w, 3)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def copy_image_to_split(image_path: Path, split_name: str, image_dir: Path) -> Path:
    dest_path = image_dir / split_name / image_path.name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dest_path)
    return dest_path


def write_split_list(
    out_dir: Path, split_name: str, json_files: Iterable[Path]
) -> None:
    """写 <out_dir>/<split_name>.txt，路径相对 out_dir（YOLO 标准）。"""
    list_path = out_dir / f"{split_name}.txt"
    rel_paths: list[str] = []
    for json_path in json_files:
        image_path = find_image_path(json_path)
        rel_paths.append(f"images/{split_name}/{image_path.name}")
    list_path.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")


def write_data_yaml(out_dir: Path, class_names: list[str]) -> None:
    """写 <out_dir>/data.yaml（train/val 路径相对 out_dir）——走 supervision。"""
    save_data_yaml(str(out_dir / "data.yaml"), list(class_names))


# =============================================================================
# 4. 主流程
# =============================================================================


def process_batch(
    batch_name: str,
    json_files: list[Path],
    out_dir: Path,
    seed: int,
    class_names: list[str],
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    force: bool = DEFAULT_FORCE,
) -> None:
    """处理单个 batch：划分 + 转标签 + 复制图片 + 写 train/val 列表 + 写 data.yaml。

    若目标 label 与 image 均已存在且比源文件新，则默认跳过，以加速增量转换。
    可通过 force=True 强制全部重新转换。
    """
    label_dir = out_dir / "labels"
    image_dir = out_dir / "images"

    for split in splits:
        (label_dir / split).mkdir(parents=True, exist_ok=True)
        (image_dir / split).mkdir(parents=True, exist_ok=True)

    if not json_files:
        print(f"[{batch_name}] 没有 json，跳过")
        return

    split_sets = split_files(json_files, seed=seed)
    assert len(split_sets) == len(splits), "splits 数量与 ratios 数量必须一致"

    converted = 0
    skipped = 0
    for split_name, split_group in zip(splits, split_sets):
        for json_path in split_group:
            stem = json_path.stem
            image_path = find_image_path(json_path)
            label_path = label_dir / split_name / f"{stem}.txt"
            dest_image_path = image_dir / split_name / image_path.name

            if (
                not force
                and is_up_to_date(json_path, label_path)
                and is_up_to_date(image_path, dest_image_path)
            ):
                skipped += 1
                continue

            convert_labelme_to_yolo(json_path, label_path, class_names=class_names)
            copy_image_to_split(image_path, split_name, image_dir)
            converted += 1
        write_split_list(out_dir, split_name, split_group)

    write_data_yaml(out_dir, class_names)
    print(f"[{batch_name}] 共 {len(json_files)} 个 -> {out_dir}")
    print(f"  - 新转换: {converted} 个")
    print(f"  - 已存在跳过: {skipped} 个")
    for split_name, split_group in zip(splits, split_sets):
        print(f"  - {split_name}: {len(split_group)} images")


# =============================================================================
# 5. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert labelme JSON to YOLO format. Supports multi-batch input."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help=f"源数据集目录（默认：{DEFAULT_SRC_DIR}）。会自动识别 4 位数字子目录作为 batch。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"YOLO 输出根目录（默认：{DEFAULT_OUT_DIR}）。每个 batch 输出到 <out>/<batch>/。",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASS_NAMES),
        help=(
            "类别名（逗号分隔，顺序即 YOLO 类别 id）。"
            f"默认：{','.join(DEFAULT_CLASS_NAMES)}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子（默认 {DEFAULT_SEED}）。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新转换所有文件，不跳过已存在的输出。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    src = args.src.resolve()
    out_root = args.out.resolve()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]

    if not src.is_dir():
        print(f"[错误] 源目录不存在：{src}")
        return 1

    batches = discover_batches(src)
    if not batches:
        print(f"[错误] 在 {src} 下未找到任何 JSON 标注文件。")
        return 1

    print(f"源目录    ：{src}")
    print(f"输出根目录：{out_root}")
    print(f"类别      ：{class_names}")
    print(f"检测到 {len(batches)} 个 batch：{', '.join(batches.keys())}")

    for batch_idx, (batch_name, json_files) in enumerate(batches.items()):
        batch_out = out_root / batch_name
        # "all" 虚拟 batch 用 SEED；其他 batch 用 SEED + idx 偏移
        seed = DEFAULT_SEED if batch_name == "all" else DEFAULT_SEED + batch_idx
        process_batch(
            batch_name,
            json_files,
            batch_out,
            seed=seed,
            class_names=class_names,
            force=args.force,
        )

    # 顶层写一个 names.txt 方便查看
    (out_root / "names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
