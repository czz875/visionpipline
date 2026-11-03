r"""
tools/convert/labelme_to_yolo.py
把 LabelMe 标注批量转换为 YOLO 格式，支持多 batch 划分与增量更新。

实现要点（按 AGENTS.md §4.3 优先调用官方库）：
- 标签解析与归一化：``tools.core.labelme.labelme_dict_to_detections``（sv.Detections）；
- 写出 YOLO txt：``supervision.dataset.formats.yolo.detections_to_yolo_annotations``；
- 写 data.yaml：本项目自行写出 ``path`` / ``train`` / ``val`` / ``nc`` / ``names``
  （supervision 0.29.1 的 ``save_data_yaml`` 只写 ``nc`` / ``names``，缺
  ``train`` / ``val`` 时 ultralytics 无法加载数据集）；
- 自身只负责：batch 发现、train/val 划分、图片复制或硬链接、增量跳过。

典型用法：

    .conda\python.exe tools\convert\labelme_to_yolo.py ^
        --src datasets/behavior ^
        --out datasets/yolo

    # 输出图片改用硬链接以节省空间（源和输出必须位于同一卷）
    .conda\python.exe tools\convert\labelme_to_yolo.py ^
        --src datasets/behavior ^
        --out datasets/yolo ^
        --hardlink-images

    .conda\python.exe tools\convert\labelme_to_yolo.py --help
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from uuid import uuid4
from pathlib import Path
from typing import Iterable

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image
from supervision.dataset.formats.yolo import detections_to_yolo_annotations

from tools.core import (
    IMAGE_EXTENSIONS,
    YOLO_CLASS_NAMES,
    labelme_dict_to_detections,
    load_labelme,
    unregistered_labels,
)


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SRC_DIR = DEFAULT_PROJECT_ROOT / "datasets" / "behavior"
DEFAULT_OUT_DIR = DEFAULT_PROJECT_ROOT / "datasets" / "yolo"
# 类别表取自 tools.core.constants.YOLO_CLASS_NAMES（唯一权威来源），
# 与磁盘上 datasets/yolo/*/data.yaml 保持一致；顺序即 class id。
DEFAULT_CLASS_NAMES: tuple[str, ...] = YOLO_CLASS_NAMES
DEFAULT_SPLITS: tuple[str, ...] = ("train", "val")
DEFAULT_RATIOS: tuple[float, ...] = (0.9, 0.1)
DEFAULT_SEED = 3407
DEFAULT_FORCE = False
DEFAULT_HARDLINK_IMAGES = False
DEFAULT_IMAGE_SUFFIXES: tuple[str, ...] = tuple(sorted(IMAGE_EXTENSIONS))

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
    """按 ``ratios`` 把文件随机划分为若干组，``seed`` 固定则结果可复现。"""
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


def class_table_signature(class_names: Iterable[str]) -> str:
    """返回类别表的稳定指纹，用于判断标签是否可用当前类别表解释。

    类别名顺序即 YOLO class id，因此顺序变化也必须视为「类别表变了」。
    """
    return ",".join(class_names)


def read_class_table_signature(out_dir: Path) -> str | None:
    """从 ``<out_dir>/data.yaml`` 读回上次转换用的类别表指纹。

    Returns:
        指纹字符串；data.yaml 不存在、无 names 或无法解析时返回 ``None``
        （此时按「类别表未知」处理，触发重转，避免沿用无法验证的旧标签）。
    """
    yaml_path = out_dir / "data.yaml"
    if not yaml_path.exists():
        return None
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("names:"):
            continue
        raw = line[len("names:"):].strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        names = [n.strip().strip("'\"") for n in raw.split(",") if n.strip()]
        return class_table_signature(names) if names else None
    return None


def resolve_image_size(data: dict, image_path: Path) -> tuple[int, int]:
    """取 LabelMe JSON 记录的图片尺寸；缺失或非法时回退到实际图片尺寸。

    LabelMe 偶尔写出没有 ``imageWidth`` / ``imageHeight`` 的 JSON。若直接按 1 处理，
    YOLO 归一化坐标会整体放大到图片之外，因此这里必须回退到真实尺寸。

    Raises:
        ValueError: JSON 与图片都拿不到有效尺寸。
    """
    try:
        width = int(data.get("imageWidth"))
        height = int(data.get("imageHeight"))
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return width, height
    try:
        with Image.open(image_path) as img:
            real_width, real_height = img.size
    except OSError as e:
        raise ValueError(f"无法确定图片尺寸（JSON 未记录且读取失败）：{image_path}") from e
    if real_width <= 0 or real_height <= 0:
        raise ValueError(f"图片尺寸非法：{image_path}")
    return real_width, real_height


def convert_labelme_to_yolo(
    json_path: Path,
    output_path: Path,
    *,
    class_names: list[str],
    image_path: Path | None = None,
) -> list[str]:
    """把单个 LabelMe JSON 转成 YOLO txt（核心走 supervision）。

    全部标注都未注册时**不写**标签文件：写空 txt 会让 ultralytics 判 ``ne=1``，
    该图仍进训练集并被当成「这里没有目标」，等于主动训练错误的负样本。
    原本就没有标注的图是真实负样本，照常写空文件。

    Args:
        json_path: LabelMe JSON 路径。
        output_path: 目标 YOLO txt 路径。
        class_names: 类别名列表，顺序即 YOLO 类别 id。
        image_path: 对应图片路径；未给出时按同目录同名规则查找，用于回退读取尺寸。

    Returns:
        该 JSON 中未注册、因此被丢弃的标签名列表（空列表表示全部保留）。
        调用方可用 ``output_path.exists()`` 判断是否真的写出了标签。
    """
    data = load_labelme(json_path)
    if image_path is None:
        image_path = find_image_path(json_path)
    image_w, image_h = resolve_image_size(data, image_path)
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    detections = labelme_dict_to_detections(
        data,
        class_name_to_id=class_name_to_id,
        image_size=(image_w, image_h),
    )
    if len(detections) == 0:
        lines: list[str] = []
    else:
        # supervision 要求 image_shape = (H, W, C)
        lines = detections_to_yolo_annotations(
            detections, image_shape=(image_h, image_w, 3)
        )

    dropped = unregistered_labels(data, class_name_to_id)
    had_shapes = any(
        shape.get("shape_type") == "rectangle" for shape in data.get("shapes", [])
    )
    if not lines and had_shapes:
        # 源图有标注但一条都没保留：不写文件，避免伪造负样本。
        output_path.unlink(missing_ok=True)
        return dropped

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return dropped


def copy_image_to_split(
    image_path: Path,
    split_name: str,
    image_dir: Path,
    *,
    hardlink: bool = DEFAULT_HARDLINK_IMAGES,
) -> Path:
    dest_path = image_dir / split_name / image_path.name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not hardlink:
        if dest_path.exists() and image_path.samefile(dest_path):
            temp_path = dest_path.with_name(f".{dest_path.name}.{uuid4().hex}.tmp")
            shutil.copy2(image_path, temp_path)
            os.replace(temp_path, dest_path)
            return dest_path
        shutil.copy2(image_path, dest_path)
        return dest_path

    if dest_path.exists() and image_path.samefile(dest_path):
        return dest_path

    temp_path = dest_path.with_name(f".{dest_path.name}.{uuid4().hex}.tmp")
    try:
        os.link(image_path, temp_path)
        os.replace(temp_path, dest_path)
    except OSError as e:
        temp_path.unlink(missing_ok=True)
        raise OSError(
            f"无法为图片创建硬链接：{image_path} -> {dest_path}。"
            f"请确认源目录和输出目录位于同一卷。原始错误：{e}"
        ) from e
    return dest_path


def write_split_list(
    out_dir: Path, split_name: str, image_names: Iterable[str]
) -> None:
    """写 <out_dir>/<split_name>.txt 图片列表。

    每行写成 ``./images/<split>/<name>``。ultralytics 的 ``get_img_files`` 只在行首为
    ``./`` 时才按列表文件所在目录解析，否则会当作相对当前工作目录的路径，导致
    ``No labels found``。
    """
    list_path = out_dir / f"{split_name}.txt"
    rel_paths = [f"./images/{split_name}/{name}" for name in image_names]
    list_path.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")


def write_data_yaml(out_dir: Path, class_names: list[str], splits: Iterable[str] = DEFAULT_SPLITS) -> None:
    """写 <out_dir>/data.yaml。

    ``path`` 用绝对路径（含中文/空格的仓库路径也能被 ultralytics 解析），
    ``train`` / ``val`` 指向同目录的 ``<split>.txt`` 列表文件。

    不复用 ``supervision.save_data_yaml``：它只写 ``nc`` / ``names``，缺少
    ``train`` / ``val`` 时 ``yolo detect train data=...`` 会直接报错。
    """
    splits = list(splits)
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        f"train: {splits[0]}.txt",
    ]
    if len(splits) > 1:
        lines.append(f"val: {splits[1]}.txt")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names: [" + ", ".join(class_names) + "]")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# 4. 主流程
# =============================================================================


def _prune_stale_split_files(
    label_dir: Path,
    image_dir: Path,
    split_name: str,
    keep_names: list[str],
) -> int:
    """删除该 split 目录中不属于本次划分的残留文件。

    增量重跑时全批次会重新 shuffle，某张图可能从 train 换到 val。旧副本若留在
    原目录，同一张图会同时出现在 ``images/train`` 和 ``images/val``，造成
    train/val 泄漏（下游按目录消费时尤其严重）。

    Returns:
        删除的文件数。
    """
    keep = {Path(name).stem for name in keep_names}
    removed = 0
    for path in (image_dir / split_name).glob("*"):
        if path.is_file() and path.stem not in keep:
            path.unlink()
            removed += 1
    for path in (label_dir / split_name).glob("*.txt"):
        if path.stem not in keep:
            path.unlink()
            removed += 1
    return removed


def process_batch(
    batch_name: str,
    json_files: list[Path],
    out_dir: Path,
    seed: int,
    class_names: list[str],
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    ratios: Iterable[float] = DEFAULT_RATIOS,
    force: bool = DEFAULT_FORCE,
    hardlink_images: bool = DEFAULT_HARDLINK_IMAGES,
) -> list[str]:
    """处理单个 batch：划分 + 转标签 + 输出图片 + 写 train/val 列表 + 写 data.yaml。

    若目标 label 与 image 均已存在且比源文件新，则默认跳过，以加速增量转换。
    可通过 force=True 强制全部重新转换。

    Returns:
        本 batch 中被丢弃的未注册标签名（去重后），供调用方汇总报告。
    """
    label_dir = out_dir / "labels"
    image_dir = out_dir / "images"

    for split in splits:
        (label_dir / split).mkdir(parents=True, exist_ok=True)
        (image_dir / split).mkdir(parents=True, exist_ok=True)

    if not json_files:
        print(f"[{batch_name}] 没有 json，跳过")
        return []

    # 增量跳过必须确认「旧标签能用当前类别表解释」。只比 mtime 会在换 --classes
    # 后沿用旧 class id，而 data.yaml 被改写成新类别表 —— 标签与类别表整体错位，
    # ultralytics 会把越界 id 的图判为 corrupt 整张丢弃，且不报错。
    expected_sig = class_table_signature(class_names)
    previous_sig = read_class_table_signature(out_dir)
    if force or previous_sig != expected_sig:
        if previous_sig is not None and previous_sig != expected_sig:
            print(
                f"[{batch_name}] 类别表已变更（{previous_sig} -> {expected_sig}），"
                f"忽略增量跳过，全部重转"
            )
        force = True

    split_sets = split_files(json_files, seed=seed, ratios=ratios)
    assert len(split_sets) == len(splits), "splits 数量与 ratios 数量必须一致"

    converted = 0
    skipped = 0
    dropped: dict[str, None] = {}
    unlabeled: list[str] = []
    for split_name, split_group in zip(splits, split_sets):
        image_names: list[str] = []
        for json_path in split_group:
            stem = json_path.stem
            image_path = find_image_path(json_path)
            label_path = label_dir / split_name / f"{stem}.txt"
            dest_image_path = image_dir / split_name / image_path.name

            if (
                not force
                and is_up_to_date(json_path, label_path)
                and is_up_to_date(image_path, dest_image_path)
                and (
                    not hardlink_images
                    or image_path.samefile(dest_image_path)
                )
            ):
                image_names.append(image_path.name)
                skipped += 1
                continue

            for label in convert_labelme_to_yolo(
                json_path, label_path, class_names=class_names, image_path=image_path
            ):
                dropped.setdefault(label, None)

            if not label_path.exists():
                # 标注全被丢弃：不复制图片、不进 train/val 列表，避免成为负样本。
                unlabeled.append(image_path.name)
                dest_image_path.unlink(missing_ok=True)
                continue

            image_names.append(image_path.name)
            copy_image_to_split(
                image_path,
                split_name,
                image_dir,
                hardlink=hardlink_images,
            )
            converted += 1
        _prune_stale_split_files(label_dir, image_dir, split_name, image_names)
        write_split_list(out_dir, split_name, image_names)

    write_data_yaml(out_dir, class_names, splits)
    print(f"[{batch_name}] 共 {len(json_files)} 个 -> {out_dir}")
    print(f"  - 新转换: {converted} 个")
    print(f"  - 已存在跳过: {skipped} 个")
    for split_name, split_group in zip(splits, split_sets):
        print(f"  - {split_name}: {len(split_group)} images")
    if dropped:
        print(
            f"  [警告] {len(dropped)} 个标签未在类别表中，已丢弃："
            f"{', '.join(dropped)}"
        )
    if unlabeled:
        print(
            f"  [警告] {len(unlabeled)} 张图的标注全部未注册，已排除出数据集"
            f"（避免当作负样本）：{', '.join(unlabeled[:5])}"
            f"{' ...' if len(unlabeled) > 5 else ''}"
        )
    return list(dropped)


def convert_to_yolo(
    src: Path,
    out: Path,
    *,
    class_names: list[str],
    seed: int = DEFAULT_SEED,
    ratios: Iterable[float] = DEFAULT_RATIOS,
    force: bool = DEFAULT_FORCE,
    hardlink_images: bool = DEFAULT_HARDLINK_IMAGES,
    dry_run: bool = False,
) -> None:
    """LabelMe → YOLO 的 Python API（与 CLI main() 行为一致）。

    ``hardlink_images=True`` 时以硬链接代替复制图片；默认保持复制行为。
    ``dry_run=True`` 时只 print 计划，不写盘。
    ``ratios`` 为各 split 的划分比例，顺序对应 ``DEFAULT_SPLITS``。
    """
    src = src.resolve()
    out_root = out.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"源目录不存在：{src}")

    batches = discover_batches(src)
    if not batches:
        raise FileNotFoundError(f"在 {src} 下未找到任何 JSON 标注文件。")
    # discover_batches 对存在的目录总会返回条目（平铺模式返回 {"all": [...]}），
    # 因此必须按实际 JSON 数量判断，否则空目录会一路走到「成功」却不产 data.yaml。
    total_json = sum(len(files) for files in batches.values())
    if total_json == 0:
        raise FileNotFoundError(
            f"在 {src} 下未找到任何 JSON 标注文件（检查目录结构与 4 位数字批次命名）。"
        )

    ratios = list(ratios)
    if len(ratios) != len(DEFAULT_SPLITS):
        raise ValueError(
            f"ratios 数量必须为 {len(DEFAULT_SPLITS)}（对应 {DEFAULT_SPLITS}），"
            f"当前为 {len(ratios)}：{ratios}"
        )

    print(f"源目录    ：{src}")
    print(f"输出根目录：{out_root}")
    print(f"类别      ：{class_names}")
    print(f"划分比例  ：{dict(zip(DEFAULT_SPLITS, ratios))}，seed={seed}")
    print(f"检测到 {len(batches)} 个 batch：{', '.join(batches.keys())}")

    if dry_run:
        return

    all_dropped: dict[str, None] = {}
    for batch_idx, (batch_name, json_files) in enumerate(batches.items()):
        batch_out = out_root / batch_name
        # 各 batch 用 seed + idx 偏移，保证不同 batch 的划分互不相同但可复现。
        batch_seed = seed + batch_idx
        dropped = process_batch(
            batch_name,
            json_files,
            batch_out,
            seed=batch_seed,
            class_names=class_names,
            ratios=ratios,
            force=force,
            hardlink_images=hardlink_images,
        )
        for label in dropped:
            all_dropped.setdefault(label, None)

    # 顶层写一个 names.txt 方便查看
    (out_root / "names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    if all_dropped:
        print(
            f"\n[警告] 以下 {len(all_dropped)} 个标签未在 --classes 中注册，"
            f"对应标注已被丢弃：{', '.join(all_dropped)}\n"
            f"       如属漏配，请把类别补进 --classes 后重跑。"
        )


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
        "--ratios",
        type=str,
        default=",".join(str(r) for r in DEFAULT_RATIOS),
        help=(
            "train/val 划分比例（逗号分隔，顺序对应 train,val）。"
            f"默认：{','.join(str(r) for r in DEFAULT_RATIOS)}"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新转换所有文件，不跳过已存在的输出。",
    )
    parser.add_argument(
        "--hardlink-images",
        action="store_true",
        default=DEFAULT_HARDLINK_IMAGES,
        help=(
            "为输出图片创建硬链接以节省空间（源和输出必须位于同一卷；"
            "修改任一链接的图片内容会影响另一处）。"
        ),
    )
    return parser


def parse_ratios(text: str) -> list[float]:
    """把 ``"0.9,0.1"`` 解析成 ``[0.9, 0.1]`` 并校验数量与总和。"""
    ratios = [float(x) for x in text.split(",") if x.strip()]
    if len(ratios) != len(DEFAULT_SPLITS):
        raise ValueError(
            f"--ratios 需要 {len(DEFAULT_SPLITS)} 个值（train,val），当前：{text}"
        )
    if any(r <= 0 for r in ratios):
        raise ValueError(f"--ratios 必须为正数，当前：{text}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"--ratios 之和必须为 1，当前：{text}（合计 {sum(ratios)}）")
    return ratios


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        print("[错误] --classes 不能为空。")
        return 1
    try:
        ratios = parse_ratios(args.ratios)
        convert_to_yolo(
            args.src,
            args.out,
            class_names=class_names,
            seed=args.seed,
            ratios=ratios,
            force=args.force,
            hardlink_images=args.hardlink_images,
        )
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"[错误] {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
