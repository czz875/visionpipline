"""
tools/merge/inherit_dataset.py
把 ``autolabel`` 这种"平铺"数据集，按 1000 张/批接续到 ``behavior`` 目录中。

行为：
- 扫描 ``behavior`` 现有 4 位数字 batch，找最大编号 + 1 作为起点；
- 把 ``autolabel`` 里所有 (PNG+JSON) 对按 1000/批切分到新 batch；
- 每个新 batch 内部按 JSON 内 label 归类到 behavior 现有 8 个子目录；
- 默认 **复制**（不动 autolabel 原文件），加 ``--move`` 才改为移动；
- 默认 ``--dry-run``，加 ``--apply`` 才会真复制。

归类规则（与 behavior/0022 现有分布对齐）：

    仅 face            -> only_head
    仅 hand            -> only_hand
    仅 cigarette       -> has_cigarette
    含 phone           -> has_phone
    face 数量 >= 2     -> multi_face
    hand 数量 >= 2     -> multi_hand
    face + hand 混合   -> multi_hand
    类别数 >= 3        -> multi_label
    无 shape           -> other

典型用法：

    .conda\\python.exe tools\\merge\\inherit_dataset.py ^
        --source datasets\\autolabel ^
        --target datasets\\behavior ^
        --batch-size 1000

    .conda\\python.exe tools\\merge\\inherit_dataset.py ^
        --source datasets\\autolabel ^
        --target datasets\\behavior ^
        --batch-size 1000 ^
        --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_SOURCE_DIR = r"datasets\autolabel"
DEFAULT_TARGET_DIR = r"datasets\behavior"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_DRY_RUN = True
DEFAULT_MOVE = False

# behavior 现有 8 个分类子目录（顺序无业务含义，仅用于建目录时排序）
CATEGORY_DIRS: tuple[str, ...] = (
    "has_cigarette",
    "has_phone",
    "multi_face",
    "multi_hand",
    "multi_label",
    "only_hand",
    "only_head",
    "other",
)

# 匹配 4 位数字 batch 目录名
BATCH_DIR_RE = re.compile(r"^\d{4}$")


# =============================================================================
# 2. 数据类
# =============================================================================


@dataclass(frozen=True)
class Pair:
    """一个 (PNG, JSON) 配对。"""
    image: Path
    json: Path


# =============================================================================
# 3. 核心逻辑
# =============================================================================


def discover_max_batch(target_dir: Path) -> int:
    """扫描 ``target_dir`` 下所有 4 位数字 batch 目录，返回最大编号。"""
    if not target_dir.is_dir():
        return 0
    max_idx = 0
    for d in target_dir.iterdir():
        if d.is_dir() and BATCH_DIR_RE.match(d.name):
            max_idx = max(max_idx, int(d.name))
    return max_idx


def collect_pairs(source_dir: Path) -> list[Pair]:
    """扫描 ``source_dir`` 下所有 PNG+JSON 配对（按文件名排序）。"""
    pngs = sorted(source_dir.glob("*.png"))
    pairs: list[Pair] = []
    for png in pngs:
        json_path = png.with_suffix(".json")
        if not json_path.exists():
            # PNG 没有配对 JSON 的情况：单独打包
            pairs.append(Pair(image=png, json=json_path))
            continue
        pairs.append(Pair(image=png, json=json_path))
    return pairs


def read_labels(json_path: Path) -> list[str]:
    """读取 JSON 里的所有 label（已去重保序）。失败时返回空列表。"""
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    seen: set[str] = set()
    labels: list[str] = []
    for shape in data.get("shapes", []):
        lbl = shape.get("label", "")
        if lbl and lbl not in seen:
            seen.add(lbl)
            labels.append(lbl)
    return labels


def classify(labels: list[str]) -> str:
    """根据 JSON 内的 label 列表，决定该图应归到哪个子目录。

    归类规则见模块 docstring。
    """
    if not labels:
        return "other"
    label_set = set(labels)
    counts = Counter(labels)

    if "phone" in label_set:
        return "has_phone"
    if label_set == {"cigarette"}:
        return "has_cigarette"
    if label_set == {"face"}:
        return "only_head"
    if label_set == {"hand"}:
        return "only_hand"
    if counts.get("face", 0) >= 2:
        return "multi_face"
    if counts.get("hand", 0) >= 2:
        return "multi_hand"
    if "face" in label_set and "hand" in label_set:
        return "multi_hand"
    if len(label_set) >= 3:
        return "multi_label"
    return "other"


def chunk_pairs(pairs: list[Pair], batch_size: int) -> list[list[Pair]]:
    """把 ``pairs`` 切成 ``batch_size`` 一组的批。"""
    return [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]


def inherit_dataset(
    source_dir: Path,
    target_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = DEFAULT_DRY_RUN,
    move: bool = DEFAULT_MOVE,
) -> list[dict]:
    """执行接续，返回每个新 batch 的计划报告。"""
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在：{source_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"目标目录不存在：{target_dir}")

    pairs = collect_pairs(source_dir)
    if not pairs:
        print(f"[警告] 源目录 {source_dir} 下没有 PNG 文件。")
        return []

    start_idx = discover_max_batch(target_dir) + 1
    batches = chunk_pairs(pairs, batch_size)

    reports: list[dict] = []
    # 第一段进度：扫描 + 分类（dry-run 和 apply 都会跑，因为 classify 要读 JSON）
    scan_pbar = tqdm(
        pairs,
        desc="扫描分类",
        unit="file",
        disable=len(pairs) == 0,
    )

    for offset, batch in enumerate(batches):
        batch_name = f"{start_idx + offset:04d}"
        batch_dir = target_dir / batch_name
        category_counter: Counter[str] = Counter()
        plan_items: list[tuple[Pair, Path, Path, str]] = []
        for pair in batch:
            scan_pbar.update(1)
            labels = read_labels(pair.json)
            category = classify(labels)
            dst_dir = batch_dir / category
            dst_image = dst_dir / pair.image.name
            dst_json = dst_dir / pair.json.name
            plan_items.append((pair, dst_image, dst_json, category))
            category_counter[category] += 1

        if not dry_run:
            batch_dir.mkdir(parents=True, exist_ok=True)
            for cat in CATEGORY_DIRS:
                (batch_dir / cat).mkdir(parents=True, exist_ok=True)
            # 第二段进度：实际复制 / 移动
            op_name = "移动" if move else "复制"
            for pair, dst_image, dst_json, _cat in tqdm(
                plan_items,
                desc=f"{op_name} {batch_name}",
                unit="file",
                leave=False,
            ):
                if pair.image.exists():
                    if move:
                        shutil.move(str(pair.image), str(dst_image))
                    else:
                        shutil.copy2(pair.image, dst_image)
                if pair.json.exists():
                    if move:
                        shutil.move(str(pair.json), str(dst_json))
                    else:
                        shutil.copy2(pair.json, dst_json)

        reports.append({
            "batch": batch_name,
            "dst_dir": str(batch_dir),
            "total": len(plan_items),
            "by_category": dict(category_counter),
        })

    scan_pbar.close()
    return reports


# =============================================================================
# 4. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "把平铺数据集接续到按 batch 组织的目录中。"
            "按 1000 张/批切分到新 batch 并按 JSON label 归类到子目录。"
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(DEFAULT_SOURCE_DIR),
        help=f"平铺源目录（默认：{DEFAULT_SOURCE_DIR}）。",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(DEFAULT_TARGET_DIR),
        help=f"按 batch 组织的目标目录（默认：{DEFAULT_TARGET_DIR}）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"每个新 batch 的图片数（默认 {DEFAULT_BATCH_SIZE}）。",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="移动而不是复制（默认复制，autolabel 原文件保留）。",
    )
    parser.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="真正执行（默认是 dry-run 预览）。",
    )
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        reports = inherit_dataset(
            args.source,
            args.target,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            move=args.move,
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"[错误] {e}")
        return 1

    if not reports:
        print("[完成] 没有需要接续的数据。")
        return 0

    mode = "预览" if args.dry_run else ("已移动" if args.move else "已复制")
    print(f"[{mode}] 共生成 {len(reports)} 个新 batch：")
    for r in reports:
        print(f"  {r['batch']} -> {r['dst_dir']}（共 {r['total']} 张）")
        for cat, n in sorted(r["by_category"].items(), key=lambda kv: -kv[1]):
            print(f"      {cat:<16} {n}")

    if args.dry_run:
        print("\n（这是 dry-run 预览，加上 --apply 才会真正执行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
