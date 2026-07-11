"""
tools/rename/timestamp_rename.py
按时间戳顺序，把目录下的文件重命名为 ``YYYYMMDD_HHMMSS_NNNNNN`` 格式。

行为：
- 按 ``mtime`` 升序排序（可改 ``ctime``）；
- 文件名格式为 ``YYYYMMDD_HHMMSS_序号``，例如 ``20260710_143052_000001.png``；
- 保留原文件扩展名（多扩展名如 ``.tar.gz`` 保留最后一段）；
- 默认递归子目录；
- 默认 ``--dry-run``，需要显式 ``--apply`` 才会真正改名。

典型用法：

    # 预览
    .conda\python.exe tools\rename\timestamp_rename.py ^
        --source-dir D:\photos

    # 真实执行（递归）
    .conda\python.exe tools\rename\timestamp_rename.py ^
        --source-dir D:\photos ^
        --apply

    # 只处理顶层 + 用 ctime
    .conda\python.exe tools\rename\timestamp_rename.py ^
        --source-dir D:\photos ^
        --time-source ctime ^
        --no-recursive ^
        --apply

    # 同时同步 LabelMe JSON 的 imagePath（推荐对 LabelMe 数据集使用）
    .conda\python.exe tools\rename\timestamp_rename.py ^
        --source-dir datasets\behavior ^
        --labelme-sync ^
        --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_SOURCE_DIR = r"D:\photos"
DEFAULT_TIME_SOURCE = "mtime"           # mtime / ctime
DEFAULT_RECURSIVE = True
DEFAULT_DRY_RUN = True
DEFAULT_SEQ_WIDTH = 6                   # 序号位数：6 -> 000001
DEFAULT_LABELME_SYNC = False            # 改名时是否同步同目录 LabelMe JSON 的 imagePath


# =============================================================================
# 2. 工具函数
# =============================================================================


def _is_multi_ext(path: Path) -> str:
    """拿到需要保留的扩展名（最后一段，如 ``.png`` / ``.gz``）。"""
    return path.suffix  # 简单场景只取最后一段；多段扩展名（.tar.gz）也只留 .gz


def _timestamp_to_str(ts: float) -> str:
    """把 epoch 秒转成 ``YYYYMMDD_HHMMSS`` 字符串（按本地时区）。"""
    from datetime import datetime

    # 不传 tz 时，fromtimestamp 走系统本地时区
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y%m%d_%H%M%S")


def _collect_files(
    source_dir: Path,
    *,
    recursive: bool,
) -> list[Path]:
    """收集目录下所有文件（不包括子目录），按时间戳升序排序。"""
    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在：{source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"源路径不是目录：{source_dir}")

    if recursive:
        files = [p for p in source_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in source_dir.iterdir() if p.is_file()]

    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def _resolve_collision(
    target_dir: Path,
    stem: str,
    suffix: str,
    *,
    seq_width: int = DEFAULT_SEQ_WIDTH,
) -> Path:
    """找一个不会撞名的目标路径，序号递增。

    stem 形如 ``YYYYMMDD_HHMMSS_NNNNNN``，碰撞时继续增大序号。
    """
    # stem 末尾为 _NNNNNN，解析出基础部分和当前序号
    base_stem = stem[: -(seq_width + 1)]
    seq = int(stem[-seq_width:]) + 1
    while True:
        candidate = target_dir / f"{base_stem}_{seq:0{seq_width}d}{suffix}"
        if not candidate.exists():
            return candidate
        seq += 1


def _plan_labelme_sync(
    old_image: Path,
    new_image: Path,
) -> tuple[Path, Path] | None:
    """计算 LabelMe JSON 同步计划。

    只处理 PNG/JPG 的同名 JSON；JSON 不参与图片的 mtime 排序。
    返回 ``(json_old, json_new)``；找不到同名 JSON 时返回 ``None``。
    """
    if old_image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return None
    json_path = old_image.with_suffix(".json")
    if not json_path.exists():
        return None
    new_json_path = new_image.with_suffix(".json")
    if json_path == new_json_path:
        return None
    return (json_path, new_json_path)


def _apply_labelme_sync(
    old_image: Path,
    new_image: Path,
) -> None:
    """把 ``old_image`` 同目录同名 JSON 跟着 PNG 一起改名 + 同步 imagePath。

    找不到同名 JSON / JSON 解析失败时静默跳过（视为该 PNG 没有配对标注）。
    撞名时在 JSON 文件名后加 ``__dupN`` 兜底。
    """
    plan = _plan_labelme_sync(old_image, new_image)
    if plan is None:
        return
    json_path, new_json_path = plan
    new_image_name = new_image.name

    # 1) 同步 imagePath 字段
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = None
    if data is not None and data.get("imagePath") != new_image_name:
        data["imagePath"] = new_image_name
        try:
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    # 2) 改名 JSON 文件（撞名兜底）
    if new_json_path.exists():
        base = new_json_path.stem
        suffix = new_json_path.suffix
        n = 1
        while True:
            cand = new_json_path.parent / f"{base}__dup{n}{suffix}"
            if not cand.exists():
                new_json_path = cand
                break
            n += 1
    try:
        json_path.rename(new_json_path)
    except OSError:
        pass


def rename_by_timestamp(
    source_dir: Path,
    *,
    time_source: str = DEFAULT_TIME_SOURCE,
    recursive: bool = DEFAULT_RECURSIVE,
    dry_run: bool = DEFAULT_DRY_RUN,
    seq_width: int = DEFAULT_SEQ_WIDTH,
    labelme_sync: bool = DEFAULT_LABELME_SYNC,
) -> list[tuple[Path, Path]]:
    """执行批量重命名。返回 ``(old, new)`` 计划列表（dry-run 时也返回，但不会写盘）。

    **只对 PNG / JPG 排序改名**——LabelMe JSON 不参与 mtime 排序，否则
    JSON 复制时间或 auto-annotate 生成时间会和 PNG 原始拍照时间错位，
    导致 PNG 和 JSON 改名为不同序号、不同时间戳，配对关系彻底乱掉。

    当 ``labelme_sync=True`` 时，会让每个 PNG/JPG 的**当前同名** LabelMe
    JSON 一起改名为 ``<新图名>.json``，并同步其 ``imagePath`` 字段。
    """
    files = _collect_files(source_dir, recursive=recursive)

    attr = "st_mtime" if time_source == "mtime" else "st_ctime"

    image_suffixes = {".png", ".jpg", ".jpeg"}
    image_files = [p for p in files if p.suffix.lower() in image_suffixes]
    image_files.sort(key=lambda p: getattr(p.stat(), attr))

    plans: list[tuple[Path, Path]] = []
    for seq, src in enumerate(tqdm(image_files, desc="规划改名", unit="file", leave=False), start=1):
        ts = getattr(src.stat(), attr)
        stem = _timestamp_to_str(ts)
        suffix = _is_multi_ext(src)
        full_stem = f"{stem}_{seq:0{seq_width}d}"
        target = src.parent / f"{full_stem}{suffix}"
        if target == src:
            continue
        if target.exists():
            target = _resolve_collision(src.parent, full_stem, suffix, seq_width=seq_width)
        plans.append((src, target))

    if dry_run:
        return plans

    op_label = "改名" + (" + 同步 JSON" if labelme_sync else "")
    for old, new in tqdm(plans, desc=op_label, unit="file"):
        old.rename(new)
        if labelme_sync:
            _apply_labelme_sync(old, new)
    return plans


# =============================================================================
# 3. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按时间戳顺序，把目录下的文件批量重命名为 YYYYMMDD_HHMMSS_NNNNNN 格式。",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(DEFAULT_SOURCE_DIR),
        help=f"要处理的源目录（默认：{DEFAULT_SOURCE_DIR}）。",
    )
    parser.add_argument(
        "--time-source",
        type=str,
        choices=("mtime", "ctime"),
        default=DEFAULT_TIME_SOURCE,
        help=f"排序所用的时间戳来源（默认：{DEFAULT_TIME_SOURCE}）。",
    )
    parser.add_argument(
        "--seq-width",
        type=int,
        default=DEFAULT_SEQ_WIDTH,
        help=f"序号位数（默认 {DEFAULT_SEQ_WIDTH}）。",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RECURSIVE,
        help="是否递归子目录（默认开）。",
    )
    parser.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="真正执行重命名（默认是 dry-run 预览）。",
    )
    parser.add_argument(
        "--labelme-sync",
        action="store_true",
        default=DEFAULT_LABELME_SYNC,
        help=(
            "PNG/JPG 改名后，同步同名 LabelMe JSON 的 imagePath 字段。"
            "对 LabelMe 数据集强烈建议开启。"
        ),
    )
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        plans = rename_by_timestamp(
            args.source_dir,
            time_source=args.time_source,
            recursive=args.recursive,
            dry_run=args.dry_run,
            seq_width=args.seq_width,
            labelme_sync=args.labelme_sync,
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"[错误] {e}")
        return 1

    mode = "预览" if args.dry_run else "已执行"
    sync_note = "（含 LabelMe imagePath 同步）" if args.labelme_sync else ""
    print(f"[{mode}] {len(plans)} 个文件将被改名（按 {args.time_source} 升序）{sync_note}")
    for old, new in plans:
        print(f"  {old.name}  ->  {new.name}")
    if args.dry_run:
        print("\n（这是 dry-run 预览，加上 --apply 才会真正改名）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
