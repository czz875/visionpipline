"""
tools/label/align_labelme.py
把目录里 PNG + LabelMe JSON 按 JSON 内 ``imagePath`` 字段对齐：
JSON 文件名 ↔ PNG 文件名（basename 一致），并同步 ``imagePath`` 字段。

典型场景：
- 旧版 ``timestamp_rename.py --labelme-sync`` 只改 ``imagePath`` 不改 JSON
  文件名，导致 PNG 和 JSON 名字错位；用本脚本一次性补救对齐。
- ``inherit_dataset`` 后 autolabel 阶段的 JSON 复制时间和 PNG 原始拍照
  时间不同，再经过 ``rename`` 后也会出现 PNG/JSON 错位。

典型用法：

    # 预览（默认 dry-run）
    .conda\python.exe tools\label\align_labelme.py --root datasets\behavior

    # 真正执行
    .conda\python.exe tools\label\align_labelme.py --root datasets\behavior --apply
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

DEFAULT_ROOT = Path("datasets/behavior")
DEFAULT_RECURSIVE = True
DEFAULT_DRY_RUN = True

IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")


# =============================================================================
# 2. 对齐逻辑
# =============================================================================


def collect_files(
    root: Path,
    *,
    recursive: bool,
) -> tuple[list[Path], list[Path]]:
    """收集目录下所有 PNG 和 JSON（按需递归）。"""
    if recursive:
        pngs = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        jsons = [p for p in root.rglob("*.json") if p.is_file()]
    else:
        pngs = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        jsons = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    return sorted(pngs), sorted(jsons)


def _find_matching_png(
    json_path: Path,
    png_index: dict[str, list[Path]],
) -> Path | None:
    """根据 JSON 的 ``imagePath`` 字段，找匹配的 PNG。"""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    image_path_str = data.get("imagePath", "")
    if not image_path_str:
        return None

    # 优先：JSON 同目录下、basename 与 imagePath 一致的 PNG
    image_basename = Path(image_path_str).name
    same_dir = json_path.parent / image_basename
    if same_dir.is_file():
        return same_dir

    # 兜底：从索引里查（按 imagePath basename 找）
    candidates = png_index.get(image_basename, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for c in candidates:
            if c.parent == json_path.parent:
                return c
        return candidates[0]
    return None


def plan_alignments(
    pngs: list[Path],
    jsons: list[Path],
) -> list[tuple[Path, Path, Path, str]]:
    """规划对齐计划。

    返回 ``[(json_old, json_new, png_matched, new_image_name), ...]``。
    """
    png_index: dict[str, list[Path]] = {}
    for p in pngs:
        png_index.setdefault(p.name, []).append(p)

    plans: list[tuple[Path, Path, Path, str]] = []
    for j in jsons:
        matched_png = _find_matching_png(j, png_index)
        if matched_png is None:
            continue
        target_json = matched_png.with_suffix(".json")
        if target_json == j:
            continue
        plans.append((j, target_json, matched_png, matched_png.name))
    return plans


def _resolve_json_collision(target: Path) -> Path:
    """撞名兜底：在 stem 后加 ``__dupN`` 直到不撞。"""
    base = target.stem
    suffix = target.suffix
    n = 1
    while True:
        cand = target.parent / f"{base}__dup{n}{suffix}"
        if not cand.exists():
            return cand
        n += 1


def apply_alignments(plans: list[tuple[Path, Path, Path, str]]) -> None:
    """真正执行对齐：同步 imagePath + 改 JSON 文件名。"""
    for json_old, json_new, _png, new_image_name in tqdm(
        plans, desc="对齐 JSON", unit="file", leave=False,
    ):
        # 1) 同步 imagePath 字段
        try:
            data = json.loads(json_old.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        if data is not None and data.get("imagePath") != new_image_name:
            data["imagePath"] = new_image_name
            try:
                json_old.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                pass

        # 2) 改 JSON 文件名（撞名兜底）
        if json_new.exists():
            json_new = _resolve_json_collision(json_new)
        try:
            json_old.rename(json_new)
        except OSError:
            pass


def align_labelme(
    root: Path,
    *,
    recursive: bool = DEFAULT_RECURSIVE,
    dry_run: bool = DEFAULT_DRY_RUN,
) -> list[tuple[Path, Path, Path, str]]:
    """执行对齐，返回对齐计划。"""
    pngs, jsons = collect_files(root, recursive=recursive)
    plans = plan_alignments(pngs, jsons)
    if not dry_run:
        apply_alignments(plans)
    return plans


# =============================================================================
# 3. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把目录里 PNG + LabelMe JSON 按 JSON 内 imagePath 字段对齐。",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"要处理的根目录（默认 {DEFAULT_ROOT}）。",
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
        help="真正执行对齐（默认 dry-run 预览）。",
    )
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"[错误] 根目录不存在：{args.root}")
        return 1

    plans = align_labelme(
        args.root,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )

    mode = "预览" if args.dry_run else "已执行"
    print(f"[{mode}] {len(plans)} 个 JSON 将对齐到同名 PNG")
    for json_old, json_new, png, _ in plans:
        print(f"  {json_old.name}  ->  {json_new.name}  (匹配 {png.name})")
    if args.dry_run:
        print("\n（这是 dry-run 预览，加 --apply 才会真改）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
