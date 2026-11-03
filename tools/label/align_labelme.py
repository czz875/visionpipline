r"""
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

from tools.core import IMAGE_EXTENSIONS


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_ROOT = Path("datasets/behavior")
DEFAULT_RECURSIVE = True
DEFAULT_DRY_RUN = True


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
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        jsons = [p for p in root.rglob("*.json") if p.is_file()]
    else:
        pngs = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        jsons = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    return sorted(pngs), sorted(jsons)


def _find_matching_png(
    json_path: Path,
    png_index: dict[str, list[Path]],
) -> Path | None:
    """根据 JSON 的 ``imagePath`` 字段，找匹配的 PNG。

    同名 PNG 多于一张且没有任何一张与 JSON 同目录时，**不做猜测**：猜错会把
    标注绑到另一张图上并连带改名，污染会一路带进训练集。这类情况返回 ``None``，
    由 ``plan_alignments`` 计入「多候选跳过」并上报，交给人工处理。
    """
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
        return None  # 多候选且无同目录命中：不猜
    return None


def plan_alignments(
    pngs: list[Path],
    jsons: list[Path],
) -> tuple[list[tuple[Path, Path, Path, str]], list[Path]]:
    """规划对齐计划。

    Returns:
        ``(plans, ambiguous)``。``plans`` 为
        ``[(json_old, json_new, png_matched, new_image_name), ...]``；
        ``ambiguous`` 是「同名 PNG 多张、无法确定归属」而跳过对齐的 JSON 列表。
    """
    png_index: dict[str, list[Path]] = {}
    for p in pngs:
        png_index.setdefault(p.name, []).append(p)

    plans: list[tuple[Path, Path, Path, str]] = []
    ambiguous: list[Path] = []
    for j in jsons:
        matched_png = _find_matching_png(j, png_index)
        if matched_png is None:
            # 只有「确实存在多张同名 PNG」才值得上报，避免把坏 JSON 也算进来
            try:
                data = json.loads(j.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            basename = Path(data.get("imagePath", "")).name
            if basename and len(png_index.get(basename, [])) > 1:
                ambiguous.append(j)
            continue
        target_json = matched_png.with_suffix(".json")
        if target_json == j:
            continue
        plans.append((j, target_json, matched_png, matched_png.name))
    return plans, ambiguous


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


def apply_alignments(plans: list[tuple[Path, Path, Path, str]]) -> list[str]:
    """真正执行对齐：改 JSON 文件名 + 同步 imagePath。

    顺序是「先 rename、成功后再改 imagePath」：反过来一旦 rename 失败，JSON 会
    留在旧文件名上而 imagePath 已指向新名字，制造出本脚本要修的错位。

    Returns:
        失败信息列表；空列表表示全部成功。
    """
    errors: list[str] = []
    for json_old, json_new, _png, new_image_name in tqdm(
        plans, desc="对齐 JSON", unit="file", leave=False,
    ):
        # 1) 先改 JSON 文件名（撞名兜底）
        if json_new.exists():
            json_new = _resolve_json_collision(json_new)
        try:
            json_old.rename(json_new)
        except OSError as exc:
            errors.append(f"{json_old.name} 改名失败，已跳过：{exc}")
            continue

        # 2) 改名成功后再同步 imagePath 字段
        try:
            data = json.loads(json_new.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{json_new.name} 改名后无法读回，imagePath 未同步：{exc}")
            continue
        if data.get("imagePath") == new_image_name:
            continue
        data["imagePath"] = new_image_name
        try:
            json_new.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            # 文件名已改、imagePath 未改：必须让人看见，否则又是一次静默错位
            errors.append(
                f"{json_new.name} 已改名但 imagePath 写入失败（现指向 "
                f"{data.get('imagePath')!r}）：{exc}"
            )
    return errors


def align_labelme(
    root: Path,
    *,
    recursive: bool = DEFAULT_RECURSIVE,
    dry_run: bool = DEFAULT_DRY_RUN,
) -> tuple[list[tuple[Path, Path, Path, str]], list[Path], list[str]]:
    """执行对齐。

    Returns:
        ``(plans, ambiguous, errors)``：对齐计划、因多候选而跳过的 JSON、
        以及执行期的失败信息（dry-run 时恒为空）。
    """
    pngs, jsons = collect_files(root, recursive=recursive)
    plans, ambiguous = plan_alignments(pngs, jsons)
    errors: list[str] = []
    if not dry_run:
        errors = apply_alignments(plans)
    return plans, ambiguous, errors


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

    plans, ambiguous, errors = align_labelme(
        args.root,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )

    mode = "预览" if args.dry_run else "已执行"
    print(f"[{mode}] {len(plans)} 个 JSON 将对齐到同名 PNG")
    for json_old, json_new, png, _ in plans:
        print(f"  {json_old.name}  ->  {json_new.name}  (匹配 {png.name})")
    if ambiguous:
        print(
            f"\n[警告] {len(ambiguous)} 个 JSON 的同名 PNG 有多张、无法确定归属，"
            f"已跳过（需人工指定）："
        )
        for j in ambiguous:
            print(f"  {j}")
    if errors:
        print(f"\n[错误] {len(errors)} 个文件处理失败：", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n（这是 dry-run 预览，加 --apply 才会真改）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
