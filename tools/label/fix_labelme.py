"""
tools/label/fix_labelme.py
扫描并修复 LabelMe JSON 的常见损坏问题，避免下游脚本（YOLO 转换 / 训练）报错。

可修复的问题：

1. ``imagePath`` 指向了已不存在的图片文件名 → 改成当前目录下真实存在的图片名；
2. 矩形 points 不是两点（如只有 1 个点 / 3+ 个点）→ 丢弃；
3. points 不是 ``[[x1, y1], [x2, y2]]`` 格式（如 ``[x1, y1, x2, y2]`` 拍平） → 归一化；
4. ``x1 > x2`` 或 ``y1 > y2`` → 自动交换为左上和右下；
5. ``imageWidth/imageHeight`` 与同名图片实际像素不一致 → 以图片实际尺寸为准；
6. 缺少顶层字段（``version/flags/shapes/imageData`` 等） → 补默认值；
7. 缺同名图片的孤立 JSON → 默认不动（可通过 ``--remove-orphan`` 删掉）；
8. 缺同名 JSON 的孤立图片 → 不在本脚本处理范围（归 ``tools/clean/orphan_images.py``）。

典型用法：

    .conda\\python.exe tools\\label\\fix_labelme.py ^
        --root datasets\\behavior ^
        --recursive

    .conda\\python.exe tools\\label\\fix_labelme.py ^
        --root datasets\\behavior ^
        --recursive ^
        --apply

    .conda\\python.exe tools\\label\\fix_labelme.py ^
        --root datasets\\behavior ^
        --recursive ^
        --remove-orphan ^
        --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import IMAGE_EXTENSIONS, find_image_for_json, list_labelme_files


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_ROOT_DIR = r"datasets\behavior"
DEFAULT_RECURSIVE = True
DEFAULT_DRY_RUN = True
DEFAULT_REMOVE_ORPHAN = False
DEFAULT_VERSION = "5.3.1"


# =============================================================================
# 2. 修复结果统计
# =============================================================================


@dataclass
class FixReport:
    total: int = 0
    fixed: int = 0
    removed: int = 0
    unchanged: int = 0
    failed: int = 0
    issues: dict[str, int] = field(default_factory=dict)

    def add_issue(self, name: str) -> None:
        self.issues[name] = self.issues.get(name, 0) + 1


# =============================================================================
# 3. 修复函数
# =============================================================================


def _read_image_size(image_path: Path) -> tuple[int, int] | None:
    """读图片真实尺寸。返回 ``(width, height)``，失败返回 ``None``。"""
    try:
        from PIL import Image  # Pillow 是项目间接依赖
    except ImportError:
        return None
    try:
        with Image.open(image_path) as img:
            return img.size
    except (OSError, ValueError):
        return None


def _normalize_rectangle(points: object) -> list[list[float]] | None:
    """把 shape 的 points 归一化成 ``[[x1, y1], [x2, y2]]``，无法修复则返回 ``None``。"""
    if not isinstance(points, list) or len(points) < 2:
        return None

    # 处理拍平形式 [x1, y1, x2, y2]
    if len(points) == 2 and all(isinstance(p, (int, float)) for p in points):
        return None
    if len(points) == 4 and all(isinstance(p, (int, float)) for p in points):
        x1, y1, x2, y2 = points
        pts = [[float(x1), float(y1)], [float(x2), float(y2)]]
    else:
        try:
            p0 = points[0]
            p1 = points[1]
            x1, y1 = float(p0[0]), float(p0[1])
            x2, y2 = float(p1[0]), float(p1[1])
            pts = [[x1, y1], [x2, y2]]
        except (TypeError, IndexError, ValueError):
            return None

    # 归一化为左上、右下
    if pts[0][0] > pts[1][0]:
        pts[0][0], pts[1][0] = pts[1][0], pts[0][0]
    if pts[0][1] > pts[1][1]:
        pts[0][1], pts[1][1] = pts[1][1], pts[0][1]
    return pts


def _fix_image_path(data: dict, json_path: Path) -> bool:
    """修复 ``imagePath``，使其指向同目录真实存在的图片。返回是否有改动。"""
    current = data.get("imagePath", "")
    candidate = json_path.with_name(current) if current else None
    if candidate is not None and candidate.exists():
        return False
    real = find_image_for_json(json_path)
    if real is None:
        return False
    data["imagePath"] = real.name
    return True


def _fix_image_size(data: dict, image_path: Path) -> bool:
    """以图片实际尺寸为准修正 ``imageWidth/imageHeight``。返回是否有改动。"""
    real = _read_image_size(image_path)
    if real is None:
        return False
    w, h = real
    changed = False
    if data.get("imageWidth") != w:
        data["imageWidth"] = w
        changed = True
    if data.get("imageHeight") != h:
        data["imageHeight"] = h
        changed = True
    return changed


def _fix_shapes(data: dict) -> bool:
    """清理 shapes 列表：丢掉非 rectangle / 坏 points 的项。返回是否有改动。"""
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        data["shapes"] = []
        return True

    new_shapes: list[dict] = []
    changed = False
    for shape in shapes:
        if not isinstance(shape, dict):
            changed = True
            continue
        if shape.get("shape_type") != "rectangle":
            new_shapes.append(shape)
            continue
        pts = _normalize_rectangle(shape.get("points"))
        if pts is None:
            changed = True
            continue
        if shape.get("points") != pts:
            shape["points"] = pts
            changed = True
        # 补默认字段
        shape.setdefault("group_id", None)
        shape.setdefault("description", "")
        shape.setdefault("flags", {})
        new_shapes.append(shape)
    if len(new_shapes) != len(shapes):
        changed = True
    if changed:
        data["shapes"] = new_shapes
    return changed


def _fix_top_level(data: dict) -> bool:
    """补全 LabelMe 顶层必填字段。返回是否有改动。"""
    changed = False
    defaults: dict = {
        "version": DEFAULT_VERSION,
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageData": None,
    }
    for key, default in defaults.items():
        if key not in data:
            data[key] = default
            changed = True
    return changed


def fix_one(json_path: Path, *, dry_run: bool) -> tuple[str, FixReport]:
    """修复单个 JSON。返回 ``(status, report)``。status ∈ {fixed, unchanged, removed, failed}。"""
    report = FixReport(total=1)
    if not json_path.exists():
        report.failed = 1
        return "failed", report

    try:
        raw = json_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        report.failed = 1
        report.add_issue(f"json_decode_error:{type(e).__name__}")
        return "failed", report

    changed = False
    changed |= _fix_top_level(data)
    report.add_issue("top_level_defaults") if "top_level_defaults" in str(changed) else None

    if _fix_image_path(data, json_path):
        changed = True
        report.add_issue("image_path")

    image_path = find_image_for_json(json_path)
    if image_path is not None:
        if _fix_image_size(data, image_path):
            changed = True
            report.add_issue("image_size")
    else:
        report.add_issue("missing_image")

    if _fix_shapes(data):
        changed = True
        report.add_issue("shapes")

    if not changed:
        return "unchanged", FixReport(total=1, unchanged=1)

    if not dry_run:
        try:
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            report.failed = 1
            return "failed", report
    return "fixed", FixReport(total=1, fixed=1)


# =============================================================================
# 4. 主流程
# =============================================================================


def fix_root(
    root: Path,
    *,
    recursive: bool = DEFAULT_RECURSIVE,
    dry_run: bool = DEFAULT_DRY_RUN,
    remove_orphan: bool = DEFAULT_REMOVE_ORPHAN,
) -> FixReport:
    """递归修复 ``root`` 下所有 LabelMe JSON。"""
    json_files = list_labelme_files(root, recursive=recursive)
    overall = FixReport(total=len(json_files))

    for json_path in tqdm(
        json_files,
        desc="修复 JSON",
        unit="file",
        disable=not json_files,
    ):
        status, sub = fix_one(json_path, dry_run=dry_run)
        if status == "fixed":
            overall.fixed += 1
        elif status == "unchanged":
            overall.unchanged += 1
        elif status == "removed":
            overall.removed += 1
        else:
            overall.failed += 1
        for k, v in sub.issues.items():
            overall.issues[k] = overall.issues.get(k, 0) + v

    if remove_orphan:
        # 再扫一遍：找缺同名图片的孤立 JSON
        orphans = [p for p in tqdm(
            json_files,
            desc="扫描孤儿",
            unit="file",
            disable=not json_files,
            leave=False,
        ) if find_image_for_json(p) is None]
        for json_path in tqdm(
            orphans,
            desc="删除孤儿",
            unit="file",
            disable=not orphans,
            leave=False,
        ):
            if not dry_run:
                try:
                    json_path.unlink()
                except OSError:
                    overall.failed += 1
                    continue
            overall.removed += 1
            overall.issues["orphan_removed"] = overall.issues.get("orphan_removed", 0) + 1

    return overall


# =============================================================================
# 5. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="扫描并修复 LabelMe JSON 的常见损坏问题（默认 dry-run）。",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(DEFAULT_ROOT_DIR),
        help=f"要修复的根目录（默认：{DEFAULT_ROOT_DIR}）。",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RECURSIVE,
        help="是否递归子目录（默认开）。",
    )
    parser.add_argument(
        "--remove-orphan",
        action="store_true",
        help="删除缺同名图片的孤立 JSON（默认保留，需 --apply 才生效）。",
    )
    parser.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="真正执行修复（默认是 dry-run 预览）。",
    )
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"[错误] 根目录不存在：{args.root}")
        return 1

    report = fix_root(
        args.root,
        recursive=args.recursive,
        dry_run=args.dry_run,
        remove_orphan=args.remove_orphan,
    )

    mode = "预览" if args.dry_run else "已执行"
    print(f"[{mode}] 扫描 {report.total} 个 JSON")
    print(f"  - 修复: {report.fixed}")
    print(f"  - 未改动: {report.unchanged}")
    print(f"  - 删除孤儿: {report.removed}")
    print(f"  - 失败: {report.failed}")
    if report.issues:
        print("  - 问题分布：")
        for k, v in sorted(report.issues.items(), key=lambda kv: -kv[1]):
            print(f"      {k:<32} {v}")
    if args.dry_run:
        print("\n（这是 dry-run 预览，加上 --apply 才会真正改盘）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
