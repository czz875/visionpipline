"""
tools/convert/jpg_to_png.py
把 JPG 批量转换为 PNG，同时修复同名 LabelMe JSON 标注。

典型用法：

    .conda\python.exe tools\convert\jpg_to_png.py ^
        --input E:\czz\0024 ^
        --output E:\czz\0024\PNG

    .conda\python.exe tools\convert\jpg_to_png.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

import cv2

# 允许以 `python tools/convert/jpg_to_png.py` 直接运行。
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core.labelme import load_labelme, save_labelme  # noqa: E402

# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_INPUT_DIR = r"E:\czz\0024"
DEFAULT_OUTPUT_DIR = r"E:\czz\0024\PNG"
DEFAULT_NUM_THREADS = 16
DEFAULT_PNG_COMPRESSION = 9
DEFAULT_JPG_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg")
DEFAULT_LABELME_VERSION = "5.2.1"
DEFAULT_RECURSIVE = True


# =============================================================================
# 2. LabelMe JSON 规范化（项目约定 5.2.1 结构）
# =============================================================================


def _normalize_rectangle_points(points: list) -> list[list[float]]:
    """将旧版四点矩形或无序两点矩形统一为左上/右下两点格式。"""
    valid_points: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        valid_points.append((float(point[0]), float(point[1])))

    if len(valid_points) < 2:
        return []

    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    return [[min(xs), min(ys)], [max(xs), max(ys)]]


def _normalize_shape(shape: dict) -> dict:
    """清洗单个 shape，仅保留项目约定的核心字段。"""
    shape_type = shape.get("shape_type", "")
    points = shape.get("points", [])

    if shape_type == "rectangle":
        normalized = _normalize_rectangle_points(points)
        if normalized:
            points = normalized

    flags = shape.get("flags")
    return {
        "label": shape.get("label", ""),
        "group_id": shape.get("group_id"),
        "description": shape.get("description") or "",
        "shape_type": shape_type,
        "flags": flags if isinstance(flags, dict) else {},
        "points": points,
    }


def normalize_labelme_data(data: dict, new_image_path: str | None = None) -> dict:
    """将 LabelMe 标注数据规范化为项目约定的 5.2.1 结构。"""
    shapes = data.get("shapes", [])
    normalized_shapes: list[dict] = []
    if isinstance(shapes, list):
        for shape in shapes:
            if isinstance(shape, dict):
                normalized_shapes.append(_normalize_shape(shape))

    image_path = new_image_path if new_image_path is not None else data.get("imagePath")

    flags = data.get("flags")
    return {
        "version": DEFAULT_LABELME_VERSION,
        "flags": flags if isinstance(flags, dict) else {},
        "shapes": normalized_shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": data.get("imageHeight"),
        "imageWidth": data.get("imageWidth"),
    }


# =============================================================================
# 3. 单文件处理
# =============================================================================


def _collect_tasks(input_dir: Path, output_dir: Path, recursive: bool) -> list[tuple[Path, Path]]:
    """枚举所有待处理的 JPG 文件，返回 (jpg 路径, png 输出路径) 列表。"""
    tasks: list[tuple[Path, Path]] = []
    suffix_set = tuple(s.lower() for s in DEFAULT_JPG_SUFFIXES)

    def _on_walk(root: str, _: list[str], files: list[str]) -> None:
        for name in files:
            if not name.lower().endswith(suffix_set):
                continue
            jpg_path = Path(root) / name
            relative_dir = Path(root).relative_to(input_dir)
            png_path = (output_dir / relative_dir / name).with_suffix(".png")
            tasks.append((jpg_path, png_path))

    if recursive:
        for root, _, files in os.walk(input_dir):
            _on_walk(root, _, files)
    else:
        for entry in input_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() in suffix_set:
                png_path = (output_dir / entry.name).with_suffix(".png")
                tasks.append((entry, png_path))
    return tasks


def process_single_file(
    jpg_path: Path,
    png_path: Path,
    *,
    png_compression: int,
    print_lock: threading.Lock,
    counter_lock: threading.Lock,
    counters: dict,
) -> bool:
    """处理单个 JPG：转 PNG + 修复同名 JSON。"""
    img = cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        with print_lock:
            print("读取失败:", jpg_path)
        return False

    png_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(png_path), img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression]):
        with print_lock:
            print("保存失败:", png_path)
        return False

    with print_lock:
        print("转换成功:", jpg_path, "->", png_path)
    with counter_lock:
        counters["image"] += 1

    # 处理同名 JSON 标注
    base_name = jpg_path.stem
    input_json = jpg_path.with_suffix(".json")
    output_json = png_path.with_suffix(".json")
    if input_json.exists():
        try:
            data = load_labelme(input_json)
            if isinstance(data, dict):
                normalized = normalize_labelme_data(data, new_image_path=png_path.name)
                save_labelme(normalized, output_json)
                with print_lock:
                    print("修复JSON:", input_json, "->", output_json)
                with counter_lock:
                    counters["json"] += 1
            else:
                with print_lock:
                    print("JSON顶层不是对象，跳过:", input_json)
        except Exception as e:
            with print_lock:
                print(f"修复JSON失败 {input_json}: {e}")

    return True


# =============================================================================
# 4. 主流程
# =============================================================================


def convert_jpg_to_png(
    input_dir: Path,
    output_dir: Path,
    *,
    num_threads: int = DEFAULT_NUM_THREADS,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
    recursive: bool = DEFAULT_RECURSIVE,
) -> tuple[int, int]:
    """批量把 JPG 转为 PNG，并修复同名 JSON。返回 (image_count, json_count)。"""
    if not input_dir.exists():
        print("输入目录不存在:", input_dir)
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(input_dir, output_dir, recursive)
    print(f"共发现 {len(tasks)} 个JPG文件，使用 {num_threads} 个线程处理...\n")

    print_lock = threading.Lock()
    counter_lock = threading.Lock()
    counters = {"image": 0, "json": 0}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                process_single_file,
                jpg_path,
                png_path,
                png_compression=png_compression,
                print_lock=print_lock,
                counter_lock=counter_lock,
                counters=counters,
            ): jpg_path
            for jpg_path, png_path in tasks
        }
        for future in as_completed(futures):
            jpg_path = futures[future]
            try:
                future.result()
            except Exception as e:
                with print_lock:
                    print(f"处理异常 {jpg_path}: {e}")

    print("\n总共转换:", counters["image"], "张图片")
    print("总共修复:", counters["json"], "个JSON")
    return counters["image"], counters["json"]


# =============================================================================
# 5. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 JPG 批量转为 PNG，并修复同名 LabelMe JSON 标注。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"JPG 输入目录（默认：{DEFAULT_INPUT_DIR}）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"PNG 输出目录（默认：{DEFAULT_OUTPUT_DIR}）。",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=f"线程数（默认：{DEFAULT_NUM_THREADS}）。",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=DEFAULT_PNG_COMPRESSION,
        help="PNG 压缩等级 0-9（默认 9，最小体积）。",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RECURSIVE,
        help="递归处理子目录（默认开，--no-recursive 关闭）。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    convert_jpg_to_png(
        input_dir=args.input,
        output_dir=args.output,
        num_threads=args.num_threads,
        png_compression=args.png_compression,
        recursive=args.recursive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
