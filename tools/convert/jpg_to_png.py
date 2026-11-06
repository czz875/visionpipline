r"""
tools/convert/jpg_to_png.py
把 JPG 批量转换为 PNG，同时修复同名 LabelMe JSON 标注。

典型用法：

    .conda\python.exe tools\convert\jpg_to_png.py ^
        --input datasets\raw_jpg ^
        --output datasets\raw

    .conda\python.exe tools\convert\jpg_to_png.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

import cv2

# 允许以 `python tools/convert/jpg_to_png.py` 直接运行。
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import build_timestamped_output_dir  # noqa: E402
from tools.core.image_io import read_image, write_image  # noqa: E402
from tools.core.labelme import load_labelme, save_labelme  # noqa: E402
from tools.core.file_io import move_file_without_overwrite, remove_owned_file  # noqa: E402

# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_INPUT_DIR = Path("datasets/raw_jpg")
DEFAULT_OUTPUT_DIR = Path("datasets/raw")
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
    """在独立临时目录完成图片与标注，再排他发布；只统计完整成功的样本。"""
    img = read_image(jpg_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        with print_lock:
            print("读取失败:", jpg_path)
        return False

    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        input_json = jpg_path.with_suffix(".json")
        output_json = png_path.with_suffix(".json")
        normalized = None
        if input_json.exists() or input_json.is_symlink():
            data = load_labelme(input_json)
            if not isinstance(data, dict):
                raise ValueError(f"JSON 顶层必须是对象：{input_json}")
            normalized = normalize_labelme_data(data, new_image_path=png_path.name)
            normalized["imageHeight"], normalized["imageWidth"] = img.shape[:2]

        for target in (png_path, output_json):
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"目标已存在，禁止覆盖：{target}")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".jpg_to_png_", dir=png_path.parent.resolve()) as directory:
            staging = Path(directory)
            staged_image = staging / png_path.name
            write_image(staged_image, img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
            if normalized is not None:
                save_labelme(normalized, staged_image.with_suffix(".json"))
            identity = move_file_without_overwrite(staged_image, png_path)
            published.append((png_path, identity))
            if normalized is not None:
                identity = move_file_without_overwrite(staged_image.with_suffix(".json"), output_json)
                published.append((output_json, identity))
    except (OSError, ValueError, TypeError) as e:
        for target, identity in reversed(published):
            remove_owned_file(target, identity)
        with print_lock:
            print(f"转换失败: {jpg_path}（{e}）")
        return False

    with print_lock:
        print("转换成功:", jpg_path, "->", png_path)
    with counter_lock:
        counters["image"] += 1
        counters["json"] += int(normalized is not None)
    return True


# =============================================================================
# 4. 主流程
# =============================================================================


def _validate_input_parameters(input_dir: Path, num_threads: int, png_compression: int) -> None:
    """校验输入和参数，不创建输出。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录：{input_dir}")
    if num_threads < 1:
        raise ValueError("线程数必须大于 0。")
    if not 0 <= png_compression <= 9:
        raise ValueError("PNG 压缩等级必须在 0 到 9 之间。")


def _validate_output_path(output_dir: Path) -> Path:
    """在解析符号链接前检查用户提供的完整输出路径。"""
    output_dir = Path(os.path.abspath(output_dir))
    if output_dir.resolve() != output_dir:
        raise ValueError(f"输出路径经过目录链接：{output_dir}")
    return output_dir


def convert_jpg_to_png(
    input_dir: Path,
    output_dir: Path,
    *,
    num_threads: int = DEFAULT_NUM_THREADS,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
    recursive: bool = DEFAULT_RECURSIVE,
) -> tuple[int, int]:
    """批量转换并返回完整成功的图片、JSON 数；预检或任一样本失败时抛异常。"""
    input_dir, output_dir = input_dir.resolve(), _validate_output_path(output_dir)
    _validate_input_parameters(input_dir, num_threads, png_compression)
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("输入与输出目录不能重叠。")
    tasks = _collect_tasks(input_dir, output_dir, recursive)
    if not tasks:
        raise FileNotFoundError(f"未找到 JPG/JPEG 图片：{input_dir}")
    destinations: set[str] = set()
    planned_files: set[str] = set()
    planned_directories: set[str] = set()
    for jpg_path, png_path in tasks:
        key = str(png_path).casefold()
        if key in destinations:
            raise ValueError(f"转换目标冲突，多个 JPG/JPEG 对应同一 PNG：{png_path}")
        destinations.add(key)
        for target in (png_path, png_path.with_suffix(".json")):
            planned_files.add(str(target).casefold())
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"目标已存在，禁止覆盖：{target}")
            if target.resolve() != target:
                raise ValueError(f"输出路径经过目录链接：{target}")
            for parent in target.parents:
                planned_directories.add(str(parent).casefold())
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"输出的父路径不是目录：{parent}")
                if parent == output_dir:
                    break
    if planned_files & planned_directories:
        raise ValueError("转换目标冲突：同一路径同时被规划为文件和目录。")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"共发现 {len(tasks)} 个JPG文件，使用 {num_threads} 个线程处理...\n")

    print_lock = threading.Lock()
    counter_lock = threading.Lock()
    counters = {"image": 0, "json": 0}
    failures: list[Path] = []

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
                if not future.result():
                    failures.append(jpg_path)
            except Exception as e:
                failures.append(jpg_path)
                with print_lock:
                    print(f"处理异常 {jpg_path}: {e}")

    print("\n总共转换:", counters["image"], "张图片")
    print("总共修复:", counters["json"], "个JSON")
    if failures:
        raise RuntimeError(f"{len(failures)} 个样本转换失败；完整成功 {counters['image']} 张图片、{counters['json']} 个 JSON。")
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
        default=None,
        help="输出目录；未指定时自动生成时间戳子目录。",
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
        help=f"PNG 压缩等级 0-9（默认 {DEFAULT_PNG_COMPRESSION}，最小体积）。",
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

    output_dir = args.output
    try:
        _validate_input_parameters(args.input, args.num_threads, args.png_compression)
        if output_dir is None:
            output_base = _validate_output_path(DEFAULT_OUTPUT_DIR)
            output_dir = build_timestamped_output_dir(output_base, "jpg_to_png")
        convert_jpg_to_png(
            input_dir=args.input, output_dir=output_dir, num_threads=args.num_threads,
            png_compression=args.png_compression, recursive=args.recursive,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        if args.output is None and output_dir is not None and output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        return 1
    print(f"OUTPUT_PATH:{output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
