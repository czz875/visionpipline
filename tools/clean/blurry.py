"""
基于 CleanVision 检测并移除含有模糊目标区域的图片。

流程：
1. 从 LabelMe 标注中裁剪出目标类别的区域；
2. 用 CleanVision 的 blurry 指标给裁剪图打分；
3. 只要一张原图对应的目标区域里有任意一个被判定为模糊，就把原图移出数据集。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cleanvision import Imagelab
from PIL import Image

from tools.core import (
    DEFAULT_DATASET_PATH,
    YOLO_CLASS_NAMES,
    list_labelme_files,
    load_labelme,
)

DEFAULT_CROP_PATH = Path("datasets_label_crop").resolve()
DEFAULT_OUTPUT_PATH = Path("datasets_blurry").resolve()
DEFAULT_BLURRY_THRESHOLD = 0.18
DEFAULT_WORKERS = 16
# Windows 多进程会在每个评分子进程重新加载 SciPy，实测可耗尽系统提交内存。
DEFAULT_SCORE_WORKERS = 1
# 模糊检测的目标类别取自 tools.core.constants.YOLO_CLASS_NAMES（唯一权威来源），
# 与 YOLO 类别表口径一致——原先这里少了 helmet，会导致头盔区域的模糊永远不被检出。
DEFAULT_TARGET_LABELS: set[str] = set(YOLO_CLASS_NAMES)


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="检测并移除含有模糊目标区域的图片。"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"输入数据集目录（默认：{DEFAULT_DATASET_PATH}）。",
    )
    parser.add_argument(
        "--crop-path",
        type=Path,
        default=DEFAULT_CROP_PATH,
        help=f"裁剪缓存根目录，每轮创建独立子目录（默认：{DEFAULT_CROP_PATH}）。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"模糊原图移出后的目录（默认：{DEFAULT_OUTPUT_PATH}）。",
    )
    parser.add_argument(
        "--blurry-threshold",
        type=float,
        default=DEFAULT_BLURRY_THRESHOLD,
        help=f"模糊图片移出的分数上限，越小移出越少（默认：{DEFAULT_BLURRY_THRESHOLD}）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"裁剪阶段并发数（默认：{DEFAULT_WORKERS}）。",
    )
    parser.add_argument(
        "--target-labels",
        default=",".join(sorted(DEFAULT_TARGET_LABELS)),
        help=(
            "需要检测的目标类别，逗号分隔"
            f"（默认：{','.join(sorted(DEFAULT_TARGET_LABELS))}）。"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子目录（默认只处理顶层）。",
    )
    parser.add_argument(
        "--cleanup-crop",
        action="store_true",
        help="运行结束后只删除本轮裁剪子目录，保留缓存根目录中的其他内容。",
    )
    return parser


def _parse_labels(arg: str) -> set[str]:
    """把逗号分隔的类别字符串解析为集合。"""
    return {label.strip() for label in arg.split(",") if label.strip()}


def _points_to_bbox(points: list[list[float]], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """从 LabelMe points 计算裁剪框，限制在图片范围内。"""
    img_w, img_h = image_size
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(img_w, int(max(xs)))
    y2 = min(img_h, int(max(ys)))
    return x1, y1, x2, y2


def _crop_one_labelme(args: tuple[Path, Path, Path, set[str], int]) -> tuple[dict[Path, Path], Path]:
    """裁剪单个标注，返回裁剪文件到原图的精确映射；失败时中止后续清洗。"""
    json_path, dataset_path, crop_path, target_labels, file_index = args
    data = load_labelme(json_path)
    image_path = Path(os.path.abspath(json_path.parent / data["imagePath"]))
    if not image_path.is_relative_to(dataset_path):
        raise ValueError(f"标注指向数据集之外的图片：{json_path} -> {image_path}")
    if image_path.resolve() != image_path:
        raise ValueError(f"图片路径经过符号链接，请先对齐到实际图片路径：{image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"标注对应的图片不存在：{json_path} -> {image_path}")

    crop_sources: dict[Path, Path] = {}
    with Image.open(image_path) as img:
        for idx, shape in enumerate(data.get("shapes", [])):
            if shape.get("label") not in target_labels:
                continue
            x1, y1, x2, y2 = _points_to_bbox(shape["points"], img.size)
            if x2 <= x1 or y2 <= y1:
                continue
            # 使用任务序号命名，避免同名图、含下划线的类别或扩展名导致反查歧义。
            save_path = crop_path / f"{file_index:08d}_{idx:06d}.png"
            img.crop((x1, y1, x2, y2)).convert("RGB").save(save_path)
            crop_sources[save_path] = image_path
    return crop_sources, image_path


def _crop_labelme(
    dataset_path: Path,
    crop_path: Path,
    target_labels: set[str],
    workers: int,
    recursive: bool,
) -> tuple[dict[Path, Path], dict[Path, set[Path]]]:
    """裁剪目标区域，返回裁剪到原图、原图到所有来源 JSON 的映射。"""
    from concurrent.futures import ThreadPoolExecutor

    dataset_path, crop_path = dataset_path.resolve(), crop_path.resolve()
    json_files = list_labelme_files(dataset_path, recursive=recursive)
    crop_path.mkdir(parents=True, exist_ok=True)
    if not json_files:
        print(f"[警告] 在 {dataset_path} 中未找到 LabelMe JSON。")
        return {}, {}

    tasks = [(jp, dataset_path, crop_path, target_labels, i) for i, jp in enumerate(json_files)]

    print(f"开始裁剪 {len(tasks)} 个 LabelMe 标注...")
    crop_sources: dict[Path, Path] = {}
    source_annotations: dict[Path, set[Path]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for json_path, (crops, image_path) in zip(json_files, executor.map(_crop_one_labelme, tasks)):
            crop_sources.update(crops)
            source_annotations.setdefault(image_path, set()).add(json_path)
    return crop_sources, source_annotations


def _find_blurry_originals(
    crop_path: Path,
    blurry_threshold: float,
    crop_sources: dict[Path, Path],
) -> set[Path]:
    """运行 CleanVision，返回需要移出的原图路径集合。"""
    if not crop_sources:
        print("[警告] 裁剪目录为空，跳过模糊检测。")
        return set()

    imagelab = Imagelab(data_path=str(crop_path))
    imagelab.find_issues(issue_types={"blurry": {}}, n_jobs=DEFAULT_SCORE_WORKERS)

    issues = imagelab.issues
    print(f"\nCleanVision 字段: {issues.columns.tolist()}")

    blurry = issues[
        issues["is_blurry_issue"] & (issues["blurry_score"] <= blurry_threshold)
    ]
    print(f"\n检测到模糊目标区域: {len(blurry)}")

    bad_images: set[Path] = set()
    for crop_name in blurry.index:
        crop_file = Path(crop_name)
        if not crop_file.is_absolute():
            crop_file = crop_path / crop_file
        original = crop_sources.get(crop_file.resolve())
        if original is None:
            raise ValueError(f"模糊检测结果无法对应本轮裁剪：{crop_name}")
        bad_images.add(original)

    return bad_images


def _move_file_without_overwrite(source: Path, target: Path) -> None:
    """独占创建目标再移除源；跨卷复制失败时清理本次创建的目标。

    同卷优先使用硬链接保留文件内容及元数据；不支持硬链接时采用排他复制。
    """
    try:
        os.link(source, target)
    except FileExistsError:
        raise
    except OSError:
        with source.open("rb") as source_file:
            # 创建失败时不能进入清理分支，以免删除其他任务已经创建的文件。
            target_file = target.open("xb")
            try:
                with target_file:
                    shutil.copyfileobj(source_file, target_file)
                shutil.copystat(source, target)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
    try:
        source.unlink()
    except OSError:
        if source.exists():
            target.unlink(missing_ok=True)
        raise


def _move_bad_images(
    dataset_path: Path,
    output_path: Path,
    bad_images: set[Path],
    source_annotations: dict[Path, set[Path]],
) -> int:
    """把模糊原图连同同名 LabelMe JSON 移动到输出目录，返回移动的图片数。

    图片及其来源 JSON 分别按数据集相对路径镜像移动，保留跨目录 imagePath。
    任一目标不可用时在移动前报错；执行失败时尝试回滚已完成的移动。
    """
    dataset_path, output_path = dataset_path.resolve(), output_path.resolve()
    _validate_directory_layout(dataset_path, output_path)
    plans: dict[Path, Path] = {}
    for image_path in sorted(bad_images):
        if image_path not in source_annotations:
            raise ValueError(f"缺少图片的来源标注记录：{image_path}")
        for annotation in source_annotations[image_path]:
            image_reference = load_labelme(annotation).get("imagePath", "")
            if image_reference and Path(image_reference).anchor:
                raise ValueError(f"待移出标注的 imagePath 带根或盘符，请先改为相对 JSON 的路径：{annotation}")
        for source in (image_path, *sorted(source_annotations[image_path])):
            if not source.is_relative_to(dataset_path) or not source.is_file() or source.resolve() != source:
                raise ValueError(f"待移动文件不在数据集内、不存在或经过链接：{source}")
            target = output_path / source.relative_to(dataset_path)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"模糊移出目标已存在，禁止覆盖：{target}")
            if target.resolve() != target:
                raise ValueError(f"移出目标经过目录链接，拒绝写入：{target}")
            for parent in target.parents:
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"移出目标的父路径不是目录：{parent}")
                if parent == output_path:
                    break
            plans[source] = target

    completed: list[tuple[Path, Path]] = []
    try:
        for source, target in plans.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            _move_file_without_overwrite(source, target)
            completed.append((source, target))
    except OSError as exc:
        rollback_errors: list[str] = []
        for source, target in reversed(completed):
            try:
                if source.exists() or source.is_symlink():
                    raise FileExistsError(f"回滚源路径已被占用：{source}")
                _move_file_without_overwrite(target, source)
            except OSError as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            raise OSError(f"移动失败：{exc}；部分文件回滚失败：{'；'.join(rollback_errors)}") from exc
        raise
    return len(bad_images)


def _validate_directory_layout(*paths: Path) -> None:
    """输入、裁剪缓存和移出目录必须互不重叠，已有路径必须是目录。"""
    for i, left in enumerate(paths):
        if left.exists() and not left.is_dir():
            raise ValueError(f"路径不是目录：{left}")
        for right in paths[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"数据处理目录不能重叠：{left}、{right}")


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    crop_path = args.crop_path.resolve()
    output_path = args.output_path.resolve()
    target_labels = _parse_labels(args.target_labels)

    # 数据集目录不存在时必须报错：list_labelme_files 对不存在的目录返回空列表，
    # 直接往下走会打印「未找到 LabelMe JSON」并返回 0，让工作流误判该阶段成功。
    if not dataset_path.is_dir():
        print(f"[错误] 数据集目录不存在：{dataset_path}")
        return 1

    try:
        _validate_directory_layout(dataset_path, crop_path, output_path)
        if args.workers < 1:
            raise ValueError("--workers 必须大于 0。")
        crop_path.mkdir(parents=True, exist_ok=True)
        run_crop_path = Path(tempfile.mkdtemp(prefix="blurry_", dir=crop_path))
        crop_sources, source_annotations = _crop_labelme(
            dataset_path, run_crop_path, target_labels, args.workers, args.recursive,
        )
        bad_images = _find_blurry_originals(run_crop_path, args.blurry_threshold, crop_sources)
        moved = _move_bad_images(dataset_path, output_path, bad_images, source_annotations)
        print(f"\n移出模糊原图: {moved} 张 -> {output_path}")

        if args.cleanup_crop:
            # 只清理本进程新建的直接子目录，并在删除前再次确认解析后的范围。
            if run_crop_path.resolve().parent != crop_path:
                raise ValueError(f"本轮裁剪目录超出缓存范围，拒绝清理：{run_crop_path}")
            shutil.rmtree(run_crop_path)
            print(f"已清理本轮裁剪目录: {run_crop_path}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[错误] 模糊清洗失败：{exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
