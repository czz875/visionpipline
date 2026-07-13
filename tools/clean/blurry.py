"""
基于 CleanVision 检测并移除含有模糊目标区域的图片。

流程：
1. 从 LabelMe 标注中裁剪出目标类别的区域；
2. 用 CleanVision 的 blurry 指标给裁剪图打分；
3. 只要一张原图对应的目标区域里有任意一个被判定为模糊，就把原图移出数据集。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cleanvision import Imagelab
from PIL import Image

from tools.core import DEFAULT_DATASET_PATH, list_labelme_files, load_labelme

DEFAULT_CROP_PATH = Path("datasets_label_crop").resolve()
DEFAULT_OUTPUT_PATH = Path("datasets_blurry").resolve()
DEFAULT_BLURRY_THRESHOLD = 0.18
DEFAULT_WORKERS = 16
DEFAULT_TARGET_LABELS = {"face", "hand", "phone", "cigarette"}


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
        help=f"目标区域临时裁剪目录（默认：{DEFAULT_CROP_PATH}）。",
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
        help=f"CleanVision 模糊阈值，越小越严格（默认：{DEFAULT_BLURRY_THRESHOLD}）。",
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
        help="需要检测的目标类别，逗号分隔（默认：face,hand,phone,cigarette）。",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子目录（默认只处理顶层）。",
    )
    parser.add_argument(
        "--cleanup-crop",
        action="store_true",
        help="运行结束后删除临时裁剪目录。",
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


def _crop_one_labelme(args: tuple[Path, Path, Path, set[str]]) -> None:
    """对单个 LabelMe JSON 中的目标 shape 进行裁剪并保存。"""
    json_path, dataset_path, crop_path, target_labels = args

    try:
        data = load_labelme(json_path)
        image_path = dataset_path / data["imagePath"]
        if not image_path.exists():
            return

        with Image.open(image_path) as img:
            for idx, shape in enumerate(data.get("shapes", [])):
                if shape.get("label") not in target_labels:
                    continue

                x1, y1, x2, y2 = _points_to_bbox(shape["points"], img.size)
                if x2 <= x1 or y2 <= y1:
                    continue

                save_path = crop_path / f"{image_path.stem}_{shape['label']}_{idx}{image_path.suffix}"
                img.crop((x1, y1, x2, y2)).save(save_path)

    except Exception as e:
        print(f"处理失败 {json_path}: {e}")


def _crop_labelme(
    dataset_path: Path,
    crop_path: Path,
    target_labels: set[str],
    workers: int,
    recursive: bool,
) -> None:
    """多线程裁剪 LabelMe 标注中的目标区域。"""
    from concurrent.futures import ThreadPoolExecutor

    json_files = list_labelme_files(dataset_path, recursive=recursive)
    if not json_files:
        print(f"[警告] 在 {dataset_path} 中未找到 LabelMe JSON。")
        return

    crop_path.mkdir(parents=True, exist_ok=True)
    tasks = [(jp, dataset_path, crop_path, target_labels) for jp in json_files]

    print(f"开始裁剪 {len(tasks)} 个 LabelMe 标注...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_crop_one_labelme, tasks))


def _find_blurry_originals(
    crop_path: Path,
    blurry_threshold: float,
) -> set[Path]:
    """运行 CleanVision，返回需要移出的原图路径集合。"""
    if not any(crop_path.iterdir()):
        print("[警告] 裁剪目录为空，跳过模糊检测。")
        return set()

    imagelab = Imagelab(data_path=str(crop_path))
    imagelab.find_issues(issue_types={"blurry": {}})

    issues = imagelab.issues
    print(f"\nCleanVision 字段: {issues.columns.tolist()}")

    blurry = issues[
        issues["is_blurry_issue"] & (issues["blurry_score"] <= blurry_threshold)
    ]
    print(f"\n检测到模糊目标区域: {len(blurry)}")

    bad_images: set[Path] = set()
    for crop_name in blurry.index:
        stem = Path(crop_name).stem
        # 裁剪文件名格式：{original_stem}_{label}_{idx}
        original_stem = stem.rsplit("_", 2)[0]
        bad_images.add(original_stem)

    return bad_images


def _move_bad_images(
    dataset_path: Path,
    output_path: Path,
    bad_stems: set[str],
) -> int:
    """把模糊原图移动到输出目录，返回移动数量。"""
    output_path.mkdir(parents=True, exist_ok=True)
    moved = 0
    for image_path in list_images(dataset_path, recursive=True):
        if image_path.stem in bad_stems:
            dst = output_path / image_path.name
            shutil.move(str(image_path), str(dst))
            moved += 1
    return moved


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    crop_path = args.crop_path.resolve()
    output_path = args.output_path.resolve()
    target_labels = _parse_labels(args.target_labels)

    # 1. 裁剪目标区域
    _crop_labelme(
        dataset_path,
        crop_path,
        target_labels,
        args.workers,
        args.recursive,
    )

    # 2. CleanVision 模糊检测
    bad_stems = _find_blurry_originals(crop_path, args.blurry_threshold)

    # 3. 移出模糊原图
    moved = _move_bad_images(dataset_path, output_path, bad_stems)
    print(f"\n移出模糊原图: {moved} 张 -> {output_path}")

    # 4. 可选清理临时裁剪目录
    if args.cleanup_crop and crop_path.exists():
        shutil.rmtree(crop_path)
        print(f"已清理临时裁剪目录: {crop_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
