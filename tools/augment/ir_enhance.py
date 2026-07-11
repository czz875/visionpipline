"""
tools/augment/ir_enhance.py
IR（红外）图像增强脚本。

对图像做：
1. 全局光照增强（对比度、亮度、gamma、方向性光斑、暗角）。
2. 目标区域局部光照增强（CLAHE + 方向光 + 高斯光斑 + 柔化融合）。
3. 传感器效果（高斯噪声 + 轻微模糊）。

局部增强默认对所有 YOLO 类别都生效（人脸 / 人 / 车 / 手机等），
也可以通过 ``--local-light-classes`` 限定为指定类别。

标签坐标保持不变，只复制写出对应 .txt。

典型用法：

    .conda\python.exe tools\augment\ir_enhance.py ^
        --input-dir datasets\raw ^
        --output-dir datasets\raw_ir_aug ^
        --repeat 3 ^
        --workers 8

    .conda\python.exe tools\augment\ir_enhance.py --help
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# 允许以 `python tools/augment/ir_enhance.py` 直接运行。
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_INPUT_DIR = r"D:\A_CJET_WORKSPACE\01_CJET_DATASET\11_activateBody\yongqing_all_txt"
DEFAULT_OUTPUT_DIR = r"D:\A_CJET_WORKSPACE\01_CJET_DATASET\11_activateBody\yongqing_all_enh"
DEFAULT_REPEAT = 3
DEFAULT_MAX_WORKERS = 8
DEFAULT_LOCAL_LIGHT_EXPAND_MIN = 1.25
DEFAULT_LOCAL_LIGHT_EXPAND_MAX = 1.55
DEFAULT_VALID_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_LABEL_EXT = ".txt"
DEFAULT_RECURSIVE = True


# =============================================================================
# 2. 标签读写
# =============================================================================


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yolo_labels(label_path: Path) -> list[list[float]]:
    """读取 YOLO 标签文件，返回 [[cls, cx, cy, w, h], ...]。"""
    boxes: list[list[float]] = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts[:5])
            boxes.append([int(cls_id), cx, cy, w, h])
    return boxes


def save_yolo_labels(label_path: Path, boxes: list[list[float]]) -> None:
    """写出 YOLO 标签文件。"""
    _ensure_dir(label_path.parent)
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def collect_image_label_pairs(
    img_root: Path,
    lbl_root: Path,
    *,
    image_exts: tuple[str, ...] = DEFAULT_VALID_IMAGE_EXTS,
    label_ext: str = DEFAULT_LABEL_EXT,
) -> list[tuple[Path, Path, str]]:
    """递归收集图像与标签配对 [(img, lbl, rel_dir), ...]。"""
    suffix_set = tuple(s.lower() for s in image_exts)
    pairs: list[tuple[Path, Path, str]] = []
    for root, _, files in os.walk(img_root):
        for name in files:
            if not name.lower().endswith(suffix_set):
                continue
            img_path = Path(root) / name
            rel_path = img_path.relative_to(img_root)
            rel_dir = str(rel_path.parent) if rel_path.parent != Path(".") else ""
            base = rel_path.stem
            lbl_path = lbl_root / rel_path.parent / f"{base}{label_ext}"
            pairs.append((img_path, lbl_path, rel_dir))
    return pairs


# =============================================================================
# 3. 几何 / 蒙版工具
# =============================================================================


def yolo_box_to_xyxy(box: list[float], img_w: int, img_h: int) -> list[int]:
    """把 YOLO 归一化框 [cls, cx, cy, w, h] 转成像素 [x1, y1, x2, y2]。"""
    _, cx, cy, w, h = box
    center_x = cx * img_w
    center_y = cy * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = max(0, int(round(center_x - bw / 2)))
    y1 = max(0, int(round(center_y - bh / 2)))
    x2 = min(img_w, int(round(center_x + bw / 2)))
    y2 = min(img_h, int(round(center_y + bh / 2)))
    return [x1, y1, x2, y2]


def expand_box(box: list[int], img_w: int, img_h: int, scale: float) -> list[int]:
    """按比例扩大矩形框并夹回图像范围。"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    new_w = w * scale
    new_h = h * scale
    new_x1 = max(0, int(round(cx - new_w / 2.0)))
    new_y1 = max(0, int(round(cy - new_h / 2.0)))
    new_x2 = min(img_w, int(round(cx + new_w / 2.0)))
    new_y2 = min(img_h, int(round(cy + new_h / 2.0)))
    return [new_x1, new_y1, new_x2, new_y2]


def get_target_boxes(
    boxes: list[list[float]],
    img_w: int,
    img_h: int,
    *,
    class_filter: set[int] | None,
) -> list[list[int]]:
    """根据 class_filter 提取要做局部增强的目标框。None 表示所有类别。"""
    if class_filter is None:
        return [yolo_box_to_xyxy(b, img_w, img_h) for b in boxes]
    return [yolo_box_to_xyxy(b, img_w, img_h) for b in boxes if b[0] in class_filter]


def _build_direction_light_map(height: int, width: int) -> np.ndarray:
    """构建平滑的方向性光照图，范围约 [-1, 1]。"""
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    angle = random.uniform(0.0, 2.0 * np.pi)
    light_map = np.cos(angle) * grid_x + np.sin(angle) * grid_y
    max_abs = max(float(np.max(np.abs(light_map))), 1e-6)
    return light_map / max_abs


def _build_spot_light_map(height: int, width: int) -> np.ndarray:
    """构建平滑的高斯光斑图，范围约 [0, 1]。"""
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    cx = random.uniform(0.2 * width, 0.8 * width)
    cy = random.uniform(0.2 * height, 0.8 * height)
    sigma_x = random.uniform(0.22 * width, 0.45 * width)
    sigma_y = random.uniform(0.22 * height, 0.45 * height)
    spot = np.exp(
        -(
            ((grid_x - cx) ** 2) / (2.0 * sigma_x * sigma_x)
            + ((grid_y - cy) ** 2) / (2.0 * sigma_y * sigma_y)
        )
    )
    return spot.astype(np.float32)


def _build_soft_target_mask(height: int, width: int) -> np.ndarray:
    """柔化椭圆蒙版（兼容人脸 / 其他目标，过渡自然）。"""
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    mask = np.exp(-((grid_x * grid_x) / 0.55 + (grid_y * grid_y) / 0.75))
    mask = cv2.GaussianBlur(
        mask, (0, 0), max(width / 8.0, 1.0), max(height / 8.0, 1.0)
    )
    return np.clip(mask, 0.0, 1.0)


# =============================================================================
# 4. 增强算子
# =============================================================================


def _apply_global_lighting(img: np.ndarray) -> np.ndarray:
    """对整张图做全局光照增强。"""
    img_f = img.astype(np.float32)
    height, width = img.shape[:2]

    mean_value = float(np.mean(img_f))
    contrast = random.uniform(0.92, 1.12)
    brightness_bias = random.uniform(-10.0, 12.0)
    gamma = random.uniform(0.88, 1.18)

    direction_map = _build_direction_light_map(height, width)
    spot_map = _build_spot_light_map(height, width)
    direction_gain = 1.0 + direction_map * random.uniform(-0.10, 0.18)
    spot_gain = 1.0 + spot_map * random.uniform(-0.08, 0.20)
    light_gain = np.clip(direction_gain * spot_gain, 0.72, 1.30)

    img_f = (img_f - mean_value) * contrast + mean_value + brightness_bias
    img_f = np.clip(img_f, 0, 255)
    img_f = 255.0 * np.power(img_f / 255.0, gamma)
    img_f = img_f * light_gain

    vignette_strength = random.uniform(0.0, 0.10)
    if vignette_strength > 0:
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        radius = np.sqrt(grid_x * grid_x + grid_y * grid_y)
        vignette = 1.0 - vignette_strength * np.clip(radius, 0.0, 1.5)
        img_f = img_f * np.clip(vignette, 0.82, 1.0)

    return np.clip(img_f, 0, 255).astype(np.uint8)


def _apply_local_lighting(
    img: np.ndarray,
    target_boxes: list[list[int]],
    *,
    expand_min: float,
    expand_max: float,
) -> np.ndarray:
    """对指定目标区域做局部光照和细节增强。"""
    result = img.astype(np.float32).copy()
    img_h, img_w = img.shape[:2]

    for box in target_boxes:
        scale = random.uniform(expand_min, expand_max)
        x1, y1, x2, y2 = expand_box(box, img_w, img_h, scale)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue

        roi = result[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]

        soft_mask = _build_soft_target_mask(roi_h, roi_w)
        direction_map = _build_direction_light_map(roi_h, roi_w)
        center_map = _build_spot_light_map(roi_h, roi_w)

        roi_uint8 = np.clip(roi, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=random.uniform(1.2, 2.2), tileGridSize=(4, 4)
        )
        detail_roi = clahe.apply(roi_uint8).astype(np.float32)

        detail_blend = random.uniform(0.20, 0.42)
        local_gain = (
            1.0
            + direction_map * random.uniform(-0.08, 0.15)
            + center_map * random.uniform(0.02, 0.12)
        )
        local_gain = np.clip(local_gain, 0.82, 1.25)
        enhanced_roi = roi * (1.0 - detail_blend) + detail_roi * detail_blend
        enhanced_roi = enhanced_roi * local_gain + random.uniform(-4.0, 8.0)

        blend_mask = soft_mask * random.uniform(0.55, 0.85)
        result[y1:y2, x1:x2] = roi * (1.0 - blend_mask) + enhanced_roi * blend_mask

    return np.clip(result, 0, 255).astype(np.uint8)


def _add_ir_sensor_effect(img: np.ndarray) -> np.ndarray:
    """添加轻微高斯噪声和模糊，模拟 IR 设备成像。"""
    img_f = img.astype(np.float32)
    noise_std = random.uniform(1.5, 4.5)
    noise = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
    img_f = img_f + noise
    if random.random() < 0.35:
        kernel = random.choice([3, 5])
        sigma = random.uniform(0.4, 1.0)
        img_f = cv2.GaussianBlur(img_f, (kernel, kernel), sigma)
    return np.clip(img_f, 0, 255).astype(np.uint8)


def ir_light_augment(
    img: np.ndarray,
    boxes: list[list[float]],
    *,
    class_filter: set[int] | None,
    expand_min: float,
    expand_max: float,
) -> np.ndarray:
    """执行完整 IR 增强：全局 → 局部 → 传感器效果。"""
    aug = _apply_global_lighting(img)
    img_h, img_w = aug.shape[:2]
    target_boxes = get_target_boxes(boxes, img_w, img_h, class_filter=class_filter)
    if target_boxes:
        aug = _apply_local_lighting(
            aug, target_boxes, expand_min=expand_min, expand_max=expand_max
        )
    aug = _add_ir_sensor_effect(aug)
    return aug


# =============================================================================
# 5. 单文件处理
# =============================================================================


def process_one(
    img_path: Path,
    lbl_path: Path,
    rel_dir: str,
    output_dir: Path,
    *,
    repeat: int,
    class_filter: set[int] | None,
    expand_min: float,
    expand_max: float,
) -> None:
    """处理单张图像，输出 repeat 份增强样本和对应标签。"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    boxes = load_yolo_labels(lbl_path)
    base_name = img_path.stem
    ext = img_path.suffix

    out_img_dir = output_dir / rel_dir if rel_dir else output_dir
    _ensure_dir(out_img_dir)

    for i in range(repeat):
        aug_img = ir_light_augment(
            img.copy(),
            boxes,
            class_filter=class_filter,
            expand_min=expand_min,
            expand_max=expand_max,
        )
        img_name = f"{base_name}_ir_aug{i}{ext}"
        lbl_name = f"{base_name}_ir_aug{i}.txt"
        cv2.imwrite(str(out_img_dir / img_name), aug_img)
        save_yolo_labels(out_img_dir / lbl_name, boxes)


# =============================================================================
# 6. 主流程
# =============================================================================


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    repeat: int = DEFAULT_REPEAT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    class_filter: set[int] | None = None,
    expand_min: float = DEFAULT_LOCAL_LIGHT_EXPAND_MIN,
    expand_max: float = DEFAULT_LOCAL_LIGHT_EXPAND_MAX,
    recursive: bool = DEFAULT_RECURSIVE,
) -> None:
    """批量对数据集执行 IR 增强。"""
    if not input_dir.exists():
        print(f"[错误] 输入目录不存在：{input_dir}")
        return

    _ensure_dir(output_dir)
    pairs = collect_image_label_pairs(input_dir, input_dir)
    if not recursive:
        # 已经在 collect_image_label_pairs 内做了递归，这里保留参数位以便未来扩展
        pass
    print(f"[信息] 找到 {len(pairs)} 组图像与标签，repeat={repeat}, workers={max_workers}")
    if class_filter is not None:
        print(f"[信息] 仅对类别 {sorted(class_filter)} 做局部增强")
    else:
        print("[信息] 对所有类别都做局部增强")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one,
                img_path,
                lbl_path,
                rel_dir,
                output_dir,
                repeat=repeat,
                class_filter=class_filter,
                expand_min=expand_min,
                expand_max=expand_max,
            )
            for img_path, lbl_path, rel_dir in pairs
        ]
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="IR 光照增强",
            ncols=100,
        ):
            pass

    print("[完成] IR 图像增强完成。")


# =============================================================================
# 7. 命令行
# =============================================================================


def _parse_classes(text: str) -> set[int] | None:
    """把 ``'0,1,2'`` 转成 ``{0, 1, 2}``；空串或 'all' 返回 None。"""
    text = text.strip()
    if not text or text.lower() in {"all", "none"}:
        return None
    return {int(x) for x in text.replace(" ", "").split(",") if x}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IR 图像增强：全局光照 + 目标区域局部光照 + 传感器效果。",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"输入数据集根目录（默认：{DEFAULT_INPUT_DIR}）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"输出数据集根目录（默认：{DEFAULT_OUTPUT_DIR}）。",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help=f"每张图增强份数（默认 {DEFAULT_REPEAT}）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"线程数（默认 {DEFAULT_MAX_WORKERS}）。",
    )
    parser.add_argument(
        "--local-light-classes",
        type=str,
        default="all",
        help=(
            "要做局部增强的 YOLO 类别 ID，多个用逗号分隔；"
            "留空或 'all' 表示所有类别（默认 all）。"
        ),
    )
    parser.add_argument(
        "--expand-min",
        type=float,
        default=DEFAULT_LOCAL_LIGHT_EXPAND_MIN,
        help=f"目标框最小放大倍数（默认 {DEFAULT_LOCAL_LIGHT_EXPAND_MIN}）。",
    )
    parser.add_argument(
        "--expand-max",
        type=float,
        default=DEFAULT_LOCAL_LIGHT_EXPAND_MAX,
        help=f"目标框最大放大倍数（默认 {DEFAULT_LOCAL_LIGHT_EXPAND_MAX}）。",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RECURSIVE,
        help="递归子目录（默认开）。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repeat=args.repeat,
        max_workers=args.workers,
        class_filter=_parse_classes(args.local_light_classes),
        expand_min=args.expand_min,
        expand_max=args.expand_max,
        recursive=args.recursive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
