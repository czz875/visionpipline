# ONNX 重标注后端接入 auto.py 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 ONNX 推理能力通用化并接入 `tools/annotate/auto.py`，新增 `reannotate.py` 与 `mask_regions.py`，最终删除过度特化的 `reannotate_face_hand_onnx.py`。

**架构：** `auto.py` 新增 `ONNXLabeler`，解码逻辑抽离到 `tools/annotate/onnx_decoder.py`；`reannotate.py` 通过 YAML 配置驱动多个 `ONNXLabeler`，按类别对旧 LabelMe JSON 执行覆盖/保留/合并，并输出 `.removed.json`；`mask_regions.py` 读取 `.removed.json` 对原图做模糊/马赛克/涂黑。

**技术栈：** Python 3.11、onnxruntime、supervision、PyYAML、OpenCV、pytest。

---

## 文件清单

| 文件 | 职责 |
|---|---|
| `tools/annotate/onnx_decoder.py` | ONNX 输出解码器抽象 + 内置 YOLOv8/YOLOv5/yolov5s-lmk 解码器。 |
| `tools/annotate/auto.py` | 新增 `ONNXLabeler` 与 `--model-type onnx` CLI。 |
| `tools/annotate/reannotate.py` | 按 YAML 配置重标注并合并旧 LabelMe JSON。 |
| `tools/augment/mask_regions.py` | 按 `.removed.json` 对图片做区域后处理。 |
| `tests/tools/annotate/test_onnx_decoder.py` | 解码器单元测试。 |
| `tests/tools/annotate/test_reannotate.py` | 配置解析与合并逻辑测试。 |
| `tests/tools/augment/test_mask_regions.py` | 区域遮罩处理测试。 |
| `tools/annotate/reannotate_face_hand_onnx.py` | 删除。 |
| `tests/tools/annotate/test_reannotate_face_hand_onnx.py` | 删除。 |
| `tools/core/labelme.py` | 为 `labelme_dict_to_detections` 补充默认 `confidence=1.0`，保证与 ONNX 检测结果合并时字段一致。 |
| `tools/cfg/reannotate_onnx.yaml.example` | 新增通用重标注配置示例。 |
| `AGENTS.md` | 补充新脚本的 CLI 示例。 |

---

## 任务 1：创建 `tools/annotate/onnx_decoder.py` 与单元测试

**文件：**
- 创建：`tools/annotate/onnx_decoder.py`
- 创建：`tests/tools/annotate/test_onnx_decoder.py`

### 步骤 1：编写失败的测试

创建 `tests/tools/annotate/test_onnx_decoder.py`：

```python
from __future__ import annotations

import numpy as np
import pytest

from tools.annotate.onnx_decoder import DECODER_REGISTRY, YoloV8Decoder, YoloV5Decoder, Yolov5sLmkDecoder


def test_yolov8_decoder_empty():
    decoder = YoloV8Decoder()
    outputs = [np.zeros((1, 84, 1), dtype=np.float32)]
    dets = decoder.decode(outputs, (640, 640, 3), (640, 640), 0.25, 0.45)
    assert len(dets) == 0


def test_yolov8_decoder_one_box():
    decoder = YoloV8Decoder()
    # (1, 84, 2): 2 candidates, 4 bbox + 80 classes
    out = np.zeros((1, 84, 2), dtype=np.float32)
    # candidate 0: center (320,320), wh (64,64), class 0 score 0.9
    out[0, 0, 0] = 320.0        # x center
    out[0, 1, 0] = 320.0        # y center
    out[0, 2, 0] = 64.0         # width
    out[0, 3, 0] = 64.0         # height
    out[0, 4, 0] = 0.9          # class 0 confidence
    # candidate 1: low score
    out[0, 4, 1] = 0.1
    dets = decoder.decode([out], (640, 640, 3), (640, 640), 0.25, 0.45)
    assert len(dets) == 1
    np.testing.assert_allclose(dets.xyxy[0], [288.0, 288.0, 352.0, 352.0], atol=1.0)


def test_yolov5_decoder_one_box():
    decoder = YoloV5Decoder()
    out = np.zeros((1, 2, 85), dtype=np.float32)
    out[0, 0, 0] = 320.0
    out[0, 0, 1] = 320.0
    out[0, 0, 2] = 64.0
    out[0, 0, 3] = 64.0
    out[0, 0, 4] = 0.9
    out[0, 1, 4] = 0.1
    dets = decoder.decode([out], (640, 640, 3), (640, 640), 0.25, 0.45)
    assert len(dets) == 1


def test_yolov5s_lmk_decoder_one_box():
    decoder = Yolov5sLmkDecoder()
    out = np.zeros((1, 2, 16), dtype=np.float32)
    out[0, 0, 0] = 320.0
    out[0, 0, 1] = 320.0
    out[0, 0, 2] = 64.0
    out[0, 0, 3] = 64.0
    out[0, 0, 4] = 0.95
    out[0, 0, 15] = 0.95
    out[0, 1, 4] = 0.1
    dets = decoder.decode([out], (640, 640, 3), (640, 640), 0.25, 0.45)
    assert len(dets) == 1


def test_decoder_registry_contains_expected_keys():
    assert set(DECODER_REGISTRY.keys()) >= {"yolo_v8", "yolo_v5", "yolov5s_lmk"}
```

### 步骤 2：运行测试验证失败

```bash
.conda\python.exe -m pytest tests\tools\annotate\test_onnx_decoder.py -v
```

预期：4 个 `ImportError` 或 `ModuleNotFoundError`。

### 步骤 3：编写最少实现代码

创建 `tools/annotate/onnx_decoder.py`：

```python
"""
tools/annotate/onnx_decoder.py

ONNX 检测模型输出解码器。支持通过注册表扩展。
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if TYPE_CHECKING:
    import supervision as sv


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    return np.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2.0,
            boxes[:, 1] - boxes[:, 3] / 2.0,
            boxes[:, 0] + boxes[:, 2] / 2.0,
            boxes[:, 1] + boxes[:, 3] / 2.0,
        ],
        axis=1,
    ).astype(np.float32)


def _scale_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int, int],
    input_size: tuple[int, int],
) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    image_h, image_w = image_shape[:2]
    input_w, input_h = input_size
    scaled = boxes.copy().astype(np.float32)
    scaled[:, [0, 2]] *= image_w / float(input_w)
    scaled[:, [1, 3]] *= image_h / float(input_h)
    return scaled


class OnnxDecoder(ABC):
    @abstractmethod
    def decode(
        self,
        outputs: list[np.ndarray],
        image_shape: tuple[int, int, int],
        input_size: tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
    ) -> sv.Detections:
        """把 ONNX 原始输出解析为 supervision.Detections。"""
        ...


class YoloV8Decoder(OnnxDecoder):
    def decode(self, outputs, image_shape, input_size, conf_threshold, iou_threshold):
        import supervision as sv

        rows = outputs[0][0].T  # (N, 84)
        scores = np.max(rows[:, 4:], axis=1)
        keep = scores >= conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()
        boxes = _xywh_to_xyxy(rows[keep, :4])
        class_ids = np.argmax(rows[keep, 4:], axis=1).astype(int)
        scores = scores[keep]
        boxes = _scale_boxes(boxes, image_shape, input_size)
        dets = sv.Detections(
            xyxy=boxes,
            confidence=scores.astype(np.float32),
            class_id=class_ids,
        )
        return dets.with_nms(class_agnostic=True, threshold=iou_threshold)


class YoloV5Decoder(OnnxDecoder):
    def decode(self, outputs, image_shape, input_size, conf_threshold, iou_threshold):
        import supervision as sv

        rows = outputs[0][0]  # (N, 85)
        scores = rows[:, 4]
        keep = scores >= conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()
        boxes = _xywh_to_xyxy(rows[keep, :4])
        scores = scores[keep]
        class_ids = np.zeros(len(boxes), dtype=int)
        boxes = _scale_boxes(boxes, image_shape, input_size)
        dets = sv.Detections(
            xyxy=boxes,
            confidence=scores.astype(np.float32),
            class_id=class_ids,
        )
        return dets.with_nms(class_agnostic=True, threshold=iou_threshold)


class Yolov5sLmkDecoder(OnnxDecoder):
    def decode(self, outputs, image_shape, input_size, conf_threshold, iou_threshold):
        import supervision as sv

        rows = outputs[0][0]  # (N, K)
        scores = rows[:, 4] * rows[:, 15]
        keep = scores >= conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()
        boxes = _xywh_to_xyxy(rows[keep, :4])
        scores = scores[keep]
        class_ids = np.zeros(len(boxes), dtype=int)
        boxes = _scale_boxes(boxes, image_shape, input_size)
        dets = sv.Detections(
            xyxy=boxes,
            confidence=scores.astype(np.float32),
            class_id=class_ids,
        )
        return dets.with_nms(class_agnostic=True, threshold=iou_threshold)


DECODER_REGISTRY: dict[str, type[OnnxDecoder]] = {
    "yolo_v8": YoloV8Decoder,
    "yolo_v5": YoloV5Decoder,
    "yolov5s_lmk": Yolov5sLmkDecoder,
}
```

### 步骤 4：运行测试验证通过

```bash
.conda\python.exe -m pytest tests\tools\annotate\test_onnx_decoder.py -v
```

预期：4 个测试全部 PASS。

### 步骤 5：Commit

```bash
git add tools/annotate/onnx_decoder.py tests/tools/annotate/test_onnx_decoder.py
git commit -m "feat(annotate): 新增 ONNX 输出解码器及注册表

- 提供 YoloV8Decoder、YoloV5Decoder、Yolov5sLmkDecoder
- 解码器通过 DECODER_REGISTRY 按名称注册，便于扩展
- 添加单元测试覆盖空输出与单框场景"
```

---

## 任务 2：在 `tools/annotate/auto.py` 中接入 `ONNXLabeler`

**文件：**
- 修改：`tools/annotate/auto.py`
- 创建：`tests/tools/annotate/test_auto_onnx.py`

### 步骤 1：编写失败的测试

创建 `tests/tools/annotate/test_auto_onnx.py`：

```python
from __future__ import annotations

from pathlib import Path

from tools.annotate.auto import _build_parser


def test_parser_accepts_onnx_model_type():
    parser = _build_parser()
    args = parser.parse_args(["--model-type", "onnx", "--model", "weight/face.onnx", "--onnx-decoder", "yolo_v8", "--onnx-classes", "face"])
    assert args.model_type == "onnx"
    assert args.model == Path("weight/face.onnx")
    assert args.onnx_decoder == "yolo_v8"
    assert args.onnx_classes == "face"
```

### 步骤 2：运行测试验证失败

```bash
.conda\python.exe -m pytest tests\tools\annotate\test_auto_onnx.py -v
```

预期：`AttributeError`（`--onnx-decoder` 等参数不存在）。

### 步骤 3：扩展 `tools/annotate/auto.py`

在 `DEFAULT_` 区域新增：

```python
DEFAULT_ONNX_DECODER = "yolo_v8"
DEFAULT_ONNX_CLASSES = ""
DEFAULT_ONNX_INPUT_SIZE = 640
DEFAULT_ONNX_NORMALIZE = True
```

在 `from tools.core import ...` 后新增导入：

```python
from tools.annotate.onnx_decoder import DECODER_REGISTRY
```

在 `SAM3Labeler` 后新增 `ONNXLabeler`：

```python
class ONNXLabeler(AutoLabeler):
    """基于 ONNX Runtime 的检测标注器。"""

    def __init__(
        self,
        model_path: Path,
        classes: list[str],
        decoder: str,
        conf: float = 0.25,
        iou: float = 0.45,
        input_size: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.classes = classes
        if decoder not in DECODER_REGISTRY:
            raise ValueError(f"未知 decoder：{decoder}，可用：{list(DECODER_REGISTRY.keys())}")
        self.decoder = DECODER_REGISTRY[decoder]()
        self.conf = conf
        self.iou = iou
        self.normalize = normalize

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in ort.get_available_providers() \
            else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        input_tensor = self.session.get_inputs()[0]
        self.input_name = input_tensor.name
        if input_size is not None:
            self.input_size = input_size
        else:
            shape = input_tensor.shape
            if len(shape) >= 4:
                self.input_size = (int(shape[3] or 640), int(shape[2] or 640))
            else:
                self.input_size = (640, 640)

    def predict(self, image_path: Path) -> DetectionsLike:
        import supervision as sv
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片：{image_path}")
        resized = cv2.resize(image, self.input_size).astype(np.float32)
        if self.normalize:
            resized /= 255.0
        tensor = np.transpose(resized, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {self.input_name: tensor})
        return self.decoder.decode(
            outputs=outputs,
            image_shape=image.shape,
            input_size=self.input_size,
            conf_threshold=self.conf,
            iou_threshold=self.iou,
        )
```

在 `_build_parser()` 中 `--model-type` 的 `choices` 增加 `onnx`，并新增参数：

```python
parser.add_argument(
    "--onnx-decoder",
    default=DEFAULT_ONNX_DECODER,
    help=f"ONNX 解码器名称（默认：{DEFAULT_ONNX_DECODER}）",
)
parser.add_argument(
    "--onnx-classes",
    default=DEFAULT_ONNX_CLASSES,
    help="ONNX 模式下类别名，逗号分隔",
)
parser.add_argument(
    "--onnx-input-size",
    type=int,
    default=DEFAULT_ONNX_INPUT_SIZE,
    help=f"ONNX 模型输入尺寸（默认：{DEFAULT_ONNX_INPUT_SIZE}）",
)
parser.add_argument(
    "--onnx-normalize",
    action="store_true",
    default=DEFAULT_ONNX_NORMALIZE,
    help="ONNX 输入是否做 /255.0 归一化（默认启用）",
)
```

在 `main()` 的模型分支中新增 ONNX 处理：

```python
elif args.model_type == "onnx":
    if not args.model:
        print("[错误] ONNX 模式需要 --model 指定 ONNX 文件", file=sys.stderr)
        return 1
    if not args.onnx_classes:
        print("[错误] ONNX 模式需要 --onnx-classes 指定类别", file=sys.stderr)
        return 1
    classes = [c.strip() for c in args.onnx_classes.split(",") if c.strip()]
    labeler = ONNXLabeler(
        model_path=Path(args.model),
        classes=classes,
        decoder=args.onnx_decoder,
        conf=args.conf,
        iou=args.iou,
        input_size=(args.onnx_input_size, args.onnx_input_size),
        normalize=args.onnx_normalize,
    )
    keep_ids = None
```

### 步骤 4：运行测试与帮助验证

```bash
.conda\python.exe -m pytest tests\tools\annotate\test_auto_onnx.py -v
.conda\python.exe tools\annotate\auto.py --help
```

预期：测试 PASS；`--help` 输出包含 `--model-type onnx` 及相关参数。

### 步骤 5：Commit

```bash
git add tools/annotate/auto.py tests/tools/annotate/test_auto_onnx.py
git commit -m "feat(annotate): auto.py 增加 ONNXLabeler 后端

- 支持 --model-type onnx 及 --onnx-decoder/--onnx-classes/--onnx-input-size
- ONNXLabeler 使用 onnxruntime 推理，复用 onnx_decoder 注册表
- 添加 parser 测试"
```

---

## 任务 3：创建 `tools/annotate/reannotate.py`

**文件：**
- 创建：`tools/annotate/reannotate.py`
- 创建：`tests/tools/annotate/test_reannotate.py`

### 步骤 1：编写失败的测试

创建 `tests/tools/annotate/test_reannotate.py`：

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from tools.annotate.reannotate import load_config, remove_small_boxes


def test_load_config_basic():
    cfg = {
        "models": [
            {"class": "face", "action": "overwrite", "model": "face.onnx", "decoder": "yolov5s_lmk"},
            {"class": "hand", "action": "keep"},
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = load_config(cfg_path)
    assert loaded["models"][0]["class"] == "face"
    assert loaded["models"][1]["action"] == "keep"


def test_remove_small_boxes():
    import supervision as sv

    boxes = np.array([[0, 0, 10, 10], [0, 0, 100, 100]], dtype=np.float32)
    conf = np.array([0.9, 0.9], dtype=np.float32)
    class_id = np.array([0, 0], dtype=int)
    dets = sv.Detections(xyxy=boxes, confidence=conf, class_id=class_id)
    kept, removed = remove_small_boxes(dets, (100, 100, 3), 0.05)
    assert len(kept) == 1
    assert len(removed) == 1
    np.testing.assert_array_equal(kept.xyxy[0], [0, 0, 100, 100])
```

### 步骤 2：运行测试验证失败

```bash
.conda\python.exe -m pytest tests\tools\annotate\test_reannotate.py -v
```

预期：`ImportError` 或 `ModuleNotFoundError`。

### 步骤 3：更新 `tools/core/labelme.py` 的 `labelme_dict_to_detections`

修改 `tools/core/labelme.py` 第 155–160 行：

```python
    if not xyxy_list:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        confidence=np.ones(len(xyxy_list), dtype=np.float32),
        class_id=np.array(class_id_list, dtype=int),
    )
```

运行 `tools/core` 相关测试确认无回归：

```bash
.conda\python.exe -m pytest tests\tools\core\ -q
```

预期：全部通过。

```bash
git add tools/core/labelme.py
git commit -m "fix(core): labelme_dict_to_detections 默认填充 confidence

- 让旧标注与 ONNX 新标注在 sv.Detections.merge 时字段一致
- 不影响已有转换脚本，confidence 原被忽略"
```

### 步骤 4：实现 `tools/annotate/reannotate.py`

```python
"""
tools/annotate/reannotate.py

按 YAML 配置使用 ONNX 模型对已有 LabelMe 数据集进行类别级重标注。
支持覆盖(overwrite)、保留(keep)、合并(merge)三种动作，并可按面积
比例过滤小框，输出 .removed.json 供后续图片区域处理使用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import supervision as sv

from tools.annotate.auto import ONNXLabeler
from tools.core import (
    find_json_for_image,
    list_images,
    load_labelme,
    save_labelme,
    detections_to_labelme_dict,
    labelme_dict_to_detections,
)


DEFAULT_CONFIG = "tools/cfg/reannotate_onnx.yaml"
DEFAULT_INPUT = "datasets/behavior"
DEFAULT_OUTPUT = "datasets/behavior_reannotated"
DEFAULT_RECURSIVE = True
DEFAULT_DRY_RUN = True


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def remove_small_boxes(
    detections: sv.Detections,
    image_shape: tuple[int, int, int],
    min_area_ratio: float | None,
) -> tuple[sv.Detections, sv.Detections]:
    """按面积比例拆分为保留框与待删除框。"""
    if min_area_ratio is None or min_area_ratio <= 0 or len(detections) == 0:
        return detections, sv.Detections.empty()
    image_h, image_w = image_shape[:2]
    image_area = image_h * image_w
    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
        detections.xyxy[:, 3] - detections.xyxy[:, 1]
    )
    ratios = areas / image_area
    keep = ratios >= min_area_ratio
    if np.all(keep):
        return detections, sv.Detections.empty()
    return detections[keep], detections[~keep]


def build_class_name_to_id(class_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def merge_detections(existing: sv.Detections, new: sv.Detections) -> sv.Detections:
    if len(existing) == 0:
        return new
    if len(new) == 0:
        return existing
    return sv.Detections.merge([existing, new])


def rewrite_labelme(
    json_path: Path,
    detections: sv.Detections,
    class_names: list[str],
    image_path: Path,
    image_shape: tuple[int, int, int],
) -> dict:
    return detections_to_labelme_dict(
        detections,
        class_names=class_names,
        image_path=image_path.name,
        image_width=image_shape[1],
        image_height=image_shape[0],
    )


def write_removed_json(
    output_path: Path,
    removed: sv.Detections,
    class_names: list[str],
    image_path: Path,
    image_shape: tuple[int, int, int],
) -> None:
    boxes = []
    if len(removed) > 0:
        for i in range(len(removed)):
            cls_id = int(removed.class_id[i])
            if cls_id < 0 or cls_id >= len(class_names):
                continue
            boxes.append({
                "label": class_names[cls_id],
                "xyxy": removed.xyxy[i].tolist(),
                "reason": "min_area_ratio",
            })
    data = {
        "imagePath": image_path.name,
        "imageWidth": image_shape[1],
        "imageHeight": image_shape[0],
        "boxes": boxes,
    }
    save_labelme(data, output_path)


def build_labelers(
    config: dict[str, Any],
) -> tuple[dict[str, ONNXLabeler], list[str]]:
    class_names = [item["class"] for item in config["models"]]
    labelers: dict[str, ONNXLabeler] = {}
    for item in config["models"]:
        action = item.get("action", "overwrite")
        if action == "keep":
            continue
        model_path = Path(item["model"])
        decoder = item["decoder"]
        input_size = item.get("input_size")
        size = (int(input_size), int(input_size)) if input_size else None
        labelers[item["class"]] = ONNXLabeler(
            model_path=model_path,
            classes=[item["class"]],
            decoder=decoder,
            conf=item.get("conf", 0.25),
            iou=item.get("iou", 0.45),
            input_size=size,
            normalize=item.get("normalize", True),
        )
    return labelers, class_names


def process_image(
    image_path: Path,
    json_path: Path,
    config: dict[str, Any],
    labelers: dict[str, ONNXLabeler],
    class_names: list[str],
) -> tuple[sv.Detections, sv.Detections]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    shape = image.shape

    class_to_id = build_class_name_to_id(class_names)
    existing_dets = sv.Detections.empty()
    if json_path.exists():
        data = load_labelme(json_path)
        existing_dets = labelme_dict_to_detections(data, class_to_id)

    all_kept: list[sv.Detections] = []
    all_removed: list[sv.Detections] = []

    for item in config["models"]:
        class_name = item["class"]
        action = item.get("action", "overwrite")
        min_ratio = item.get("min_area_ratio")

        if action == "keep":
            mask = existing_dets.class_id == class_to_id[class_name]
            all_kept.append(existing_dets[mask])
            continue

        labeler = labelers[class_name]
        new_dets = labeler.predict(image_path)
        kept, removed = remove_small_boxes(new_dets, shape, min_ratio)

        if action == "merge":
            existing_class = existing_dets[existing_dets.class_id == class_to_id[class_name]]
            kept = merge_detections(existing_class, kept)

        all_kept.append(kept)
        all_removed.append(removed)

    final_kept = sv.Detections.merge(all_kept) if all_kept else sv.Detections.empty()
    final_removed = sv.Detections.merge(all_removed) if all_removed else sv.Detections.empty()
    return final_kept, final_removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 YAML 配置重标注 LabelMe 数据集")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--recursive", action="store_true", default=DEFAULT_RECURSIVE)
    parser.add_argument("--apply", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"[错误] 配置文件不存在：{args.config}", file=sys.stderr)
        return 1
    config = load_config(args.config)
    labelers, class_names = build_labelers(config)

    images = list_images(args.input, recursive=args.recursive)
    if not images:
        print(f"[错误] 未找到图片：{args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_removed = 0

    for image_path in tqdm(images, desc="重标注"):
        json_path = find_json_for_image(image_path)
        kept, removed = process_image(image_path, json_path, config, labelers, class_names)
        total_kept += len(kept)
        total_removed += len(removed)

        if args.dry_run:
            continue

        new_json = args.output / json_path.name
        data = rewrite_labelme(new_json, kept, class_names, image_path, cv2.imread(str(image_path)).shape)
        save_labelme(data, new_json)

        removed_json = args.output / f"{image_path.stem}.removed.json"
        write_removed_json(removed_json, removed, class_names, image_path, cv2.imread(str(image_path)).shape)

    mode = "预览" if args.dry_run else "完成"
    print(f"[{mode}] 图片数：{len(images)}")
    print(f"[{mode}] 保留框：{total_kept}")
    print(f"[{mode}] 待删除框：{total_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 步骤 5：运行测试与帮助验证

```bash
.conda\python.exe -m pytest tests\tools\annotate\test_reannotate.py -v
.conda\python.exe tools\annotate\reannotate.py --help
```

预期：测试 PASS；帮助信息正常。

### 步骤 6：Commit

```bash
git add tools/annotate/reannotate.py tests/tools/annotate/test_reannotate.py
git commit -m "feat(annotate): 新增通用 ONNX 重标注脚本 reannotate.py

- 支持 YAML 配置 per-class 的 overwrite/keep/merge 动作
- 按 min_area_ratio 过滤小框，输出 .removed.json
- 复用 tools.core 的 LabelMe 桥接与 auto.py 的 ONNXLabeler"
```

---

## 任务 4：创建 `tools/augment/mask_regions.py`

**文件：**
- 创建：`tools/augment/mask_regions.py`
- 创建：`tests/tools/augment/test_mask_regions.py`

### 步骤 1：编写失败的测试

创建 `tests/tools/augment/test_mask_regions.py`：

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from tools.augment.mask_regions import blackout_region, apply_mask_to_image


def test_blackout_region():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    box = np.array([10, 10, 30, 30], dtype=np.float32)
    result = blackout_region(image, box)
    assert np.all(result[10:30, 10:30] == 0)
    assert np.all(result[0:10, :] == 255)


def test_apply_mask_to_image_with_removed_json():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    removed = {
        "imagePath": "test.png",
        "imageWidth": 100,
        "imageHeight": 100,
        "boxes": [{"label": "face", "xyxy": [10, 10, 30, 30], "reason": "min_area_ratio"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "test.png"
        cv2.imwrite(str(img_path), image)
        removed_path = Path(tmp) / "test.removed.json"
        removed_path.write_text(__import__("json").dumps(removed), encoding="utf-8")
        result = apply_mask_to_image(img_path, removed_path, "blackout", {})
    assert np.all(result[10:30, 10:30] == 0)
```

### 步骤 2：运行测试验证失败

```bash
.conda\python.exe -m pytest tests\tools\augment\test_mask_regions.py -v
```

预期：`ImportError`。

### 步骤 3：实现 `tools/augment/mask_regions.py`

```python
"""
tools/augment/mask_regions.py

读取 .removed.json（或任意带 xyxy 框的 JSON），对图片指定区域做
blackout / blur / mosaic 后处理。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import list_images


DEFAULT_INPUT_DIR = "datasets/behavior_reannotated"
DEFAULT_MASK_DIR = "datasets/behavior_reannotated"
DEFAULT_OUTPUT_DIR = "datasets/behavior_masked"
DEFAULT_MODE = "blackout"
DEFAULT_BLUR_KERNEL = 51
DEFAULT_MOSAIC_SIZE = 16


def clip_box_int(box: np.ndarray, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    x1 = max(0, min(x1, image_shape[1]))
    x2 = max(0, min(x2, image_shape[1]))
    y1 = max(0, min(y1, image_shape[0]))
    y2 = max(0, min(y2, image_shape[0]))
    return x1, y1, x2, y2


def blackout_region(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = clip_box_int(box, image.shape)
    if x2 <= x1 or y2 <= y1:
        return image
    image[y1:y2, x1:x2] = 0
    return image


def blur_region(image: np.ndarray, box: np.ndarray, kernel: int) -> np.ndarray:
    x1, y1, x2, y2 = clip_box_int(box, image.shape)
    if x2 <= x1 or y2 <= y1:
        return image
    k = max(3, kernel if kernel % 2 == 1 else kernel + 1)
    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    image[y1:y2, x1:x2] = blurred
    return image


def mosaic_region(image: np.ndarray, box: np.ndarray, block_size: int) -> np.ndarray:
    x1, y1, x2, y2 = clip_box_int(box, image.shape)
    if x2 <= x1 or y2 <= y1:
        return image
    roi = image[y1:y2, x1:x2].copy()
    h, w = roi.shape[:2]
    if block_size <= 0:
        return image
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = roi[y:y + block_size, x:x + block_size]
            color = block.mean(axis=(0, 1)).astype(np.uint8)
            roi[y:y + block_size, x:x + block_size] = color
    image[y1:y2, x1:x2] = roi
    return image


def apply_mask_to_image(
    image_path: Path,
    mask_path: Path,
    mode: str,
    kwargs: dict,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    if not mask_path.exists():
        return image
    data = json.loads(mask_path.read_text(encoding="utf-8"))
    for box_info in data.get("boxes", []):
        box = np.array(box_info["xyxy"], dtype=np.float32)
        if mode == "blackout":
            image = blackout_region(image, box)
        elif mode == "blur":
            image = blur_region(image, box, kwargs.get("kernel", DEFAULT_BLUR_KERNEL))
        elif mode == "mosaic":
            image = mosaic_region(image, box, kwargs.get("size", DEFAULT_MOSAIC_SIZE))
        else:
            raise ValueError(f"不支持的模式：{mode}")
    return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 removed.json 对图片区域做后处理")
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--mask-dir", type=Path, default=Path(DEFAULT_MASK_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=["blackout", "blur", "mosaic"], default=DEFAULT_MODE)
    parser.add_argument("--blur-kernel", type=int, default=DEFAULT_BLUR_KERNEL)
    parser.add_argument("--mosaic-size", type=int, default=DEFAULT_MOSAIC_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    images = list_images(args.input_dir, recursive=True)
    if not images:
        print(f"[错误] 未找到图片：{args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"kernel": args.blur_kernel, "size": args.mosaic_size}

    for image_path in tqdm(images, desc="区域处理"):
        mask_path = args.mask_dir / f"{image_path.stem}.removed.json"
        try:
            result = apply_mask_to_image(image_path, mask_path, args.mode, kwargs)
        except ValueError as exc:
            print(f"[跳过] {exc}", file=sys.stderr)
            continue
        out_path = args.output_dir / image_path.name
        cv2.imwrite(str(out_path), result)

    print(f"[完成] 输出目录：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 步骤 4：运行测试与帮助验证

```bash
.conda\python.exe -m pytest tests\tools\augment\test_mask_regions.py -v
.conda\python.exe tools\augment\mask_regions.py --help
```

预期：测试 PASS；帮助信息正常。

### 步骤 5：Commit

```bash
git add tools/augment/mask_regions.py tests/tools/augment/test_mask_regions.py
git commit -m "feat(augment): 新增 mask_regions.py 区域后处理工具

- 支持 blackout / blur / mosaic 三种模式
- 读取 .removed.json 中的 xyxy 框对原图做处理
- 添加单元测试"
```

---

## 任务 5：删除旧脚本与旧测试

**文件：**
- 删除：`tools/annotate/reannotate_face_hand_onnx.py`
- 删除：`tests/tools/annotate/test_reannotate_face_hand_onnx.py`

### 步骤 1：删除文件

```bash
git rm tools/annotate/reannotate_face_hand_onnx.py
git rm tests/tools/annotate/test_reannotate_face_hand_onnx.py
```

### 步骤 2：Commit

```bash
git commit -m "refactor(annotate): 删除过度特化的 reannotate_face_hand_onnx.py

功能已被通用 ONNX 重标注流程替代：
- ONNX 解码逻辑迁移到 onnx_decoder.py
- 类别级重标注由 reannotate.py 承担
- 小框涂黑由 mask_regions.py 承担"
```

---

## 任务 6：更新文档与最终验证

**文件：**
- 修改：`AGENTS.md`

### 步骤 1：在 `AGENTS.md` 常用命令中补充

在 §5.2 单脚本跑示例中加入：

```markdown
# ONNX 通用重标注（默认 dry-run 预览）
.conda\python.exe tools\annotate\reannotate.py ^
    --config tools\cfg\reannotate_onnx.yaml ^
    --input datasets\behavior ^
    --output datasets\behavior_reannotated

# 执行后处理（涂黑小框区域）
.conda\python.exe tools\augment\mask_regions.py ^
    --input-dir datasets\behavior ^
    --mask-dir datasets\behavior_reannotated ^
    --output-dir datasets\behavior_masked ^
    --mode blackout
```

### 步骤 2：运行完整回归测试

```bash
.conda\python.exe -m py_compile tools/annotate/onnx_decoder.py tools/annotate/auto.py tools/annotate/reannotate.py tools/augment/mask_regions.py
.conda\python.exe -m pytest tests\tools\ -q
```

预期：所有测试通过；若存在旧 baseline 失败，记录但无需修复。

### 步骤 3：创建配置示例 `tools/cfg/reannotate_onnx.yaml.example`

```yaml
models:
  - class: face
    action: overwrite
    model: weight/yolov5s-lmk.onnx
    decoder: yolov5s_lmk
    conf: 0.25
    iou: 0.45
    input_size: 640
    normalize: true
    min_area_ratio: 0.01

  - class: hand
    action: keep
```

### 步骤 4：端到端 dry-run（可选）

复制示例配置为 `tools/cfg/reannotate_onnx.yaml` 并针对少量图片跑 `--dry-run`。

### 步骤 5：Commit

```bash
git add AGENTS.md tools/cfg/reannotate_onnx.yaml.example
git commit -m "docs(agents): 补充 ONNX 重标注与 mask_regions 命令示例

- AGENTS.md 增加 reannotate.py / mask_regions.py 命令
- 新增 tools/cfg/reannotate_onnx.yaml.example"
```

---

## 自检

- **规格覆盖度**：每个规格章节均能找到对应任务。
- **占位符扫描**：无 TODO/待定；每个代码块均为可直接运行的实现。
- **类型一致性**：`ONNXLabeler.predict` 返回 `sv.Detections`，与 `AutoLabeler` 接口一致；`reannotate.py` 使用 `sv.Detections.merge` 与 `tools.core` 桥接函数。
- **遗留风险**：`supervision` 版本问题可能影响 `sv.Detections.with_nms` 与 `sv.Detections.merge`，需在实现时验证。
