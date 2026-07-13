# ONNX 重标注后端接入 auto.py 设计规格

## 1. 背景与目标

当前 `tools/annotate/reannotate_face_hand_onnx.py` 是一个过度特化的脚本：

- 硬编码 batch 范围 `0023–0037`。
- 硬编码人脸 ONNX 模型 `yolov5s-lmk.onnx` 及其专属解码逻辑。
- 手部检测依赖 SAM3 文本 prompt，且支持「保留现有 hand」这种与模型无关的逻辑。
- 把「小框过滤」和「图片涂黑」耦合在重标注流程里。

本设计目标：

1. 将 ONNX 推理能力通用化，接入 `tools/annotate/auto.py` 的 `AutoLabeler` 体系。
2. 新增 `tools/annotate/reannotate.py`，通过 YAML 配置实现按类别「覆盖 / 保留 / 合并」旧标注。
3. 把「小框过滤」产生的待删除区域以 sidecar 文件形式输出，供独立的 `tools/augment/mask_regions.py` 做模糊/马赛克/涂黑。
4. 原 `reannotate_face_hand_onnx.py` 在新脚本可用后删除或归档到 `archive/`。

## 2. 非目标

- 不修改 `YOLOLabeler` 与 `SAM3Labeler` 的行为。
- 不一次性支持所有 ONNX 输出格式；只支持通过注册表扩展的解码器。
- `merge` 动作第一版仅做简单并集，不引入 NMS（后续按需扩展）。
- `mask_regions.py` 第一版只支持矩形框，不支持分割 mask。

## 3. 架构

```text
┌─────────────────────────────────────────────────────────────┐
│  tools/annotate/auto.py                                     │
│  ├── YOLOLabeler                                            │
│  ├── SAM3Labeler                                            │
│  └── ONNXLabeler  ← 新增，实现 AutoLabeler.predict()        │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ 复用
┌─────────────────────────────────────────────────────────────┐
│  tools/annotate/onnx_decoder.py  ← 新增                     │
│  ├── OnnxDecoder (ABC)                                      │
│  ├── YoloV8Decoder                                          │
│  ├── YoloV5Decoder                                          │
│  └── Yolov5sLmkDecoder  ← 兼容旧人脸模型                     │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ 调用
┌─────────────────────────────────────────────────────────────┐
│  tools/annotate/reannotate.py  ← 新增                       │
│  读取 YAML → 初始化 ONNXLabeler → 按 class action 合并旧 JSON│
│  输出：新 LabelMe JSON + <图>.removed.json                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  tools/augment/mask_regions.py  ← 新增                      │
│  读取原图 + .removed.json → 模糊/马赛克/涂黑 → 输出新图      │
└─────────────────────────────────────────────────────────────┘
```

## 4. 详细设计

### 4.1 `tools/annotate/onnx_decoder.py`

#### 接口

```python
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
        ...
```

#### 注册表

```python
DECODER_REGISTRY: dict[str, type[OnnxDecoder]] = {
    "yolo_v8": YoloV8Decoder,
    "yolo_v5": YoloV5Decoder,
    "yolov5s_lmk": Yolov5sLmkDecoder,
}
```

#### 内置解码器

- `yolo_v8`：输入 `(1, 84, N)`，按 Ultralytics YOLOv8 ONNX 输出解析。
- `yolo_v5`：输入 `(1, N, 85)`，按 YOLOv5 ONNX 输出解析。
- `yolov5s_lmk`：输入 `(1, N, ?)`，兼容旧脚本 `yolov5s-lmk.onnx`：
  - `score = obj_conf * landmark_conf`
  - 前 4 维为 `xywh`，解码后转 `xyxy`。

### 4.2 `tools/annotate/auto.py` 改造

#### 新增 `ONNXLabeler`

```python
class ONNXLabeler(AutoLabeler):
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
        ...
```

行为：

1. 使用 `onnxruntime.InferenceSession` 加载模型，优先 CUDAExecutionProvider。
2. 若 `input_size` 为 `None`，从 ONNX 输入 shape 自动解析。
3. `predict(image_path)`：
   - 读取图片。
   - 缩放至 `input_size`。
   - 按需 `/255.0` 归一化。
   - 推理。
   - 调用 `DECODER_REGISTRY[decoder].decode(...)`。
   - 返回 `sv.Detections(xyxy=..., confidence=..., class_id=...)`。

#### CLI 扩展

- 新增 `--model-type onnx`。
- 新增 `--onnx-classes`：逗号分隔类别名。
- 新增 `--onnx-decoder`：解码器名称。
- 新增 `--onnx-input-size`：可选。
- 新增 `--onnx-normalize`：是否 `/255.0`，默认 `true`。
- `--conf` / `--iou` / `--model` 继续复用。

### 4.3 `tools/annotate/reannotate.py`

#### 输入参数

```bash
.conda\python.exe tools\annotate\reannotate.py \
    --config tools\cfg\reannotate_onnx.yaml \
    --input datasets\behavior \
    --output datasets\behavior_reannotated \
    --recursive \
    --dry-run
```

#### YAML 配置格式

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

字段说明：

| 字段 | 必填 | 说明 |
|---|---|---|
| `class` | 是 | 类别名，对应 LabelMe `label`。 |
| `action` | 是 | `overwrite` / `keep` / `merge`。 |
| `model` | `action != keep` 时必填 | ONNX 模型路径。 |
| `decoder` | `action != keep` 时必填 | 解码器名称。 |
| `conf` | 否 | 置信度阈值，默认 0.25。 |
| `iou` | 否 | NMS IoU 阈值，默认 0.45。 |
| `input_size` | 否 | 模型输入尺寸，默认从 ONNX 解析。 |
| `normalize` | 否 | 是否做 `/255.0`，默认 `true`。 |
| `min_area_ratio` | 否 | 最小面积占比，低于此值的框进入 `.removed.json`，不填则全部保留。 |

#### 处理流程

1. 扫描 `--input` 下所有图片（`tools.core.list_images`，`--recursive` 控制）。
2. 对每个类别 `action != keep` 的项，初始化一个 `ONNXLabeler`。
3. 遍历每张图片：
   - 找到同名 LabelMe JSON（`tools.core.find_json_for_image`）。
   - 对 `overwrite` / `merge` 类别：运行推理，得到 `sv.Detections`。
   - 按 `min_area_ratio` 拆成保留框与待删除框。
   - 对 `keep` 类别：从旧 JSON 提取该 label 的矩形框。
   - 对 `merge` 类别：旧框与新框简单合并。
   - 合并所有类别的保留框，生成新的 LabelMe JSON 并写出到 `--output`。
   - 把待删除框写入 `--output/<图>.removed.json`。
4. `dry-run` 时只统计数量，不写盘。

#### `.removed.json` 格式

```json
{
  "imagePath": "0001.png",
  "imageWidth": 1920,
  "imageHeight": 1080,
  "boxes": [
    {"label": "face", "xyxy": [10, 20, 30, 40], "reason": "min_area_ratio"}
  ]
}
```

### 4.4 `tools/augment/mask_regions.py`

#### 输入参数

```bash
.conda\python.exe tools\augment\mask_regions.py \
    --input-dir datasets\behavior \
    --mask-dir datasets\behavior_reannotated \
    --output-dir datasets\behavior_masked \
    --mode blackout
```

#### 行为

1. 扫描 `--input-dir` 图片。
2. 对每张图片找 `--mask-dir/<basename>.removed.json`。
3. 按 `boxes` 中的 `xyxy` 在原图上应用 `--mode`：
   - `blackout`：填充纯黑。
   - `blur`：高斯模糊。
   - `mosaic`：马赛克块。
4. 输出到新目录，JSON 原样复制（可选）。

## 5. 与原脚本的关系

- 新脚本上线后，`tools/annotate/reannotate_face_hand_onnx.py` 将被删除。
- 原脚本中的 `yolov5s_lmk` 解码逻辑迁移到 `tools/annotate/onnx_decoder.py`。
- 原脚本中的「小框涂黑」迁移到 `tools/augment/mask_regions.py`。

## 6. 测试计划

1. **编译检查**：`.conda\python.exe -m py_compile tools/annotate/auto.py tools/annotate/onnx_decoder.py tools/annotate/reannotate.py tools/augment/mask_regions.py`
2. **帮助信息**：`--help` 能正常显示，不触发重型依赖加载。
3. **dry-run 验证**：用示例 YAML 对少量图片跑 `--dry-run`，确认统计与旧脚本一致。
4. **端到端验证**：跑完整流程，检查输出 JSON、`.removed.json`、mask 后图片。
5. **回归测试**：`tools/annotate/auto.py` 的 YOLO / SAM3 模式行为不变。

## 7. 实现顺序

1. 创建 `tools/annotate/onnx_decoder.py` 及内置解码器。
2. 在 `tools/annotate/auto.py` 中新增 `ONNXLabeler` 与 CLI 参数。
3. 创建 `tools/annotate/reannotate.py`。
4. 创建 `tools/augment/mask_regions.py`。
5. 删除 `tools/annotate/reannotate_face_hand_onnx.py`。
6. 更新 `AGENTS.md` 常用命令与 `workflow.md`（如相关）。
7. 跑测试并提交 commit。
