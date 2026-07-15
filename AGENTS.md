# Agent 协作说明

> 本文件面向在本项目里协作的 AI 编程助手（Trae / Cursor / Claude Code 等），
> 提供项目背景、目录结构、开发约定与常用命令，避免每次都要重新交代上下文。

---

## 1. 项目一句话

`cjet-vision-pipeline` 是一个 **数据生产 + 模型训练** 流水线：把补充进来的 PNG
图像自动标注成 LabelMe JSON，做合并、清洗、拆分、YOLO 训练、自标注、归档，
最终每日交付一版可训练数据。

完整数据流见 [workflow.md](workflow.md)。

---

## 2. 关键事实速查

| 项 | 值 |
|---|---|
| 项目根目录 | `d:\PycharmProjects\cjet-vision-pipeline` |
| Python 解释器 | `.conda\python.exe`（项目自带的便携式 Python，不依赖系统 Python） |
| 依赖安装 | `.conda\python.exe -m pip install -r requirements.txt` |
| 工作流入口 | `.conda\python.exe tools\workflow.py --config src\workflow_config.yaml`（当前项目级覆盖在 `src/`；也可 `--config tools\cfg\workflow.yaml` 跑系统主工作流） |
| 配置示例 | `tools\cfg\workflow_config.yaml.example`（复制为 `src\workflow_config.yaml`） |
| 主要数据目录 | `datasets/`（raw / annotated / split_30 / split_70 / yolo 等） |
| 训练产物 | `runs/train/`、`archive/` |
| 公司专用加密工具 | 统一放在 `feat/company-encrypt` 分支的 `tools/encrypt/` 模块（`image_crypto.py` 底层库 / `encrypt.py` 加密 / `decrypt.py` 解密），经该分支 worktree（`.worktrees/company-encrypt`）单独管理，不进主仓库 `main` |

---

## 3. 目录结构

```text
cjet-vision-pipeline/
├── .conda/                        # 项目自带的便携式 Python 3.11
├── datasets/                      # 数据集（raw_jpg / raw / yolo / autolabel / behavior ...）
├── runs/                          # yolo detect train 输出
├── archive/                       # tools/train/archive.py 生成的每日归档
├── weight/                        # 训练好的 YOLO 权重
├── docs/                          # 文档与参考资料（如 superpowers/）
├── tools/                         # 业务脚本（按阶段分组，仿 ultralytics 包结构）
│   ├── __init__.py                #   顶层 API 导出
│   ├── core/                      # 公共模块
│   │   ├── constants.py           #   常量（DEFAULT_DATASET_PATH 等）
│   │   ├── images.py              #   list_images 等
│   │   ├── labelme.py             #   LabelMe JSON 扫描/读写
│   │   ├── geometry.py            #   矩形/合并/距离工具
│   │   ├── README.md              #   公共模块使用说明
│   │   └── __init__.py            #   统一对外导出
│   ├── cfg/                       # 工作流配置（仿 ultralytics/cfg）
│   │   ├── __init__.py            #   load_config / resolve_config / substitute_variables
│   │   ├── default.yaml           #   系统默认（paths / parameters / log_file）
│   │   ├── workflow.yaml          #   系统主工作流 stage 定义
│   │   ├── inherit_yolo.yaml      #   任务专项：接续 + 重命名 + 转 YOLO（走 stages_only）
│   │   ├── hand_face_mosaic.yaml  #   任务专项：备份 datasets/1|2|3 + 脸(ONNX)/手(SAM) 标注打码（走 stages_only）
│   │   ├── all_modules.yaml.example # 全部模块「用法总表」：每个 tools 脚本一个 stage，带 enabled/order
│   │   ├── detectors.yaml.example #   多检测器组合标注模板（--detectors-config 引用，见 §4.10）
│   │   └── workflow_config.yaml.example   #   项目级覆盖示例（复制为 src/workflow_config.yaml）
│   ├── engine/                    # 各 stage 聚合入口（仿 ultralytics/engine）
│   │   └── __init__.py            #   把 10 个同级 stage 包（annotate/augment/.../train）re-export 到 tools.engine 命名空间（engine 自身无子包）
│   ├── annotate/                  # 标注阶段
│   │   ├── backends/              #   检测器后端（按模型类型分离）
│   │   │   ├── base.py            #     AutoLabeler 抽象接口 + 类型别名
│   │   │   ├── onnx.py            #     ONNX 功能（OnnxDetector + YOLO 风格解码）
│   │   │   ├── sam.py             #     ultralytics SAM（SAMTextDetector / SAM3Labeler）
│   │   │   ├── yolo.py            #     ultralytics YOLO（YOLOLabeler）
│   │   │   └── detr.py            #     ultralytics DETR（DETRLabeler，RT-DETR 实现）
│   │   ├── ops.py                 #   打标/覆盖共用底层：框几何 + 打码 + LabelMe IO
│   │   ├── auto.py                #   统一标注入口：supervision 后端（YOLO/SAM3/DETR）+ ONNX 后端（两路打标 / 覆盖）+ 多检测器组合（--detectors-config 任意 N 路混搭，打码抽到 ops.apply_blackout）
│   │   └── merge.py               #   框合并
│   ├── clean/                     # 清洗阶段
│   │   ├── blurry.py              #   CleanVision 模糊检测
│   │   ├── orphan_json.py         #   孤儿 JSON
│   │   └── orphan_images.py       #   缺失 JSON 的图片
│   ├── label/                     # 标签处理
│   │   ├── align_labelme.py       #   PNG/JSON 错位一键对齐（见 §6 约束）
│   │   ├── replace.py             #   批量标签替换
│   │   └── fix_labelme.py         #   修复 LabelMe JSON 常见损坏
│   ├── split/                     # 拆分阶段
│   │   └── dataset.py             #   数据集拆分
│   ├── augment/                   # 数据增强
│   │   └── ir_enhance.py          #   IR 全局/局部光照增强 + 传感器效果
│   ├── rename/                    # 文件批量改名
│   │   └── timestamp_rename.py    #   按时间戳改名为 YYYYMMDD_HHMMSS_NNNNNN[__ms]，可选同步 LabelMe imagePath
│   ├── backup/                    # 数据备份
│   │   └── snapshot.py            #   打 .tar.gz 备份（默认 dry-run）
│   ├── merge/                     # 数据接续/合并
│   │   └── inherit_dataset.py     #   autolabel 按 1000/批接续到 behavior
│   ├── convert/                   # 格式转换
│   │   ├── jpg_to_png.py          #   JPG 批量转 PNG + 修复 LabelMe JSON
│   │   ├── labelme_to_yolo.py
│   │   └── yolo_to_labelme.py
│   ├── train/                     # 训练相关
│   │   ├── archive.py             #   按时间戳归档
│   │   └── create_tar_gz.py       #   压缩归档为 .tar.gz
│   └── workflow.py                # 工作流编排器入口（薄编排，配置解析走 tools.cfg）
├── src/                           # 本地入口 + 临时工作流（被 .gitignore 整体忽略，不入 git）
│   ├── run.py                     #   临时工作流统一入口（按需切换 cfg；默认 dry_run=True）
│   ├── main.py                    #   详细参数版入口（Python 函数式）
│   ├── run_*.py                   #   任务专项入口变体（如 run_hand_face_714.py / run_clean_behavior_to_yolo.py）
│   ├── _run_hand.py               #   内部入口脚本
│   ├── workflow_config.yaml       #   当前项目级覆盖入口（--config 传入；见 §4.9）
│   └── *.yaml                     #   临时工作流 cfg（命名：<功能>.yaml，如 clean_behavior_to_yolo.yaml / recover_yolo0708.yaml）
├── workflow.md                    # 工作流说明文档
├── requirements.txt               # 依赖清单
└── AGENTS.md                      # 本文件
```

---

## 4. 开发约定（必读）

### 4.1 语言规范

- **所有对话、解释、建议**：使用简体中文。
- **代码注释**：中文。
- **commit message**：中文（遵循 Conventional Commits：`feat:`、`fix:`、`refactor:` 等）。
- **保留英文**：API / SDK / YOLO / LabelMe / Ultralytics 等专有名词。
- 严禁出现大段未翻译的英文技术名词。

### 4.2 默认参数集中到文件顶部

每个脚本最上面必须有「默认参数」常量区（`DEFAULT_*`），`argparse` 的
`default=` 与 `help` 文本、模型加载回退值等全部引用这些常量。

参考实现见 `tools/annotate/auto.py` 顶部（`DEFAULT_ONNX_CONF` 等一串 `DEFAULT_*`），
形式如下：

```python
# 默认参数（集中放文件顶部，argparse / 模型回退统一引用）
DEFAULT_ONNX_CONF = 0.5
DEFAULT_ONNX_MIN_RATIO = 0.01
DEFAULT_MOSAIC_BLOCK = 16
```


### 4.3 优先调用官方库接口（supervision / cleanvision）

> **第一原则：能用官方库现成 API 解决的，不要自己手写。**

本项目主要依赖两个官方库，先查它们有没有现成接口，再考虑 `tools.core` 或手写：

- [supervision](https://github.com/roboflow/supervision) — 目标检测 / 分割 / 跟踪 / 数据集封装
  常用：`sv.Detections`、`sv.DetectionDataset`、`as_labelme` / `as_coco` / `as_yolo` 导出等
- [cleanvision](https://github.com/cleanlab/cleanvision) — 图像数据质量检测
  常用：`Imagelab` 找出模糊 / 异常 / 重复 / 低信息量图像

调用顺序建议：**官方库 → `tools.core` → 手写**。
- 官方库有现成 API：直接用，**不要自己造轮子**；
- 官方库没提供但 `tools.core` 已经抽取过：import 后用；
- 都没有：才考虑手写，并按"等真的出现第二次重复时再抽函数"的原则，必要时回填到 `tools.core`。

`tools/core/__init__.py` 已经统一导出了 `DEFAULT_DATASET_PATH`、`list_images`、
`list_labelme_files`、`load_labelme`、`save_labelme`、`find_image_for_json`、
`find_json_for_image`、`rect_to_xyxy`、`xyxy_to_points`、`merge_near_boxes` 等。

新脚本需要扫描图片 / 读写 LabelMe JSON / 几何运算时，**直接 import**，不要再
手写：

```python
from tools.core import (
    DEFAULT_DATASET_PATH,
    list_images,
    list_labelme_files,
    load_labelme,
    save_labelme,
    merge_near_boxes,
)
```

### 4.4 路径保护

`tools/` 下的脚本顶部都必须有这一段，保证 `python tools/xxx.py` 直接运行和
`pytest tests/tools/` 都能正常 import：

```python
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

### 4.5 类别最小够用

本项目当前目标类别是 `face / person / phone` 等少数几个，不要过度扩展类别名。

### 4.6 不做无谓的"防御性"代码

- 不要添加 `try/except` 包住不会出错的逻辑；
- 不要给内部函数写多余的参数校验；
- 不要为了"未来可能用到"而新增配置项 / 参数 / 抽象层。

按需重构：等真的出现第二次重复时再抽函数。

### 4.7 改动前先看相关文件

- 改 `tools/annotate/` 下的打标 / 覆盖脚本前，先看 `tools/core/` 公共模块；
  其中：
  - 检测器后端（onnx / sam / yolo / detr）改 `tools/annotate/backends/` 对应文件；
  - 框几何 / 打码 / LabelMe IO 等共用底层改 `tools/annotate/ops.py`；
  - 编排层（`auto.py`）只做流程拼装，
    不要在这些文件里重新实现后端或底层几何逻辑；
- 改工作流前先读 [workflow.md](workflow.md) 和 [workflow_config.yaml.example](workflow_config.yaml.example)；
- 改任何公共逻辑时，**同步检查是否有别的脚本调用了旧 API**。

### 4.8 何时用哪个库（决策树）

写新功能时按下面顺序判断：

1. **官方库有现成 API？** → 用官方库（[supervision](https://github.com/roboflow/supervision) / [cleanvision](https://github.com/cleanlab/cleanvision)）。
2. **官方库有但太通用、调用啰嗦？** → 看 `tools/core/` 是否已经包了一层；有就 import，没有再考虑手写。
3. **都没有？** → 手写，并写完后再判断是否值得提到 `tools/core/`。

反面例子（不要做）：

- 自己手写 `labelme2yolo` 转换器（`supervision.as_yolo` 已经实现）；
- 自己用 `cv2` 检测模糊图像（`cleanvision.Imagelab` 已经有专门接口）；
- 自己用 `numpy` 计算 IOU / NMS（`supervision` 已有 `box_iou`、`non_max_suppression`）。

### 4.9 临时工作流规范（任务专项 cfg）

一次性 / 临时 / 任务专项工作流**不**写到 `tools/cfg/`，而是：

- **cfg 放 `src/`**（被 `.gitignore` 整体忽略，**不入 git**）。
- **优先用 `stages_only` 机制** 引用 `tools/cfg/workflow.yaml` 已有 stage
  （参考 `tools/cfg/inherit_yolo.yaml` 风格）。只有在现有 stage 拼不出来时，
  才在临时 cfg 里**直接定义 stage 命令**（不走 workflow.yaml）。
- **不写一次性专用 Python 脚本**——除非该功能**明确**有未来通用需求
  （按 §4.6 "等真的出现第二次重复时再抽函数"原则，单次需求**不**算通用）。
- **入口用 `src/run.py`**：要切 cfg / 改参数直接编辑 `run()` 函数体
  （默认 `dry_run=True` 安全预览）。Python 函数式入口用 `src/main.py`。
- **命名约定**：临时 cfg 用 `<功能>.yaml`（如 `src/recover_yolo0708.yaml`）。
  入口脚本是否新建 `<功能>.py` / `src/run_<功能>.py` **按需求询问创建**：
  默认复用 `src/run.py` 切换 cfg 即可；如下游有独立参数 / 独立 dry_run
  默认值 / 多入口并存需求，再询问用户是否新建。

### 4.10 annotate 模块分层约定

`tools/annotate/` 把「检测器后端」与「打标 / 标签覆盖编排」彻底分离，三层职责如下：

1. **`backends/` —— 检测器后端（按模型类型分文件）**
   - `base.py`：`AutoLabeler` 抽象接口 + 类型别名（`DetectionsLike` / `DatasetLike`）。
   - `onnx.py`：`OnnxDetector`（ONNX 推理 + YOLO 风格预处理 / 解码 / execution provider）。
   - `sam.py`：`SAMTextDetector`（numpy 框）、`SAM3Labeler`（返回 `sv.Detections`）。
   - `yolo.py`：`YOLOLabeler`（ultralytics YOLO）。
   - `detr.py`：`DETRLabeler`（ultralytics RT-DETR，与 YOLOLabeler 接口一致）。
   - 每个后端只负责「加载模型 + `predict()` 出检测结果」，不碰框几何 / 打码 / 文件 IO。

2. **`ops.py` —— 打标 / 覆盖共用底层**
   - 框几何：`clip_box`（裁剪越界框）/ `classify_box_by_ratio`（按面积占比切保留·删除）/
     `subtract_box_regions` / `collect_blackout_regions` / `concat_boxes` / `extract_existing_label_boxes`。
   - 打码：`mosaic_region` / `collect_blackout_regions` / `blackout_region` / `rewrite_labelme_dict`。
   - 两类编排都从这里 import，不要在编排层重复实现。

3. **编排层（顶层）**
   - `auto.py`：统一标注入口，三条并存的路：
     1. `--model-type yolo|sam3|detr`：走 supervision 数据集式导出（YOLO/LabelMe/COCO）；
     2. `--model-type onnx`：走 ONNX 后端——ONNX 一路（可选 SAM 第二路）两路打标，或加
        `--reannotate` 覆盖指定类别并保留其它类别；
     3. `--detectors-config <yaml>`：**多检测器组合**（任意 N 路混搭、同类型可多路，
        如两个 YOLO / 两个 ONNX / onnx+sam+yolo / detr+onnx+sam+yolo）。每路各自
        推理出「框 + 逐框标签」，统一走「合并保留大框 -> 逐路小框打码（重叠保护）->
        输出 LabelMe」链路。检测器与全局项由 YAML 描述（cfg 放 `src/`，不入 git；
        模板见 `tools/cfg/detectors.yaml.example`）。
     打码统一由 `ops.apply_blackout` 完成；多路合并链路是 `_run_multi_source_annotation`。
   - `merge.py`：框合并。

新增检测器后端时，**只在 `backends/` 加一个文件并实现 `AutoLabeler` 接口**，再在编排层按需调用；
若要接入多检测器组合，只需在 `auto._build_detectors` 里加一个 `type` 分支，把后端包成
`detect(image, image_path) -> (框, 逐框标签)`，**不要**把新后端的推理逻辑塞进 `ops.py`
或某个编排脚本里。

---

## 5. 常用命令速查

> 所有命令都在项目根目录 `d:\PycharmProjects\cjet-vision-pipeline` 下执行。

### 5.1 安装 / 检查依赖

```bash
# 装全部依赖
.conda\python.exe -m pip install -r requirements.txt

# 只装核心（supervision / cleanvision）
.conda\python.exe -m pip install supervision cleanvision

# 看当前版本
.conda\python.exe -c "import supervision, cleanvision; print(supervision.__version__, cleanvision.__version__)"
```

### 5.2 单脚本跑

```bash
# JPG 批量转 PNG（自动修复同名 LabelMe JSON）
.conda\python.exe tools\convert\jpg_to_png.py ^
    --input E:\czz\0024 ^
    --output E:\czz\0024\PNG ^
    --num-threads 16

# 自动标注（YOLO）
.conda\python.exe tools\annotate\auto.py ^
    --model-type yolo ^
    --source datasets\raw ^
    --output datasets\01_annotated ^
    --format labelme

# 通用标注：ONNX 标一路 + SAM 文本 prompt 标一路，LabelMe 输出；
# 类别/模型完全由参数决定（示例：ONNX 标 face + SAM 标 hand）；
# 面积<1% 的小框打马赛克后删除，重叠保留大框的部分不打码（默认预览）
.conda\python.exe tools\annotate\auto.py --model-type onnx ^
    --source datasets\raw ^
    --output datasets\01_annotated_onnx_sam ^
    --onnx-model weight\yolov5s-lmk.onnx --onnx-label face ^
    --sam-model weight\sam3.1_multiplex.pt --sam-prompt hand --sam-label hand ^
    --onnx-min-ratio 0.01 --sam-min-ratio 0.01 --mosaic-block 16
.conda\python.exe tools\annotate\auto.py --model-type onnx ^
    --source datasets\raw ^
    --output datasets\01_annotated_onnx_sam ^
    --onnx-model weight\yolov5s-lmk.onnx --onnx-label face ^
    --sam-model weight\sam3.1_multiplex.pt --sam-prompt hand --sam-label hand ^
    --onnx-min-ratio 0.01 --sam-min-ratio 0.01 --mosaic-block 16 --apply

# 该工作流也支持用 tools/workflow.py 跑（临时 cfg 在 src/，不入 git）：
#   python -m tools.workflow --config src\onnx_sam_mosaic.yaml --dry-run
#   python -m tools.workflow --config src\onnx_sam_mosaic.yaml

# 多检测器组合标注：任意 N 路混搭（onnx/sam/yolo，可混搭、同类型可多路），
# 全部检测器与全局项由 YAML 描述（cfg 放 src/，不入 git；模板见
# tools/cfg/detectors.yaml.example）。默认预览，加 --apply 才写盘。
.conda\python.exe tools\annotate\auto.py --detectors-config src\detectors.yaml
.conda\python.exe tools\annotate\auto.py --detectors-config src\detectors.yaml --apply

# 框合并
.conda\python.exe tools\annotate\merge.py ^
    --json-dir datasets\01_annotated ^
    --merge-distance-x 100 ^
    --merge-distance-y 200

# CleanVision 模糊检测
.conda\python.exe tools\clean\blurry.py ^
    --dataset-path datasets\01_annotated ^
    --output-path datasets\_blurry ^
    --blurry-threshold 0.185 ^
    --recursive --cleanup-crop

# 数据集拆分（默认 30/70）
.conda\python.exe tools\split\dataset.py ^
    --input datasets\01_annotated ^
    --output-a datasets\02_high_quality_30 ^
    --output-b datasets\02_remaining_70 ^
    --ratio 0.3 --seed 42

# IR 图像增强（默认对所有类别做局部增强；可指定 --local-light-classes 0,1）
.conda\python.exe tools\augment\ir_enhance.py ^
    --input-dir datasets\raw ^
    --output-dir datasets\raw_ir_aug ^
    --repeat 3 ^
    --workers 8

# 按时间戳批量改名（默认 dry-run 预览；加 --apply 才会真改）
.conda\python.exe tools\rename\timestamp_rename.py ^
    --source-dir D:\photos

.conda\python.exe tools\rename\timestamp_rename.py ^
    --source-dir D:\photos ^
    --apply

# 按时间戳改名 + 同步同名 LabelMe JSON 的 imagePath（LabelMe 数据集推荐）
.conda\python.exe tools\rename\timestamp_rename.py ^
    --source-dir datasets\behavior ^
    --labelme-sync ^
    --apply

# 数据备份：把 autolabel + behavior 打成 .tar.gz 到 C:\Users\EDY\Pictures（默认 dry-run）
.conda\python.exe tools\backup\snapshot.py ^
    --sources datasets\autolabel,datasets\behavior ^
    --output-dir C:\Users\EDY\Pictures

# autolabel 按 1000/批接续到 behavior/0023/（默认 dry-run；按 JSON label 归类到 8 个子目录）
.conda\python.exe tools\merge\inherit_dataset.py ^
    --source datasets\autolabel ^
    --target datasets\behavior ^
    --batch-size 1000

# 修复 LabelMe JSON 常见损坏（默认 dry-run；imagePath / 矩形 / 尺寸 等）
.conda\python.exe tools\label\fix_labelme.py ^
    --root datasets\behavior ^
    --recursive

# 转 YOLO 格式（按 batch 分子目录，默认 9:1 划分 train/val；默认 out 是 datasets/yolo）
.conda\python.exe tools\convert\labelme_to_yolo.py ^
    --src datasets\behavior ^
    --out datasets\yolo ^
    --classes phone,cigarette,face,hand ^
    --ratios 0.9,0.1
```

### 5.3 跑整个工作流

工作流配置全部在 `tools/cfg/` 下，结构：

- `tools/cfg/default.yaml`：系统默认（paths / parameters / log_file）
- `tools/cfg/workflow.yaml`：系统主工作流（20 个 stage）
- `tools/cfg/workflow_config.yaml`：`resolve_config()` 的默认查找路径（当前实际覆盖文件在 `src/workflow_config.yaml`，经 `--config` 传入）
- `tools/cfg/workflow_config.yaml.example`：项目级覆盖示例
- `tools/cfg/inherit_yolo.yaml`：任务专项精简版（接续 + 重命名 + 转 YOLO，5 个 stage）

`tools/workflow.py` 经 `--config` 加载项目覆盖（当前为 `src/workflow_config.yaml`），自动
叠加 `default.yaml` + `workflow.yaml` + 项目覆盖三层（`resolve_config` 的默认查找路径是 `tools/cfg/workflow_config.yaml`）。

**任务专项精简工作流**（`stages_only` 机制）：

如果只想跑主工作流中的某几个 stage，在 project 顶层加 `stages_only: [name1, name2, ...]`
字段即可——列出的 stage 按顺序从 `workflow.yaml` 拿完整定义，其它 stage 全部跳过。
详见 `tools/cfg/inherit_yolo.yaml`。

**每个 stage 的两个开关：**
- `enabled: true/false` —— 是否启用该 stage（`false` 跳过，默认 `true`）。
- `order: <整数>` —— 运行顺序。`order` 相同的 stage 会**并行**同时启动（组内彼此独立）；
  不同 `order` 按整数从小到大**顺序**执行。不写 `order` 的 stage 退化为各自独立成组、保序执行。
  适合「两个互不依赖的模块一起跑」的场景，例如让 `jpg_to_png` 与 `train_yolo` 都标 `order: 1`、
  把 `fix_labelme` 标 `order: 2`，即你给的写法 `1、jpg_to_png 1、train_yolo  2、fix_labelme`。

**配置合并规则**：`resolve_config()` 把 `default.yaml` + `workflow.yaml` + 项目层按 name 做
字段级合并——项目层后加载，可给同名 stage 追加/覆盖 `order` / `enabled` / `command`，
其余键（如 `workflow.yaml` 提供的 `command`）保留。示例见 `tools/cfg/all_modules.yaml.example`
（覆盖全部模块、可直接 `--config` 预览的「用法总表」）。

```bash
# 预览整条链路（不真跑）
.conda\python.exe tools\workflow.py --dry-run

# 跑第零段（接续段：备份 + autolabel 接续 + JSON 修复 + 重命名）
.conda\python.exe tools\workflow.py --from-stage backup_snapshot --to-stage convert_to_yolo

# 跑第一段（准备段：到人工清洗检查点停）
.conda\python.exe tools\workflow.py --from-stage auto_annotate

# 跑第二段（训练段：人工清洗完后再跑）
.conda\python.exe tools\workflow.py --from-stage select_subset

# 任务专项精简版（一键跑接续 + 重命名 + 转 YOLO，5 个 stage）
.conda\python.exe tools\workflow.py --config tools\cfg\inherit_yolo.yaml

# 显式指定其它入口（如直接跑系统主工作流，跳过项目覆盖）
.conda\python.exe tools\workflow.py --config tools\cfg\workflow.yaml --dry-run
```

**临时工作流**（cfg 在 `src/`，不入 git）：

```bash
# 临时 cfg 预览（src/run.py 已设好 cfg + 默认 dry_run=True）
python -m src.run

# 临时 cfg 真写盘：改 src/run.py 的 dry_run=False 再跑
# 或直接用 workflow CLI 透传：
python -m tools.workflow --config src\recover_yolo0708.yaml --from-stage inherit_yolo0708
```

### 5.4 专项工作流：备份 datasets\1|2|3 + 脸(ONNX)/手·手机·香烟(SAM) 标注打码

> 已固化为正式 cfg：`tools/cfg/hand_face_mosaic.yaml`（走 `stages_only`，只跑 4 个
> stage，不会触发主工作流的其它 20 个）。直接用 `tools/workflow.py` 编排即可：
>
> ```bash
> # 预览（snapshot / auto 各自 dry-run，不写盘）
> python tools/workflow.py --config tools/cfg/hand_face_mosaic.yaml --dry-run
>
> # 真跑：先确认 --dry-run 无误，再给 cfg 里各 stage 的 command 末尾加 --apply
> python tools/workflow.py --config tools/cfg/hand_face_mosaic.yaml
> ```
>
> 需求：先备份 `datasets\1`、`datasets\2`、`datasets\3`，再用
> `yolov5s-lmk.onnx` 标「脸(face)」、SAM 文本 prompt 标「手(hand)」，
> 输出 LabelMe；面积 < 图片 1% 的小框先打马赛克再删除，**但与大框重叠的
> 部分不打码**（重叠保护由 `ops.apply_blackout` 保证）。
>
> 用现有脚本即可拼出，无需新写 Python；遵循 §4.9，如需 workflow 编排，
> 临时 cfg 放 `src/`（不入 git）。以下命令都在项目根目录执行。

**第 1 步：备份 `datasets\1,2,3`**（默认 dry-run，确认后加 `--apply`）：

```bash
# 预览
.conda\python.exe tools\backup\snapshot.py ^
    --sources datasets\1,datasets\2,datasets\3 ^
    --output-dir C:\Users\EDY\Pictures

# 真打包（三个目录各生成一个带时间戳的 .tar.gz）
.conda\python.exe tools\backup\snapshot.py ^
    --sources datasets\1,datasets\2,datasets\3 ^
    --output-dir C:\Users\EDY\Pictures ^
    --apply
```

**第 2 步：ONNX 标脸 + SAM 标手，小框打码删除（重叠保护）**。
`auto.py --source` 只吃单个目录，`datasets\1|2|3` 各跑一次（默认预览，
去掉 `--apply` 先 dry-run，确认无误再加 `--apply`）：

```bash
# datasets\1（datasets\2、datasets\3 把 --source / --output 换成对应目录即可）
# SAM 用逗号分隔同时标 hand / phone / cigarette
.conda\python.exe tools\annotate\auto.py --model-type onnx ^
    --source datasets\1 ^
    --output datasets\1_annotated ^
    --onnx-model weight\yolov5s-lmk.onnx --onnx-label face ^
    --sam-model weight\sam3.1_multiplex.pt --sam-prompt hand,phone,cigarette --sam-label hand,phone,cigarette ^
    --onnx-min-ratio 0.01 --sam-min-ratio 0.01 --mosaic-block 16 --apply

.conda\python.exe tools\annotate\auto.py --model-type onnx ^
    --source datasets\2 --output datasets\2_annotated ^
    --onnx-model weight\yolov5s-lmk.onnx --onnx-label face ^
    --sam-model weight\sam3.1_multiplex.pt --sam-prompt hand,phone,cigarette --sam-label hand,phone,cigarette ^
    --onnx-min-ratio 0.01 --sam-min-ratio 0.01 --mosaic-block 16 --apply

.conda\python.exe tools\annotate\auto.py --model-type onnx ^
    --source datasets\3 --output datasets\3_annotated ^
    --onnx-model weight\yolov5s-lmk.onnx --onnx-label face ^
    --sam-model weight\sam3.1_multiplex.pt --sam-prompt hand,phone,cigarette --sam-label hand,phone,cigarette ^
    --onnx-min-ratio 0.01 --sam-min-ratio 0.01 --mosaic-block 16 --apply
```

参数含义：`--onnx-min-ratio` / `--sam-min-ratio 0.01` 即「面积 < 1% 判为小框」的
阈值；`--mosaic-block 16` 是马赛克块大小。「小框打码后删除、重叠大框部分不打码」
是该链路的内置行为（见 §4.10）。

### 5.5 测试

> 测试统一收在 `tests/` 下：`tests/` 根目录放独立测试（如 `test_torch_cuda.py` 校验
> torch/CUDA、`test_onnx_execution_provider.py` 校验 onnxruntime 能否在 CPU/GPU 上
> 真实推理），`tests/tools/<stage>/` 放按阶段归类的测试。pytest 递归收集，
> 直接跑 `tests/` 即可覆盖全部。

```bash
.conda\python.exe -m pytest tests\ -q
```

---

## 6. 已知约束 / 坑

- **`supervision.dataset.formats.labelme` 缺失**：`tests/` 下与 LabelMe 导出
  相关的测试会因为当前环境装的 `supervision` 版本缺少这个模块而失败。这与本次
  项目重构无关，是 baseline 问题。

- **torch / CUDA 测试**：`tests/test_torch_cuda.py` 校验 torch 可导入与 CUDA
  可用性；无 GPU 环境用 `pytest.skip` 跳过 GPU 相关断言，不报错。

- **ONNX CPU/GPU 测试**：`tests/test_onnx_execution_provider.py` 用真实
  `onnxruntime` 验证模型可在 `CPUExecutionProvider` 上推理，存在
  `CUDAExecutionProvider` 时再验证 GPU 推理；缺 `onnxruntime` 时整体
  `pytest.importorskip` 跳过，无 CUDA 时 GPU 用例 `pytest.skip` 跳过。

- **目录名带下划线**：`datasets/01_annotated` 这种带数字前缀的目录是为了让
  `ls` 时能按阶段顺序排列，不要随意改。

- **`--recursive` 默认行为不一致**：`tools/clean/` 下脚本默认递归，
  `tools/annotate/auto.py` 默认不递归（为了性能）。新增脚本时按场景决定
  并在 `--help` 中明确写出。

- **YOLO 输出目录结构**：`tools/annotate/auto.py --format yolo` 默认输出
  `images/` + `labels/` 分离结构；`tools/convert/yolo_to_labelme.py` 的
  `--labels` 要指向 `labels/`。

- **`src/` 会被插入 `sys.path`**：`tools/annotate/auto.py` 顶部会把项目 `src/`
  目录插入 `sys.path` 首位。若日后在 `src/` 下放置本地开发版库（如 `src/supervision/`），
  直接运行脚本即可覆盖安装版生效，无需重装。（当前 `src/` 下尚无此类覆盖目录。）

- **rename 不动 JSON 的 mtime 排序**：`tools/rename/timestamp_rename.py`
  只对 PNG/JPG 按 mtime 排序改名；LabelMe JSON 不参与排序（它的 mtime
  是 `inherit` / `auto-annotate` 的生成时间，跟 PNG 原始拍照时间天然
  错位）。同名 JSON 会跟着 PNG 一起改名为 `<新图名>.json`，并同步
  `imagePath` 字段。

- **labelme_to_yolo 没有 `--ratios`**：脚本默认 `DEFAULT_RATIOS = (0.9, 0.1)`
  硬编码，未暴露成 CLI 参数。`tools/cfg/workflow.yaml` 的 `convert_to_yolo`
  stage 不要传 `--ratios`，会报错。

- **PNG/JSON 错位补救**：如果历史批次 PNG 和 JSON 已经错位（JSON 文件名
  和 imagePath 指向的 PNG basename 不一致），跑
  `python tools/label/align_labelme.py --root <批次目录> --apply` 一键对齐。

- **临时工作流 cfg 放 `src/`**（被 `.gitignore` 整体忽略，不入 git）：
  不要往 `tools/cfg/` 写临时 cfg（避免污染主工作流）。`src/run.py` 是临时
  工作流统一入口，按需切换 cfg；改 cfg 字符串 / `dry_run` 即可跑不同任务。
  详见 §4.9 临时工作流规范。

- **`.worktrees` 工作流（git worktree）**：用于在独立工作目录并行开发，不干扰
  `main`。`.worktrees/` 已被 `.gitignore` 忽略（第 50 行），worktree 目录
  不进历史。
  - 公司专用加密工具已从 `main` 抽离，重构成 `feat/company-encrypt` 分支的
    `tools/encrypt/` 模块（`image_crypto.py` 底层库 / `encrypt.py` 加密 /
    `decrypt.py` 解密），worktree 在 `.worktrees/company-encrypt/` 并提交。
  - 进入工作区：`cd .worktrees/company-encrypt`；创建新 worktree：
    `git worktree add -b <分支名> .worktrees/<目录名> main`。
  - 该分支 `tools/cfg/encrypt.yaml` 是加密/解密专项工作流
    （`stages_only` 只跑两阶段）；`main` 工作区本身不含这些文件。

---

## 7. AI 助手任务执行 checklist

收到任务后请按以下顺序思考：

1. **明确目标**：是要改脚本？调工作流？加新阶段？修 bug？补测试？
2. **定位文件**：根据上面的目录结构找到相关脚本，**先读再改**。
3. **检查调用方**：修改公共逻辑时 `Grep` 一下 `from tools.core import` 看影响范围。
4. **写代码**：
   - 顶部加默认参数常量；
   - 业务逻辑放 `tools/<stage>/` 下；
   - 可复用工具放 `tools/core/` 并在 `__init__.py` 导出。
5. **验证**：
   - `.conda\python.exe -m py_compile <file>`；
   - `.conda\python.exe <file> --help`；
   - `.conda\python.exe -m pytest tests\ -q`；
   - 工作流改动用 `--dry-run` 验证。
6. **同步文档**：改了工作流 / 加了新阶段，更新 [workflow.md](workflow.md) 和
   [workflow_config.yaml.example](workflow_config.yaml.example)。

---

## 8. Git 提交规范

> **硬性要求：每次代码或文档变动都要做一次 `git commit`，不能攒到一起。**

### 8.1 流程

1. 改完代码或文档后，先 `git status --short` 探查实际改动文件。
2. **精准 add**：用 `git add <具体文件>` 逐个添加（**不要** `git add -A` /
   `git add .`，避免误把 `.env`、凭据、临时文件带进仓库）。
3. 中文 commit message 先写到临时文件（推荐 `_commit_msg.txt`，已加进
   `.gitignore`），再 `git commit -F _commit_msg.txt`。
4. **不要** 改 `git config`（user.name / user.email 保持仓库原值）。
5. 跑完 `git log --oneline -1` 确认提交成功。
6. 验证乱码：PowerShell 默认 GBK，会把 commit message 显示成乱码。提交后用
   `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` 强制 UTF-8 即可
   正常显示。仓库里存的 message 本身就是 UTF-8，无需重写。

### 8.2 Commit message 格式

遵循 [Conventional Commits](https://www.conventionalcommits.org/)，**标题与正文都用简体中文**：

```text
<type>(<scope>): <中文一句话标题，不超过 50 字>

<正文段落，72 字换行，说明 "为什么" 而不是 "做了什么">
- 改动点 1
- 改动点 2
```

允许的 `type`：

| type | 用途 |
|---|---|
| `feat` | 新功能 / 新脚本 / 新工作流 stage |
| `fix` | 修 bug |
| `refactor` | 重构（不改变行为） |
| `perf` | 性能优化 |
| `docs` | 仅文档（AGENTS.md / workflow.md 等） |
| `test` | 测试相关 |
| `chore` | 杂项（依赖、.gitignore、目录结构微调） |
| `style` | 格式调整（不影响逻辑） |

示例：

```text
feat: 新增 LabelMe JSON 修复脚本

扫描并修复常见损坏：imagePath 指向已不存在的图片、矩形 points 拍平、
x1>x2 / y1>y2、imageWidth/Height 与图片实际尺寸不一致、缺顶层字段。
默认 dry-run，加 --apply 才会真改。

- 新建 tools/label/fix_labelme.py
- 默认递归扫描 --root 下的所有 *.json
- 可选 --remove-orphan 删除缺同名图片的孤立 JSON
```

### 8.3 工作流命令速查

```bash
# 探查未提交改动
git status --short

# 精准 add（按文件）
git add tools/merge/inherit_dataset.py workflow_config.yaml.example

# 写 message 到文件（避免 PowerShell HEREDOC 解析失败）
@'
feat: 新增继承数据集脚本

把 autolabel 按 1000/批接续到 behavior/0023/...
'@ | Out-File -FilePath _commit_msg.txt -Encoding utf8

# 提交
git commit -F _commit_msg.txt

# 验证（强制 UTF-8 解决 PowerShell 中文乱码）
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
git log --oneline -5
```

### 8.4 禁止事项

- ❌ `git add -A` / `git add .`
- ❌ `git config --global ...` 或任何改 `git config` 的命令
- ❌ `git push`（除非用户明确说"push"）
- ❌ `git commit --allow-empty`（无意义空提交）
- ❌ 提交 `.env`、`*.key`、`*.pem`、`credentials.json` 等敏感文件
- ❌ 提交 `datasets/`、`runs/`、`weight/`、`archive/`、`__pycache__/`、
      `.conda/`（已在 `.gitignore` 里，但偶尔会绕过，遇到要主动排除）

---

## 9. 用户偏好

来自 `user_profile.md`（摘录）：

- 沟通语言：中文。
- 工作领域：计算机视觉（DSM 等）、GUI 应用程序调试。
- 技术栈：Python、AI 模型集成（Ultralytics/YOLO）、Supervision、PyQt5。
- 协作风格：
  - 倾向于先审阅详细设计文档和计划再实现；
  - 偏好轻量模块化架构而非单文件脚本；
  - 偏好调用官方库标准化接口而非手写解析逻辑；
  - 习惯在带未提交改动的"非干净"目录中工作。
