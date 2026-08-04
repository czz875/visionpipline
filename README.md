# Vision Pipeline

**数据生产 + 模型训练流水线**：把补充进来的 PNG 图像自动标注成 LabelMe JSON，依次经过框合并、自动清洗、人工清洗检查点、数据集拆分、YOLO 训练、自标注与归档，最终每日交付一版可训练数据。

项目结构仿 [ultralytics](https://github.com/ultralytics/ultralytics)：业务脚本按阶段分组放在 `tools/<stage>/`，YAML 配置集中在 `tools/cfg/`，由 `tools/workflow.py` 统一编排。

> 详细文档见 [workflow.md](workflow.md)（数据流与快速开始）与 [AGENTS.md](AGENTS.md)（目录结构、开发约定、常用命令）。

---

## 特性

- **多后端自动标注**：YOLO / SAM3 / DETR / ONNX 四种检测器后端，统一入口 `tools/annotate/auto.py`。
- **ONNX 两路打标 + 打码覆盖**：ONNX 标一路 + SAM 文本 prompt 标一路，面积小于阈值的小框先打马赛克再删除，与大框重叠的部分不打码（重叠保护）。
- **多检测器组合**：`--detectors-config` 引用 YAML，任意 N 路检测器混搭（同类型可多路），统一「合并保留大框 → 逐路小框打码 → 输出 LabelMe」链路。
- **工作流编排器**：YAML 描述 stage，支持 `${paths.xxx}` / `${prev.xxx}` / `${latest:glob}` 变量替换、`--dry-run` 预览、`--from-stage` 断点续跑，自动记录日志到 `workflow.log`。
- **三段式工作流**：数据接续 → 自动准备（到人工清洗检查点停）→ 训练交付，方便每日定时跑。
- **官方库优先**：supervision（检测 / 数据集封装）、cleanvision（模糊检测）有现成 API 一律复用，不自造轮子。
- **便携式 Python**：项目自带 `.conda\python.exe`（Python 3.11），不依赖系统环境。

---

## 整体链路

```text
补充 PNG 数据 / JPG 原始数据
    │
    ├──► tools/convert/jpg_to_png.py       JPG → PNG（可选）
    ▼
多后端自动标注（YOLO / SAM3 / DETR / ONNX）──► LabelMe JSON
    │
    ▼
tools/annotate/merge.py                    标注框合并
    │
    ▼
自动清洗
    ├── tools/clean/blurry.py              模糊检测并移出
    ├── tools/clean/orphan_json.py         孤儿 JSON
    └── tools/clean/orphan_images.py       缺失 JSON 的图片
    │
    ▼
【人工清洗检查点】在 LabelMe 中检查并修正自动阶段结果
    │
    ▼
tools/split/dataset.py                     随机拆分为 30% 高质量 + 70% 剩余
    │
    ├──► tools/convert/labelme_to_yolo.py  30% 转 YOLO
    │       │
    │       ▼
    │   yolo detect train                  训练 SOTA 模型
    │       │
    │       ▼
    │   tools/annotate/auto.py             自标注剩余 70%
    │       │
    │       ▼
    │   tools/convert/yolo_to_labelme.py   回 LabelMe
    │       │
    └───────┘
            ▼
    tools/clean/orphan_json.py             最终清洗
            ▼
    tools/train/archive.py                 每日归档
```

---

## 目录结构

```text
visionpipline/
├── .conda/               # 项目自带的便携式 Python 3.11
├── tools/                # 业务脚本（按阶段分组）
│   ├── core/             #   公共模块：LabelMe 读写 / 几何工具 / 常量
│   ├── engine/           #   各 stage 聚合入口（re-export）
│   ├── annotate/         #   自动标注：backends（yolo/sam/detr/onnx）+ 编排 + 框合并
│   ├── clean/            #   清洗：模糊检测 / 孤儿 JSON
│   ├── label/            #   标签处理：对齐 / 替换 / 修复 LabelMe JSON
│   ├── split/            #   数据集拆分（默认 30/70）
│   ├── convert/          #   格式转换：JPG→PNG / LabelMe↔YOLO
│   ├── merge/            #   数据接续（autolabel 按批并入 behavior）
│   ├── rename/           #   按时间戳批量改名 + 同步 imagePath
│   ├── backup/           #   数据备份（.tar.gz）
│   ├── augment/          #   数据增强（IR 光照增强）
│   ├── train/            #   训练产物归档 / 打包
│   ├── cfg/              #   工作流配置（default / workflow / 任务专项）
│   └── workflow.py       #   工作流编排器入口
├── datasets/             # 数据集（raw / annotated / split_30 / yolo ...，不入 git）
├── weight/               # 训练好的 YOLO / SAM 权重（不入 git）
├── runs/                 # yolo detect train 输出（不入 git）
├── tests/                # 测试（pytest 递归收集）
├── src/                  # 本地入口 + 临时工作流 cfg（不入 git）
├── requirements.txt      # 依赖清单
├── workflow.md           # 工作流说明文档
└── AGENTS.md             # 协作说明 / 开发约定
```

---

## 快速开始

### 环境要求

- Python 3.11（推荐使用项目自带的 `.conda\python.exe`，无需安装系统 Python）
- 可选：NVIDIA GPU + CUDA 12.x（训练 / ONNX GPU 推理更快；无 GPU 时自动回退 CPU）

### 安装依赖

```bash
.conda\python.exe -m pip install -r requirements.txt
```

安装后冒烟测试：

```bash
.conda\python.exe -c "import supervision, cleanvision, ultralytics; print(supervision.__version__, cleanvision.__version__, ultralytics.__version__)"
```

### 基本用法

**自动标注（YOLO → LabelMe）：**

```bash
.conda\python.exe tools\annotate\auto.py --model-type yolo ^
    --source datasets\raw ^
    --output datasets\01_annotated ^
    --format labelme
```

**ONNX 标 face + SAM 标 hand，小框打码删除（默认预览，加 `--apply` 写盘）：**

```bash
.conda\python.exe tools\annotate\auto.py --model-type onnx ^
    --source datasets\raw ^
    --output datasets\01_annotated_onnx_sam ^
    --onnx-model weight\yolov5s-lmk.onnx --onnx-label face ^
    --sam-model weight\sam3.1_multiplex.pt --sam-prompt hand --sam-label hand ^
    --onnx-min-ratio 0.01 --sam-min-ratio 0.01 --mosaic-block 16
```

**多检测器组合标注（配置见 `--detectors-config`）：**

```bash
.conda\python.exe tools\annotate\auto.py --detectors-config src\detectors.yaml
```

**跑整条工作流：**

```bash
# 预览（不真跑）
.conda\python.exe tools\workflow.py --dry-run

# 第一段：自动标注 → 人工清洗检查点停
.conda\python.exe tools\workflow.py --from-stage auto_annotate

# 第二段：人工清洗完后再跑（拆分 → 转 YOLO → 训练 → 自标注 → 归档）
.conda\python.exe tools\workflow.py --from-stage select_subset
```

**跑测试：**

```bash
.conda\python.exe -m pytest tests\ -q
```

更多命令见 [AGENTS.md](AGENTS.md) 第 5 节。

---

## 自动标注：多后端 / ONNX 打码覆盖 / 多检测器组合

统一入口是 [tools/annotate/auto.py](tools/annotate/auto.py)，检测器后端与编排层完全分离（见 `tools/annotate/backends/`）：

| 模式 | 说明 |
|---|---|
| `--model-type yolo\|sam3\|detr` | supervision 数据集式导出：模型推理 → `sv.Detections` → 类别 / 置信度过滤 → 导出 YOLO / LabelMe / COCO |
| `--model-type onnx` | ONNX 一路 + 可选 SAM 文本 prompt 第二路，两路打标；按面积占比切分保留 / 删除框，小框打码后删除；`--reannotate` 覆盖指定类别并保留其它类别 |
| `--detectors-config <yaml>` | 任意 N 路检测器混搭（onnx / sam / yolo / detr，可同类型多路），统一走「合并保留大框 → 逐路小框打码（重叠保护）→ 输出 LabelMe」链路 |

- 打码统一由 `tools/annotate/ops.py` 的 `apply_blackout` 完成，重叠保护内置。
- 新增检测器后端只需在 `tools/annotate/backends/` 加一个文件并实现 `AutoLabeler` 接口。
- 配置模板见 `tools/cfg/detectors.yaml.example`。

---

## 工作流编排器

[tools/workflow.py](tools/workflow.py) 读取 YAML 配置（三层叠加：`default.yaml` + `workflow.yaml` + 项目覆盖），按顺序调用各阶段脚本，支持：

- 变量替换：`${paths.xxx}`、`${parameters.xxx}`、`${prev.xxx}`（上游 stage 输出路径）、`${latest:glob}`（匹配到的最新时间戳目录）
- `--dry-run`：只打印命令，不实际执行
- `--from-stage <name>` / `--to-stage <name>`：断点续跑 / 跑到指定阶段前停止（人工检查点）
- stage 的 `output_var`：自动捕获脚本 stdout 中的 `OUTPUT_PATH:` 行，供下游 `${prev.xxx}` 引用
- `stages_only`：任务专项精简工作流，只跑列出的 stage（参考 `tools/cfg/inherit_yolo.yaml`）

配置示例：`tools/cfg/workflow_config.yaml.example`（复制为 `src/workflow_config.yaml` 作为项目级覆盖入口）。

---

## 文档

- [workflow.md](workflow.md) — 整体链路、核心脚本表、三段式工作流快速开始、定时交付配置
- [AGENTS.md](AGENTS.md) — 目录结构、开发约定（语言规范 / 路径保护 / 官方库优先）、常用命令速查
- [tools/cfg/](tools/cfg/) — 工作流配置与任务专项模板

---

## 许可证

本项目为内部数据生产工具，暂未开放许可证。如有需要请联系仓库维护者。
