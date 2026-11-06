# 数据生产工作流

本文档描述如何把「补充数据 → 自动标注 → 框合并 → 自动清洗 → 人工清洗 → 随机
筛选 30% → 转 YOLO → 训练 → 自标注 → 回 LabelMe → 再清洗 → 归档」整条链路用
脚本串起来，实现每日定时交付。

> 本项目位于 `d:\PycharmProjects\visionpipline`，
> Python 环境使用项目自带的便携式解释器 `.conda\python.exe`，
> 依赖清单见 [requirements.txt](requirements.txt)。
> 项目结构仿 [ultralytics](https://github.com/ultralytics/ultralytics)：
> 业务脚本分到 `tools/<stage>/`，所有 yaml 配置集中在 `tools/cfg/`，工作流编排器在 `tools/workflow.py`。

---

## 整体链路

```text
autolabel 平铺数据（可选）
    │
    ├──► tools/backup/snapshot.py       备份 autolabel + behavior 到 archive/backups
    │
    ▼
tools/merge/inherit_dataset.py        按 1000 张/批接续到 behavior/0023/...
    │                                   并按 JSON label 归类到 6 个分类子目录
    ▼
tools/label/fix_labelme.py            修复 LabelMe JSON 常见损坏（imagePath / 矩形 / 尺寸）
    │
    ▼
tools/rename/timestamp_rename.py      按时间戳改名 + 同步 JSON 的 imagePath
    │
    ▼
JPG 原始数据（可选）──► tools/convert/jpg_to_png.py ──► PNG
                                                        │
                                                        ▼
补充 PNG 数据（固定尺寸；脚本不据此缩放，实际尺寸由采集端保证）
    │
    ▼
多后端自动标注（YOLO / SAM3 / DETR / ONNX）──► LabelMe JSON
    │
    ▼
标注框合并（tools/annotate/merge.py）
    │
    ▼
自动丢弃不合格数据
    ├── tools/clean/blurry.py           模糊检测并移出
    ├── tools/clean/orphan_json.py      孤儿 JSON
    └── tools/clean/orphan_images.py    缺失 JSON 的图片
    │
    ▼
【人工清洗检查点】在 LabelMe 中检查并修正自动阶段结果
    │
    ▼
随机拆分为：30% 高质量  +  70% 剩余数据
    │                              │
    ├──► tools/convert/labelme_to_yolo.py  (30% 转 YOLO)
    │       │
    │       ▼
    │   yolo detect train  (训练 SOTA 模型)
    │       │
    │       ▼
    │   tools/annotate/auto.py  (自标注剩余 70%)
    │       │
    │       ▼
    │   tools/convert/yolo_to_labelme.py  (回 LabelMe)
    │       │
    └───────┘
            ▼
    tools/clean/orphan_json.py  (最终清洗)
            ▼
    tools/train/archive.py  (每日归档)
```

> **为什么分三段？**
> - **第零段（接续）**：日常补充数据时跑——备份 autolabel+behavior，把 autolabel 按 1000/批接续到 behavior 的 0023/0024/...，修复 JSON、按时间戳改名。
> - **第一段（准备）**：从「补充 PNG」跑到人工清洗检查点。
> - **第二段（训练）**：人工清洗完成后，从数据集拆分开始继续跑。
>
> 工作流配置也按这三段切分，每段在对应检查点（`manual_review`、stage 终点）停下。

---

## 核心脚本

| 阶段 | 脚本 | 说明 |
|---|---|---|
| 数据备份 | [tools/backup/snapshot.py](tools/backup/snapshot.py) | 把指定目录打成带时间戳的 .tar.gz 备份 |
| 数据接续 | [tools/merge/inherit_dataset.py](tools/merge/inherit_dataset.py) | 把 autolabel 平铺数据按 1000/批接续到 behavior 的新 batch，按 JSON label 归类到 6 个分类子目录 |
| LabelMe 修复 | [tools/label/fix_labelme.py](tools/label/fix_labelme.py) | 修复 LabelMe JSON 常见损坏（imagePath / 矩形 / 尺寸） |
| 标签对齐 | [tools/label/align_labelme.py](tools/label/align_labelme.py) | 把 PNG / LabelMe JSON 按 JSON 内 imagePath 字段对齐（basename 一致）+ 同步 imagePath；补救改名 / 接续后的 PNG-JSON 错位 |
| 批量改名 | [tools/rename/timestamp_rename.py](tools/rename/timestamp_rename.py) | 按时间戳改名为 YYYYMMDD_HHMMSS_NNNNNN，可选同步 LabelMe JSON 的 imagePath |
| JPG → PNG | [tools/convert/jpg_to_png.py](tools/convert/jpg_to_png.py) | 把原始 JPG 批量转 PNG（可选），同时修复同名 LabelMe JSON |
| IR 增强 | [tools/augment/ir_enhance.py](tools/augment/ir_enhance.py) | 对 IR 数据做全局 + 局部光照增强（可选，按类别过滤） |
| 自动标注 | [tools/annotate/auto.py](tools/annotate/auto.py) | 多后端标注：YOLO / SAM3 / DETR 数据集式导出，或 ONNX 一路+SAM 两路打标 / 覆盖，或多检测器组合（`--detectors-config`），输出 YOLO/LabelMe/COCO |
| 框合并 | [tools/annotate/merge.py](tools/annotate/merge.py) | 合并邻近同标签矩形框 |
| 模糊检测 | [tools/clean/blurry.py](tools/clean/blurry.py) | CleanVision 检测模糊目标区域并移出原图 |
| 丢弃不合格 | [tools/clean/orphan_json.py](tools/clean/orphan_json.py)<br>[tools/clean/orphan_images.py](tools/clean/orphan_images.py) | 清理孤儿 JSON / 缺失 JSON 的图片 |
| 标签替换 | [tools/label/replace.py](tools/label/replace.py) | 批量替换标签名（人脸 → face 等） |
| 数据集拆分 | [tools/split/dataset.py](tools/split/dataset.py) | 随机按比例拆分数据集（默认30/70） |
| LabelMe → YOLO | [tools/convert/labelme_to_yolo.py](tools/convert/labelme_to_yolo.py) | 批量转 YOLO 格式（默认 9:1 划分 train/val，每个 batch 单独输出） |
| YOLO 训练 | `yolo detect train ...`（Ultralytics CLI） | 训练 SOTA 模型 |
| 自标注 | [tools/annotate/auto.py](tools/annotate/auto.py)（使用训练好的权重） | 对剩余 70% 推理 |
| YOLO → LabelMe | [tools/convert/yolo_to_labelme.py](tools/convert/yolo_to_labelme.py) | 把自标注结果转回 LabelMe |
| 归档 | [tools/train/archive.py](tools/train/archive.py) | 按时间戳归档训练产物 |
| 打包 | [tools/train/create_tar_gz.py](tools/train/create_tar_gz.py) | 把归档目录压缩为 .tar.gz，方便交付 |
| 公共模块 | [tools/core/](tools/core/) | 常量、LabelMe 读写、几何工具等 |

数据处理的写入约定：

- 使用共享 `save_labelme` / `write_image` 的步骤采用原子替换：单文件写盘失败保留旧文件，
  不会连带修改其他硬链接。此约定针对单文件，图片与 JSON 成对发布仍由具体步骤负责。
- `convert/jpg_to_png.py` 在写盘前检查全部 PNG/JSON 目标，拒绝同 stem 的 JPG/JPEG 冲突、
  已有目标、重叠输入输出和目录链接。坏图片或同名坏 JSON 会使该样本失败，不留下半成品；
  完整成功的样本保留，有任何失败则整批返回非零；回滚前复核本次发布的文件身份，保留已识别的替换文件。
  转换期间应避免其他程序覆盖同一输出目录中的文件。
  输出标注的尺寸取实际图片尺寸。
  未指定输出目录时，每轮独占创建时间戳目录，同秒并发运行也不共用目录。
- `yolo_to_labelme.py` 会将图片复制到输出目录，与生成的同名 JSON 配对；
  JSON 的 `imagePath` 指向输出图片，后续孤儿 JSON 清洗不会误删有效标注。
  输出目录必须为空，且不能与图片输入目录重叠，避免平铺复制覆盖已有图片或硬链接内容。
- `split/dataset.py` 只向空目录写入；输入与 A/B 输出目录必须互不重叠。
  重跑时指定新的输出目录，或省略 `--output-a` / `--output-b` 自动创建时间戳目录。
  输入缺失、没有图片或输出非空时返回失败，保留已有数据。
- `backup/snapshot.py` 在写入前校验全部源目录；源缺失、源目录同名、输出位于源内，
  或目标备份文件已存在时返回失败。备份文件不会覆盖旧文件。
- `clean/blurry.py` 要求输入、裁剪缓存根目录、移出目录互不重叠。每轮在缓存根目录
  新建独立子目录，`--cleanup-crop` 只删除本轮子目录，保留已有文件；旧裁剪不会参与新一轮判定。
  递归输入按 JSON 所在目录解析 `imagePath`，检测结果通过完整图片路径定位，避免同名图被连带移走。
  原图与所有来源 JSON 分别按数据集相对路径移出，保留跨目录引用；输入图片经过符号链接时返回失败，
  应先将标注对齐到实际图片路径。待移出标注的 `imagePath` 必须相对于 JSON 所在目录，带根或盘符的引用会在
  移动前报错，避免移出后失效。移出前校验全部目标及其父路径，拒绝已有目标和目录链接；每个目标
  采用排他创建，跨卷复制失败时清理本次残件，移动失败时尝试回滚已完成的移动。
  `--workers` 控制裁剪线程，CleanVision 评分
  使用单进程，避免 Windows 子进程重复加载 SciPy 导致内存耗尽。
- `rename/timestamp_rename.py` 在规划时预留本轮目标名；开启 `--labelme-sync` 时同时检查并预留
  同名 JSON，避免已有文件与本轮任务之间冲突。
- YOLO 导出在写盘前检查标签文件名冲突：同 stem 的不同扩展名、仅大小写不同的图片都会报错，
  应先重命名或改用 LabelMe 导出。LabelMe 导出会为大小写冲突生成不同名称，保留图片与 JSON 配对。
- `label/fix_labelme.py` 在预览或执行中遇到无法修复的 JSON 时返回非零，防止工作流把失败当成成功。

---

## 自动标注：多后端 / ONNX 打码覆盖 / 多检测器组合

统一入口是 [tools/annotate/auto.py](tools/annotate/auto.py)，按 `--model-type` /
`--detectors-config` 走不同链路，三类能力并存：

### 1. supervision 数据集式（yolo / sam3 / detr）

- `--model-type yolo|sam3|detr`：模型推理 → `sv.Detections` → 类别 / 置信度过滤 →
  累积为 `sv.DetectionDataset` → 导出 YOLO / LabelMe / COCO（`--format`）。
- DETR 走 ultralytics RT-DETR（`DETRLabeler`），接口与 YOLO 一致。

### 2. ONNX 打标 / 覆盖（onnx）

- 一路 ONNX（可选 SAM 文本 prompt 第二路）标新框 → 按面积占比切分保留 / 删除框 →
  小框打码（默认马赛克，重叠保护）后删除 → 输出 LabelMe。**默认仅统计预览**，
  加 `--apply` 才写盘。
- 加 `--reannotate`：覆盖指定类别并保留其它现有类别，原地覆盖图片与 JSON
  （覆盖模式默认纯黑打码）。

### 3. 多检测器组合（--detectors-config）

- `--detectors-config src/detectors.yaml`：任意 N 路混搭（onnx / sam / yolo / detr，
  可同类型多路），每路各自推理出「框 + 逐框标签」，统一走「合并保留大框 → 逐路
  小框打码（重叠保护）→ 输出 LabelMe」。默认预览，加 `--apply` 写盘。
- 配置模板见 [tools/cfg/detectors.yaml.example](tools/cfg/detectors.yaml.example)；
  任务专项配置放 `src/`（不入 git，见下文）。

常用命令（更多见 [AGENTS.md](AGENTS.md) §5.2）：

```bash
# YOLO 标注 → LabelMe（--apply 才写盘，去掉即预览）
.conda\python.exe tools\annotate\auto.py --model-type yolo ^
    --source datasets\raw --output datasets\01_annotated --format labelme --apply

# ONNX 一路 + SAM 第二路，默认预览（加 --apply 才写盘）
.conda\python.exe tools\annotate\auto.py --model-type onnx ^
    --source datasets\raw --output datasets\01_annotated_onnx_sam ^
    --onnx-model weight\yolov5s-lmk.onnx --onnx-label face ^
    --sam-model weight\sam3.1_multiplex.pt --sam-prompt hand --sam-label hand ^
    --onnx-min-ratio 0.01 --sam-min-ratio 0.01 --mosaic-block 16 --apply

# 多检测器组合（cfg 放 src/，默认预览，--apply 写盘）
.conda\python.exe tools\annotate\auto.py --detectors-config src\detectors.yaml --apply
```

---

## 工作流编排器

[tools/workflow.py](tools/workflow.py) 读取 YAML 配置文件，按顺序调用上述脚本，支持：

- 变量替换：`${paths.raw_data}`、`${parameters.yolo_epochs}` 等
- `--dry-run`：只打印命令，不实际执行
- `--from-stage <name>`：从指定阶段恢复执行
- `--to-stage <name>`：跑到指定阶段之前停止（用于人工检查点）
- 自动记录日志到 `workflow.log`

### 快速开始

1. 复制示例配置（首次使用）：

   ```bash
   copy tools\cfg\workflow_config.yaml.example src\workflow_config.yaml
   ```

   一次性 / 临时覆盖配置放 `src/`（不入 git）。`tools/workflow.py` 的默认入口按顺序回退：
   `tools/cfg/workflow_config.yaml` → `src/workflow_config.yaml` → `tools/cfg/workflow.yaml`。

2. 修改配置（按实际路径/参数；重点是 `paths` 与 `project`）：
   - `tools/cfg/default.yaml`：系统默认
   - `tools/cfg/workflow.yaml`：完整工作流 stage 定义
   - `src/workflow_config.yaml`：项目级覆盖（默认入口）

3. **第零段（数据接续）**——补数据时跑：

   ```bash
   # 1) 备份 autolabel + behavior 到 archive/backups
   .conda\python.exe tools\backup\snapshot.py ^
       --sources datasets\autolabel,datasets\behavior ^
       --output-dir archive\backups ^
       --apply

   # 2) autolabel 按 1000/批接续到 behavior/0023/
   .conda\python.exe tools\merge\inherit_dataset.py ^
       --source datasets\autolabel ^
       --target datasets\behavior ^
       --batch-size 1000 ^
       --apply

   # 3) 修复 LabelMe JSON 常见损坏（默认 dry-run，看清楚再加 --apply）
   .conda\python.exe tools\label\fix_labelme.py ^
       --root datasets\behavior ^
       --recursive

   # 4) 按时间戳改名 + 同步 JSON 的 imagePath
   .conda\python.exe tools\rename\timestamp_rename.py ^
       --source-dir datasets\behavior ^
       --labelme-sync ^
       --apply
   ```

4. **第一段（准备）**——跑到人工清洗检查点停：

   ```bash
   .conda\python.exe tools\workflow.py --from-stage auto_annotate --to-stage manual_review
   ```

   `manual_review` 只是一个提示阶段（`echo`），不会自己停；必须用 `--to-stage manual_review`
   才会在它之前返回。

5. 人工在 LabelMe 中检查、修正 `${paths.annotated}` 目录的数据。

6. **第二段（训练）**——人工清洗完后再跑：

   ```bash
   .conda\python.exe tools\workflow.py --from-stage select_subset
   ```

   或者一次性预览整条链路：

   ```bash
   .conda\python.exe tools\workflow.py --dry-run
   ```

---

## 工作流变量与脚本输出约定

### 新增变量

除已有的 `${paths.xxx}`、`${parameters.xxx}`、`${project.xxx}` 外，工作流还支持：

- `${prev.<var>}`：上游 stage 声明的 `output_var` 所代表的实际输出路径。
- `${latest:<glob>}`：匹配 glob 的目录中，按目录名时间戳最新的一个。

### stage 输出路径传递

stage 可声明 `output_var`，工作流会自动捕获其 stdout 中的 `OUTPUT_PATH:` 行。
普通变量名写入 `${prev.<变量名>}`；以 `paths.` 开头则直接更新后续阶段使用的路径。
主工作流的自动标注采用后者，确保合并与清洗读取本轮实际批次：

```yaml
stages:
  - name: auto_annotate
    output_var: paths.annotated
    command: >
      python tools/annotate/auto.py
      --source "${paths.raw_data}"
      --output "${paths.annotated}"
      --format labelme
      --apply

  - name: merge_boxes
    command: >
      python tools/annotate/merge.py
      --json-dir "${paths.annotated}"
```

使用 `--from-stage` 跳过自动标注、从后续阶段恢复时，工作流会将被跳过阶段的
`paths.*` 输出绑定恢复到该阶段最近一个已有批次。需要处理指定批次时，将对应
路径明确设为 `datasets/batch_时间戳/阶段目录`；这类显式批次路径不会改选其它批次。
仅运行单模块配置时，应在配置中提供该模块的实际输入目录。

### 脚本输出约定

会写盘的脚本未指定 `--output` 时，会自动生成 `[function_name]_[YYYYMMDD_HHMMSS]` 时间戳子目录，并在 stdout 输出：

```text
OUTPUT_PATH:datasets/01_annotated/auto_annotate_20260721_153022
```

### 使用 ${latest:...} 查找最新目录

临时 cfg 或不方便声明 `output_var` 时，可用 `${latest:glob}`：

```yaml
- name: merge_boxes
  command: python tools/annotate/merge.py --json-dir ${latest:datasets/01_annotated/*}
```

---

## 每日定时交付

用 Windows 任务计划程序 / Linux cron 在每天 17:00 前触发即可。

### Windows 任务计划程序示例

创建 `daily_workflow_prep.bat`：

```batch
@echo off
cd /d D:\PycharmProjects\visionpipline
D:\PycharmProjects\visionpipline\.conda\python.exe tools\workflow.py --from-stage auto_annotate
```

以及 `daily_workflow_train.bat`：

```batch
@echo off
cd /d D:\PycharmProjects\visionpipline
D:\PycharmProjects\visionpipline\.conda\python.exe tools\workflow.py --from-stage select_subset
```

然后创建任务：

- **触发器**：数据准备每天 08:00，训练循环每天 16:30
- **操作**：启动对应 `.bat`

### Linux cron 示例

```cron
# 上午跑准备阶段
0 8 * * * cd /path/to/visionpipline && .conda/python tools/workflow.py --from-stage auto_annotate
# 下午人工清洗后跑训练阶段
30 16 * * * cd /path/to/visionpipline && .conda/python tools/workflow.py --from-stage select_subset
```

---

## 常见问题

### 1. 训练阶段报错找不到 `yolo` 命令

确保已安装 Ultralytics：

```bash
.conda\python.exe -m pip install ultralytics
```

### 2. 自标注后的 YOLO labels 目录结构不对

`tools/annotate/auto.py` 的 YOLO 输出会把 `labels/` 下的 `.txt` 平铺到输出目录
顶层（`runners/supervision.py` 的 `_flatten_yolo_labels`），因此
`tools/convert/yolo_to_labelme.py` 的 `--labels` 应指向**输出目录本身**，而不是
`labels/` 子目录。

### 3. 30% 高质量数据如何定义「高质量」

当前示例在人工清洗后使用随机采样。如需按 CleanVision 模糊分、标注框大小等
业务指标筛选，可把 `tools/split/dataset.py` 替换为自定义脚本，或在配置中调整
`select_subset` 阶段命令。

### 4. 如何迭代训练 1-2 次

在配置中把 `train_yolo` → `self_annotate` → `convert_back_labelme` 复制一份，
形成第二阶段：用第一次训练后的模型重新自标注，再训第二次。

### 5. clean/blurry 移走了清晰图怎么办？

`tools/clean/blurry.py` 只把被判定为模糊的原图移到 `--output-path`；清晰图仍保留
在原目录。移出条件同时要求 CleanVision 标记为模糊，且分数不大于 `--blurry-threshold`。
误判过多时可调低该阈值，先用独立样本复核效果。

### 6. 缺依赖怎么装

在项目根目录下执行：

```bash
.conda\python.exe -m pip install -r requirements.txt
```

如果只想装核心（`supervision` + `cleanvision`），可以临时注释掉
[requirements.txt](requirements.txt) 下半部分，或者手动：

```bash
.conda\python.exe -m pip install supervision cleanvision
```

---

## 扩展

新增业务阶段时，只需：

1. 编写单功能脚本（按阶段放在 `tools/` 对应子目录下）；
2. 在 `tools/cfg/workflow.yaml` 的 `stages` 列表中追加一段 `name` + `command`；
3. 用 `--dry-run` 验证命令是否正确。

如果新增的脚本里有可复用的常量 / 函数（图片扫描、LabelMe 读写、几何工具等），
优先放到 `tools/core/` 下，再由 `tools/core/__init__.py` 统一导出，保持「即插即用」。

---

## 任务专项精简工作流

如果只想跑主工作流中的某几个 stage（比如补数据时只跑「备份 + 接续 + 重命名 + 转 YOLO」，
跳过中间的「自动标注 / 合并 / 清洗 / 人工清洗 / 训练 / 自标注 / 归档」），

在 project 顶层加 `stages_only` 字段，列出要跑的 stage name 即可：

```yaml
# tools/cfg/inherit_yolo.yaml
stages_only:
  - backup_snapshot
  - inherit_dataset
  - fix_labelme
  - rename_with_labelme_sync
  - convert_to_yolo
```

`resolve_config()` 看到 `stages_only` 字段后，会按列出的 name 顺序从
`workflow.yaml` 拿对应 stage 的完整定义（带 command），其它 stage 全部跳过。

跑法：

```bash
.conda\python.exe tools\workflow.py --config tools\cfg\inherit_yolo.yaml
```

如果需要覆盖 paths（备份目录 / autolabel / behavior），在同 yaml 加 `paths:` 段即可
（不写就回退到 `tools/cfg/default.yaml` 里的默认值）。

> **任务专项 / 临时工作流约定（关键）**：一次性 / 临时 cfg **不要** 写进
> `tools/cfg/`，而是放到 `src/`（`src/` 被 `.gitignore` 整体忽略，不入 git）。
> 优先用 `stages_only` 引用 `workflow.yaml` 已有 stage（参考 `inherit_yolo.yaml`
> 风格）；只有现有 stage 拼不出来时，才在 `src/` 的 cfg 里直接定义 stage 命令。
>
> 两个参考模板：
> - [tools/cfg/all_modules.yaml.example](tools/cfg/all_modules.yaml.example)：
>   全部模块「用法总表」，每个 `tools/` 脚本对应一个 stage，带 `enabled` / `order`，
>   可作为拼装参考。
> - [tools/cfg/detectors.yaml.example](tools/cfg/detectors.yaml.example)：
>   多检测器组合标注的 YAML 模板（`--detectors-config` 引用，见上）。
