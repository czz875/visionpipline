# 数据生产工作流

本文档描述如何把「补充数据 → 自动标注 → 框合并 → 自动清洗 → 人工清洗 → 随机
筛选 30% → 转 YOLO → 训练 → 自标注 → 回 LabelMe → 再清洗 → 归档」整条链路用
脚本串起来，实现每日定时交付。

> 本项目位于 `d:\PycharmProjects\cjet-vision-pipeline`，
> Python 环境使用项目自带的便携式解释器 `.conda\python.exe`，
> 依赖清单见 [requirements.txt](requirements.txt)。
> 项目结构仿 [ultralytics](https://github.com/ultralytics/ultralytics)：
> 业务脚本分到 `tools/<stage>/`，所有 yaml 配置集中在 `tools/cfg/`，工作流编排器在 `tools/workflow.py`。

---

## 整体链路

```text
autolabel 平铺数据（可选）
    │
    ├──► tools/backup/snapshot.py       备份 autolabel + behavior 到 C:\Users\EDY\Pictures
    │
    ▼
tools/merge/inherit_dataset.py        按 1000 张/批接续到 behavior/0023/...
    │                                   并按 JSON label 归类到 8 个分类子目录
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
补充 PNG 数据（固定尺寸，由 project.image_width/height 控制）
    │
    ▼
SAM / YOLO 自动标注 ──► LabelMe JSON
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
| 数据接续 | [tools/merge/inherit_dataset.py](tools/merge/inherit_dataset.py) | 把 autolabel 平铺数据按 1000/批接续到 behavior 的新 batch，按 JSON label 归类到 8 个分类子目录 |
| LabelMe 修复 | [tools/label/fix_labelme.py](tools/label/fix_labelme.py) | 修复 LabelMe JSON 常见损坏（imagePath / 矩形 / 尺寸） |
| 批量改名 | [tools/rename/timestamp_rename.py](tools/rename/timestamp_rename.py) | 按时间戳改名为 YYYYMMDD_HHMMSS_NNNNNN，可选同步 LabelMe JSON 的 imagePath |
| JPG → PNG | [tools/convert/jpg_to_png.py](tools/convert/jpg_to_png.py) | 把原始 JPG 批量转 PNG（可选），同时修复同名 LabelMe JSON |
| IR 增强 | [tools/augment/ir_enhance.py](tools/augment/ir_enhance.py) | 对 IR 数据做全局 + 局部光照增强（可选，按类别过滤） |
| 自动标注 | [tools/annotate/auto.py](tools/annotate/auto.py) | YOLO / SAM3 生成标注 |
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
   copy tools\cfg\workflow.example.yaml tools\cfg\workflow.yaml
   ```

2. 修改配置（按实际路径/参数；重点是 `paths` 与 `project`）：
   - `tools/cfg/default.yaml`：系统默认
   - `tools/cfg/workflow.yaml`：完整工作流 stage 定义
   - 项目根 `workflow_config.yaml`：项目级覆盖（可选）

3. **第零段（数据接续）**——补数据时跑：

   ```bash
   # 1) 备份 autolabel + behavior 到 C:\Users\EDY\Pictures
   .conda\python.exe tools\backup\snapshot.py ^
       --sources datasets\autolabel,datasets\behavior ^
       --output-dir C:\Users\EDY\Pictures ^
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
   .conda\python.exe tools\workflow.py --config workflow_config.yaml --from-stage auto_annotate
   ```

   到达 `manual_review` 之前会自动停止。

5. 人工在 LabelMe 中检查、修正 `${paths.annotated}` 目录的数据。

6. **第二段（训练）**——人工清洗完后再跑：

   ```bash
   .conda\python.exe tools\workflow.py --config workflow_config.yaml --from-stage select_subset
   ```

   或者一次性预览整条链路：

   ```bash
   .conda\python.exe tools\workflow.py --config workflow_config.yaml --dry-run
   ```

---

## 每日定时交付

用 Windows 任务计划程序 / Linux cron 在每天 17:00 前触发即可。

### Windows 任务计划程序示例

创建 `daily_workflow_prep.bat`：

```batch
@echo off
cd /d D:\PycharmProjects\cjet-vision-pipeline
D:\PycharmProjects\cjet-vision-pipeline\.conda\python.exe tools\workflow.py --config workflow_config.yaml --from-stage auto_annotate
```

以及 `daily_workflow_train.bat`：

```batch
@echo off
cd /d D:\PycharmProjects\cjet-vision-pipeline
D:\PycharmProjects\cjet-vision-pipeline\.conda\python.exe tools\workflow.py --config workflow_config.yaml --from-stage select_subset
```

然后创建任务：

- **触发器**：数据准备每天 08:00，训练循环每天 16:30
- **操作**：启动对应 `.bat`

### Linux cron 示例

```cron
# 上午跑准备阶段
0 8 * * * cd /path/to/cjet-vision-pipeline && .conda/python tools/workflow.py --config workflow_config.yaml --from-stage auto_annotate
# 下午人工清洗后跑训练阶段
30 16 * * * cd /path/to/cjet-vision-pipeline && .conda/python tools/workflow.py --config workflow_config.yaml --from-stage select_subset
```

---

## 常见问题

### 1. 训练阶段报错找不到 `yolo` 命令

确保已安装 Ultralytics：

```bash
.conda\python.exe -m pip install ultralytics
```

### 2. 自标注后的 YOLO labels 目录结构不对

`tools/annotate/auto.py` 的 YOLO 输出默认是 `images/` + `labels/` 分开的结构，
`tools/convert/yolo_to_labelme.py` 需要 `--labels` 指向 `labels/` 目录。

### 3. 30% 高质量数据如何定义「高质量」

当前示例在人工清洗后使用随机采样。如需按 CleanVision 模糊分、标注框大小等
业务指标筛选，可把 `tools/split/dataset.py` 替换为自定义脚本，或在配置中调整
`select_subset` 阶段命令。

### 4. 如何迭代训练 1-2 次

在配置中把 `train_yolo` → `self_annotate` → `convert_back_labelme` 复制一份，
形成第二阶段：用第一次训练后的模型重新自标注，再训第二次。

### 5. clean/blurry 移走了清晰图怎么办？

`tools/clean/blurry.py` 只把被判定为模糊的原图移到 `--output-path`；清晰图仍保留
在原目录。如果阈值太严格，可调高 `--blurry-threshold`（如 0.25）。

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
2. 在 `workflow_config.yaml` 的 `stages` 列表中追加一段 `name` + `command`；
3. 用 `--dry-run` 验证命令是否正确。

如果新增的脚本里有可复用的常量 / 函数（图片扫描、LabelMe 读写、几何工具等），
优先放到 `tools/core/` 下，再由 `tools/core/__init__.py` 统一导出，保持「即插即用」。
