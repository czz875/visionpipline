# 命令参考

> 本文集中记录项目环境、单脚本和工作流的常用命令。项目约束与开发规则见 [AGENTS.md](../AGENTS.md)。

## 常用命令速查

> 所有命令都在项目根目录 `d:\PycharmProjects\visionpipline` 下执行。

### 1. 安装 / 检查依赖

```bash
# 装全部依赖
.conda\python.exe -m pip install -r requirements.txt

# 只装核心（supervision / cleanvision）
.conda\python.exe -m pip install supervision cleanvision

# 看当前版本
.conda\python.exe -c "import supervision, cleanvision; print(supervision.__version__, cleanvision.__version__)"
```

### 2. 单脚本跑

```bash
# JPG 批量转 PNG（自动修复同名 LabelMe JSON）
.conda\python.exe tools\convert\jpg_to_png.py ^
    --input datasets\raw_jpg ^
    --output datasets\raw ^
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
    --source-dir datasets\behavior

.conda\python.exe tools\rename\timestamp_rename.py ^
    --source-dir datasets\behavior ^
    --apply

# 按时间戳改名 + 同步同名 LabelMe JSON 的 imagePath（LabelMe 数据集推荐）
.conda\python.exe tools\rename\timestamp_rename.py ^
    --source-dir datasets\behavior ^
    --labelme-sync ^
    --apply

# 数据备份：把 autolabel + behavior 打成 .tar.gz 到 archive/backups（默认 dry-run）
.conda\python.exe tools\backup\snapshot.py ^
    --sources datasets\autolabel,datasets\behavior ^
    --output-dir archive\backups

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
    --classes phone,cigarette,face,hand

# 输出图片使用硬链接以减少重复存储；源目录和输出目录必须位于同一卷
.conda\python.exe tools\convert\labelme_to_yolo.py ^
    --src datasets\behavior ^
    --out datasets\yolo ^
    --classes phone,cigarette,face,hand ^
    --hardlink-images
```

### 3. 跑整个工作流

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

### 4. 专项工作流：备份 datasets\1|2|3 + 脸(ONNX)/手·手机·香烟(SAM) 标注打码

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
    --output-dir archive\backups

# 真打包（三个目录各生成一个带时间戳的 .tar.gz）
.conda\python.exe tools\backup\snapshot.py ^
    --sources datasets\1,datasets\2,datasets\3 ^
    --output-dir archive\backups ^
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

### 5. 测试

> 测试统一收在 `tests/` 下：`tests/` 根目录放独立测试（如 `test_torch_cuda.py` 校验
> torch/CUDA、`test_onnx_execution_provider.py` 校验 onnxruntime 能否在 CPU/GPU 上
> 真实推理），`tests/tools/<stage>/` 放按阶段归类的测试。pytest 递归收集，
> 直接跑 `tests/` 即可覆盖全部。

```bash
.conda\python.exe -m pytest tests\ -q
```

---


## Git 提交命令

### 提交命令示例

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
