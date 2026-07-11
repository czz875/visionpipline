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
| 工作流入口 | `.conda\python.exe tools\workflow.py --config workflow_config.yaml` |
| 配置示例 | `workflow_config.yaml.example`（复制为 `workflow_config.yaml`） |
| 主要数据目录 | `datasets/`（raw / annotated / split_30 / split_70 / yolo 等） |
| 训练产物 | `runs/train/`、`archive/` |

---

## 3. 目录结构

```text
cjet-vision-pipeline/
├── .conda/                        # 项目自带的便携式 Python 3.11
├── datasets/                      # 数据集（raw_jpg / raw / yolo / autolabel ...）
├── runs/                          # yolo detect train 输出
├── archive/                       # tools/train/archive.py 生成的每日归档
├── weight/                        # 训练好的 YOLO 权重
├── tools/                         # 业务脚本（按阶段分组）
│   ├── core/                      # 公共模块
│   │   ├── constants.py           #   常量（DEFAULT_DATASET_PATH 等）
│   │   ├── images.py              #   list_images 等
│   │   ├── labelme.py             #   LabelMe JSON 扫描/读写
│   │   ├── geometry.py            #   矩形/合并/距离工具
│   │   └── __init__.py            #   统一对外导出
│   ├── annotate/                  # 标注阶段
│   │   ├── auto.py                #   自动/自标注
│   │   └── merge.py               #   框合并
│   ├── clean/                     # 清洗阶段
│   │   ├── blurry.py              #   CleanVision 模糊检测
│   │   ├── orphan_json.py         #   孤儿 JSON
│   │   └── orphan_images.py       #   缺失 JSON 的图片
│   ├── label/                     # 标签处理
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
│   └── workflow.py                # 工作流编排器入口
├── workflow.md                    # 工作流说明文档
├── workflow_config.yaml.example   # 工作流配置示例
├── requirements.txt               # 依赖清单
├── AGENTS.md                      # 本文件
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

示例见本文件 §4.2（默认参数集中到文件顶部）。

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

- 改 `tools/annotate/auto.py` 前先读它和 `tools/core/` 下的公共模块；
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

```bash
# 第一段：准备（跑到人工清洗检查点停）
.conda\python.exe tools\workflow.py --config workflow_config.yaml --from-stage auto_annotate

# 第二段：训练循环（人工清洗完后再跑）
.conda\python.exe tools\workflow.py --config workflow_config.yaml --from-stage select_subset

# 预览整条链路，不真跑
.conda\python.exe tools\workflow.py --config workflow_config.yaml --dry-run
```

### 5.4 测试

```bash
.conda\python.exe -m pytest tests\tools\ -q
```

---

## 6. 已知约束 / 坑

- **`supervision.dataset.formats.labelme` 缺失**：`tests/tools/` 里有 3 个
  LabelMe 导出相关的测试会因为当前环境装的 `supervision` 版本缺少这个模块
  而失败。这与本次项目重构无关，是 baseline 问题。

- **目录名带下划线**：`datasets/01_annotated` 这种带数字前缀的目录是为了让
  `ls` 时能按阶段顺序排列，不要随意改。

- **`--recursive` 默认行为不一致**：`tools/clean/` 下脚本默认递归，
  `tools/annotate/auto.py` 默认不递归（为了性能）。新增脚本时按场景决定
  并在 `--help` 中明确写出。

- **YOLO 输出目录结构**：`tools/annotate/auto.py --format yolo` 默认输出
  `images/` + `labels/` 分离结构；`tools/convert/yolo_to_labelme.py` 的
  `--labels` 要指向 `labels/`。

- **优先使用本地开发版 supervision**：`tools/annotate/auto.py` 顶部会
  把 `src/` 加入 `sys.path`，所以如果你在 `src/supervision/` 下改过代码，
  直接运行脚本就能生效，不需要重装。

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
   - `.conda\python.exe -m pytest tests\tools\ -q`；
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
