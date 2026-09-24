# Agent 协作说明

> 本文件面向在本项目里协作的 AI 编程助手（Trae / Cursor / Claude Code 等），
> 集中维护协作规则和关键约束；目录结构与操作命令见下方参考文档。

---

## 1. 项目一句话

`visionpipline` 是一个 **数据生产 + 模型训练** 流水线：把补充进来的 PNG
图像自动标注成 LabelMe JSON，做合并、清洗、拆分、YOLO 训练、自标注、归档，
最终每日交付一版可训练数据。

完整数据流见 [workflow.md](workflow.md)。

---

## 2. 关键事实速查

| 项 | 值 |
|---|---|
| 项目根目录 | `d:\PycharmProjects\visionpipline` |
| Python 解释器 | `.conda\python.exe`（项目自带的便携式 Python，不依赖系统 Python） |
| 依赖安装 | `.conda\python.exe -m pip install -r requirements.txt` |
| 工作流入口 | `.conda\python.exe tools\workflow.py --config src\workflow_config.yaml`（当前项目级覆盖在 `src/`；也可 `--config tools\cfg\workflow.yaml` 跑系统主工作流） |
| 配置示例 | `tools\cfg\workflow_config.yaml.example`（复制为 `src\workflow_config.yaml`） |
| 主要数据目录 | `datasets/`（raw / annotated / split_30 / split_70 / yolo 等） |
| 训练产物 | `runs/train/`、`archive/` |
| 公司专用加密工具 | 统一放在 `feat/company-encrypt` 分支的 `tools/encrypt/` 模块（`image_crypto.py` 底层库 / `encrypt.py` 加密 / `decrypt.py` 解密），经该分支 worktree（`.worktrees/company-encrypt`）单独管理，不进主仓库 `main` |

---

## 3. 项目结构参考

完整目录树与模块说明见 [项目目录结构](docs/project_structure.md)。

---

## 4. 开发约定（必读）

### 4.0 分支与工作区约定

本项目使用 `main` 分支承载通用流水线代码，`feat/company-encrypt` 分支通过 git worktree（`.worktrees/company-encrypt`）独立管理公司专用加密工具。

**工作区边界：**

| 工作区 | 用途 | 典型内容 |
|---|---|---|
| `main` 工作区 | 通用数据生产 + 模型训练流水线 | `tools/annotate/`、`tools/clean/`、`tools/core/`、`tools/merge/` 等 |
| `.worktrees/company-encrypt` | 公司专用图片加密/解密工具 | `tools/encrypt/`、`tools/cfg/encrypt.yaml`、`tools/cfg/decrypt.yaml` |

**开发流程：**

1. **通用改动一律先在 `main` 分支开发、提交。**
2. `feat/company-encrypt` 必须始终包含 `main` 的最新提交：开始或继续该分支的工作前，以及每次准备提交加密相关改动前，都先在 `.worktrees/company-encrypt` 确认工作区干净并执行 `git rebase main`；若 `main` 在加密分支开发期间有新提交，提交加密改动前再次同步。rebase 有冲突时先解决并验证，再继续开发或提交。
3. **不要把通用模块的改动放在 worktree 里再 cherry-pick 回 main**，这会导致两边出现内容相同但 hash 不同的重复提交，历史混乱。
4. worktree 里只保留该分支专有的内容；若发现通用文件在 worktree 里被改动，应先移回 main 提交，再同步到 worktree。

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

## 5. 参考文档

环境安装、脚本用法、工作流运行和专项命令示例统一维护在 [命令参考](docs/command_reference.md)。本文件只保留需要协作助手持续遵守的项目规则与约束。

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

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：标题和正文均用简体中文；scope 可省略，标题不超过 50 个字符，正文必须填写。合并提交统一使用 `merge:` 类型。

```text
<type>(<scope 可选>): <中文一句话标题，不超过 50 字>

<正文说明目的或原因，建议按 72 字换行>
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
| `merge` | 合并分支 |

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

### 8.3 命令示例

具体提交命令见 [命令参考](docs/command_reference.md#git-提交命令)；提交要求与禁止事项仍以本节为准。

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
