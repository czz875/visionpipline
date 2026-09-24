# 开发与贡献指南

本文说明如何在 Vision Pipeline 中定位代码、实现改动、验证结果并维护文档与 Git 历史。
项目功能概览见 [README.md](README.md)，流水线流程见 [workflow.md](workflow.md)，
AI 助手专属规则见 [AGENTS.md](AGENTS.md)。

文档分工参考了 [Ultralytics AGENTS.md](https://github.com/ultralytics/ultralytics/blob/main/AGENTS.md)
和 [CONTRIBUTING.md](https://github.com/ultralytics/ultralytics/blob/main/CONTRIBUTING.md)，
并按本项目的开发方式、工具链和数据处理流程调整。

## 开始开发

### 环境

在仓库根目录执行命令，优先使用项目自带的 Python 3.11：

```powershell
.conda\python.exe -m pip install -r requirements.txt
```

### 改动流程

1. 用 `git status --short` 检查工作区、分支及相关 worktree，保留已有改动。
2. 搜索现有脚本、共享工具和调用方，确认改动归属；复用现有实现，避免重复造轮子。
3. 让改动集中在一个明确的模块和目的中。优先调整现有实现，确实无法承载时再加文件、参数或抽象。
4. 按改动范围运行相关检查，并查看完整差异：`git diff --check`、`git diff`。
5. 同步更新对应的 README、工作流说明或配置示例，再按本指南提交。

### 分支与 worktree

- 通用流水线改动在 `main` 工作区维护。
- 公司专用加密实现位于 `feat/company-encrypt` 分支和 `.worktrees/company-encrypt`。
  开始该分支工作及提交前，先确认 worktree 干净并在其中执行 `git rebase main`；
  有冲突时先解决并验证。通用模块改动先在 `main` 完成，再同步到加密分支。
- 不在没有明确授权时推送。强推或远端历史改写必须有针对该操作的明确授权。

## 代码约定

### Python 脚本

- 模块、变量和函数名遵循相邻代码的命名方式；注释和说明使用简体中文。
- 新脚本在文件顶部集中定义 `DEFAULT_*` 参数，`argparse` 的默认值和帮助文本引用这些常量。
- `tools/` 下可直接运行的脚本，在导入项目模块前设置仓库根目录：

```python
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- 图片扫描、LabelMe JSON IO 和几何计算优先使用官方库或 `tools.core` 已有接口。
- docstring 面向调用者说明用途、参数、返回值和重要副作用；简单函数可写简短 docstring，
  有多个参数或返回值时分段列出，不重复翻译显而易见的代码。
- 对会修改数据的工具，提供清楚的输入、输出和写入方式；项目现有工具默认预览时，
  保持默认预览并通过 `--apply` 等显式参数执行写入。
- 不增加未被当前需求支持的兼容层、配置项和异常分支。需要抽取共用函数时，先确认已有重复和调用边界。

### 模块归属

- 共享数据集和 LabelMe 工具放在 `tools/core/`，并通过现有公共导出接口复用。
- 自动标注的模型加载与推理放在 `tools/annotate/backends/`；框几何、打码和 LabelMe 共用操作放在
  `tools/annotate/ops.py`；`tools/annotate/auto.py` 负责流程编排。
- 工作流阶段由 `tools/cfg/workflow.yaml` 管理。个人或一次性配置放在忽略的 `src/`，
  优先用 `stages_only` 组合已有阶段，并通过 `--dry-run` 检查执行命令。
- 改公共接口或工作流变量时，搜索仓库中的调用方，并同步更新受影响的配置示例和说明。

## 验证

按改动类型选择检查。下面命令使用项目解释器：

```powershell
# 检查单个 Python 文件语法
.conda\python.exe -m py_compile tools\path\to\script.py

# 确认命令行参数和帮助信息
.conda\python.exe tools\path\to\script.py --help

# 运行相关测试；完整测试集
.conda\python.exe -m pytest tests\ -q

# 预览工作流，不执行其中的数据变更
.conda\python.exe tools\workflow.py --config tools\cfg\workflow.yaml --dry-run
```

- Python 改动至少检查语法及受影响入口；公共逻辑改动运行相关测试。
- 工作流改动用 `--dry-run` 核对 stage、参数和路径。
- 文档改动检查本地链接、代码块和命令路径，并运行 `git diff --check`。
- 数据处理结果须按任务核对样本数、图片与 JSON 配对、输出路径或文件哈希。
  命令行参数检查不能证明真实数据、GPU、设备运行或人工标注质量。
- 检查结果必须说明实际执行的命令和结果；没有运行的项目明确标注未验证。

## Markdown 与文档

- README 作为项目入口，只保留用途、快速开始、核心目录概览和主要文档链接。
- `workflow.md` 维护流水线步骤、stage 行为和工作流配置说明。
- `CONTRIBUTING.md` 维护开发、代码风格、验证和提交规范；`AGENTS.md` 只维护 AI 助手执行时必须遵守的规则。
- 先更新已有文档，再考虑新增文档；同一条命令或规则只保留一个权威位置。
- 标题按层级递进，段落聚焦一个主题；命令标注适用目录和是否会写入数据，代码块注明语言。
- 仓库内链接使用相对路径，新增链接前确认目标存在；示例命令应与脚本当前 `--help` 和配置一致。
- 文档使用简体中文，中英文和数字之间留空格；保留 API、SDK、YOLO、LabelMe 等专有名称。

## Git 提交规范

每个独立代码或文档改动单独提交。提交前检查状态，只暂存本次涉及的具体文件，不使用 `git add .` 或
`git add -A`。不得修改仓库 Git 用户配置，也不得提交 `.env`、密钥、凭据、数据集、模型权重和训练产物。

标题使用 Conventional Commits：类型保留英文，scope 可选，说明使用简体中文且不超过 50 个字符；
标题后必须有中文正文，说明改动目的和影响。合并提交使用 `merge:`。

```text
<type>(<scope 可选>): <中文标题>

<改动背景、方案或影响>
- 改动点
```

允许的类型：`feat`、`fix`、`docs`、`style`、`refactor`、`perf`、`test`、`chore`、`merge`。

```powershell
git status --short
git add README.md CONTRIBUTING.md
@'
docs: 更新开发指南

集中说明项目开发、验证和文档约定。
'@ | Set-Content -LiteralPath _commit_msg.txt -Encoding utf8
git commit -F _commit_msg.txt
git log --oneline -1
```
