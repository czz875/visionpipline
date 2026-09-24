# AI 协作说明

本文件只保留 AI 助手在仓库中必须遵守的规则。开发流程、代码风格、验证命令和
Git 规范见 [CONTRIBUTING.md](CONTRIBUTING.md)；项目概览见 [README.md](README.md)，
数据流水线说明见 [workflow.md](workflow.md)。

## 项目约定

- 所有对话、文档、代码注释和提交说明使用简体中文；API、SDK、YOLO、LabelMe 等名称保留原文。
- Python 优先使用项目解释器 `.conda\python.exe`。依赖安装方式见 CONTRIBUTING.md。
- `main` 工作区维护通用流水线；`feat/company-encrypt` 在 `.worktrees/company-encrypt`
  worktree 中维护公司专用加密模块，不把加密实现带入 `main`。
- 开始任何改动前检查当前分支、主工作区和相关 worktree 的状态，保留用户未提交内容。

## 修改代码时

- 先检查目标模块、调用方和已有工具，再确定修改位置。按“复用现有实现、修改现有归属模块、最后新增”的顺序处理。
- 调用顺序优先使用官方库 API，其次复用 `tools.core`，最后才自行实现。
- 新脚本的默认参数集中在文件顶部的 `DEFAULT_*` 常量；`argparse` 默认值、帮助文本及回退值引用这些常量。
- `tools/` 下直接运行的脚本按 CONTRIBUTING.md 的路径引导约定处理项目导入。
- LabelMe IO、图片扫描和几何操作复用 `tools.core`，不要在调用脚本中重复实现。
- 自动标注按 `tools/annotate/backends/`（模型推理）、`ops.py`（共用几何与打码）和
  `auto.py`（流程编排）分层。新增模型后端放在 `backends/`，不要把推理逻辑放进 `ops.py`。
- 一次性工作流配置放在被忽略的 `src/`，优先组合 `tools/cfg/workflow.yaml` 已有 stage；
  不为单次操作新增专用脚本或长期配置。
- 文件或数据变更涉及批量写入、覆盖、移动或删除时，先检查目标范围，优先提供预览，
  并在操作后核对文件配对、数量或路径。
- 避免重复代码、无调用方的抽象、猜测性参数以及包住正常逻辑的多余异常处理。

## 文档与验证

- 操作说明写入 README.md 或 workflow.md；开发和代码规范写入 CONTRIBUTING.md；
  AI 专属约束才写入本文件。避免复制同一份命令或说明到多个文件。
- 改工作流前阅读 workflow.md 和相关 YAML；改共享 API 时搜索所有调用方并同步更新。
- 只报告实际执行过的检查结果。按改动范围运行 CONTRIBUTING.md 中对应的验证命令；
  不以 `--help` 或静态检查代替数据处理、GPU、设备或人工效果验证。

## Git 操作

- 提交前检查 `git status --short`，仅暂存本次相关文件；禁止 `git add .` 和 `git add -A`。
- 每次相关代码或文档改动单独提交，中文 Conventional Commit 格式及正文要求见 CONTRIBUTING.md。
- 不修改 Git 用户配置。不得提交数据集、权重、训练产物、缓存或凭据。
- 未获得用户对具体推送操作的明确授权前，不执行 `git push`、强推或远端历史改写。
