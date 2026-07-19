# CJet Vision Pipeline — 项目指令

## 项目一句话

`cjet-vision-pipeline` 是一个**数据生产 + 模型训练**流水线：把补充进来的 PNG 图像自动标注成 LabelMe JSON，做合并、清洗、拆分、YOLO 训练、自标注、归档，最终每日交付一版可训练数据。

完整数据流见 [workflow.md](workflow.md)。

---

## 必读文档

- **[AGENTS.md](AGENTS.md)** — 详细的目录结构、开发约定、常用命令、已知约束。开始任何任务前先阅读此文件。

---

## 核心规范

- **语言**：对话、解释、建议、代码注释、commit message 全部使用简体中文。专有名词（API / SDK / YOLO / LabelMe 等）保留英文。
- **Python 解释器**：`.conda\python.exe`（项目自带的便携式 Python）
- **工作流入口**：`.conda\python.exe tools\workflow.py --config src\workflow_config.yaml`
- **临时工作流 cfg 放 `src/`**（被 `.gitignore` 忽略，不入 git）
- **默认参数集中到文件顶部**，每个脚本顶部有 `DEFAULT_*` 常量区
- **优先调用官方库**：supervision / cleanvision 有现成 API 的不要手写
- **通用功能先在 `main` 分支开发**，再同步到 `.worktrees/company-encrypt`；worktree 只保留公司专用加密工具相关改动

## 常用命令速查

```bash
# 安装依赖
.conda\python.exe -m pip install -r requirements.txt

# 跑工作流（预览）
.conda\python.exe tools\workflow.py --dry-run

# 跑测试
.conda\python.exe -m pytest tests\ -q
```

更多命令详见 [AGENTS.md §5](AGENTS.md)。
