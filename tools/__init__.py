"""
tools
=====

CJET 数据生产 + 模型训练流水线的业务脚本集合。

本包结构仿 [ultralytics](https://github.com/ultralytics/ultralytics)：

- ``tools.cfg``     —— 工作流配置（YAML 加载 / 合并 / 变量替换）
- ``tools.engine``  —— 各 stage 子包的聚合入口
- ``tools.core``    —— 跨 stage 复用的公共模块
- ``tools.<stage>`` —— 按业务阶段拆分的子包（annotate / clean / convert / ...）
- ``tools.workflow``—— 工作流编排器入口（薄编排，配置解析走 ``tools.cfg``）

典型用法：

    from tools import cfg
    config = cfg.resolve_config()
    mapping = cfg.flatten_dict(config)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "cfg",
    "core",
    "engine",
    "workflow",
]
