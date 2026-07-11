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

Python API 顶层导出（仿 ``from ultralytics import YOLO`` 风格）::

    from tools import snapshot_sources, fix_root, inherit_dataset, \
                      rename_by_timestamp, convert_to_yolo

典型用法::

    from tools import cfg, snapshot_sources
    config = cfg.resolve_config()
    snapshot_sources([Path("datasets/autolabel")], Path("C:/backup"))
"""

from __future__ import annotations

# Python API 顶层导出（仿 ultralytics 风格；不必再写 tools.<stage>.<script>.<api>）
from tools import workflow
from tools.backup.snapshot import snapshot_sources
from tools.convert.labelme_to_yolo import convert_to_yolo
from tools.label.fix_labelme import fix_root
from tools.merge.inherit_dataset import inherit_dataset
from tools.rename.timestamp_rename import rename_by_timestamp

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "cfg",
    "core",
    "engine",
    "workflow",
    # Python API（仿 ultralytics 顶层导出）
    "snapshot_sources",
    "convert_to_yolo",
    "fix_root",
    "inherit_dataset",
    "rename_by_timestamp",
]
