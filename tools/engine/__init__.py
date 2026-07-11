"""
tools.engine
============

各 stage 子包的聚合入口（仿 [ultralytics](https://github.com/ultralytics/ultralytics)
``ultralytics.engine`` 风格）。

``tools.<stage>/`` 目录保持扁平按业务阶段组织；这里只把它们重新挂到
``tools.engine`` 命名空间下，方便用户一处拿到所有可调用模块。

注意：这里 **不** 包含 ``cfg`` / ``core`` / ``workflow``——它们仍走
``tools.cfg``、``tools.core``、``tools.workflow`` 直接 import。
"""

from __future__ import annotations

from tools import (
    annotate,
    augment,
    backup,
    clean,
    convert,
    label,
    merge,
    rename,
    split,
    train,
)

__all__ = [
    "annotate",
    "augment",
    "backup",
    "clean",
    "convert",
    "label",
    "merge",
    "rename",
    "split",
    "train",
]
