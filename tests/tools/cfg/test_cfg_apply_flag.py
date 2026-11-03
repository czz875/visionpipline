"""配置里「默认预览、`--apply` 才写盘」的契约守护测试。

背景：``tools/annotate/auto.py`` 与 ``tools/clean/orphan_*.py`` 都默认 dry-run，
只有显式传 ``--apply`` 才真正写盘。配置里漏掉 ``--apply`` 时，stage 会「成功」
返回 0 但什么都不产出，定时任务表现为「跑完了结果不对」——这类缺陷不会报错，
只能靠契约测试兜住。

本文件同时守两件事：
1. 凡引用上述脚本的 stage，必须带上 ``--apply``（或用 ``parameters.*_flag`` 显式开关）；
2. 主工作流不得引用磁盘上不存在的 YOLO 批次目录（``datasets/yolo/all`` 从未存在）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

CFG_DIR = Path(__file__).resolve().parents[3] / "tools" / "cfg"

# 默认 dry-run、必须显式 --apply 才写盘的脚本
APPLY_REQUIRED_SCRIPTS = (
    "tools/annotate/auto.py",
    "tools/clean/orphan_json.py",
    "tools/clean/orphan_images.py",
)

# 允许用变量占位代替字面量 --apply 的写法（专项 cfg 用它做显式开关）
APPLY_FLAG_PATTERN = re.compile(r"\$\{parameters\.[a-z_]*apply[a-z_]*\}")


def _iter_stage_commands() -> list[tuple[str, str, str]]:
    """遍历 tools/cfg 下所有 YAML，产出 (文件名, stage 名, 命令) 三元组。"""
    rows: list[tuple[str, str, str]] = []
    for path in sorted(CFG_DIR.glob("*.yaml")) + sorted(CFG_DIR.glob("*.yaml.example")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for stage in data.get("stages") or []:
            command = stage.get("command") or ""
            if command:
                rows.append((path.name, stage.get("name", "unnamed"), command))
    return rows


@pytest.mark.parametrize(
    "cfg_name,stage_name,command",
    _iter_stage_commands(),
    ids=lambda v: str(v),
)
def test_apply_required_scripts_pass_apply_flag(
    cfg_name: str, stage_name: str, command: str
) -> None:
    """引用默认 dry-run 脚本的 stage 必须显式要求写盘。"""
    referenced = [s for s in APPLY_REQUIRED_SCRIPTS if s in command]
    if not referenced:
        return
    has_literal = "--apply" in command
    has_variable = bool(APPLY_FLAG_PATTERN.search(command))
    assert has_literal or has_variable, (
        f"{cfg_name} 的 stage '{stage_name}' 调用了 {referenced}，"
        f"但既没有 --apply 也没有 parameters.*apply*_flag 开关；"
        f"该阶段会静默只预览、不产出任何文件。"
    )


def test_workflow_yaml_does_not_reference_missing_yolo_batch() -> None:
    """主工作流引用的 YOLO data.yaml 必须指向真实存在的批次目录。

    历史上这里写成 ``datasets/yolo/all/data.yaml``，而磁盘上只有 ``0001``~``0156``
    每个批次一份，导致训练阶段必然失败。
    """
    data = yaml.safe_load((CFG_DIR / "workflow.yaml").read_text(encoding="utf-8")) or {}
    commands = "\n".join(
        (s.get("command") or "") for s in data.get("stages") or []
    )
    assert "datasets/yolo/all/" not in commands, (
        "workflow.yaml 引用了不存在的 datasets/yolo/all/ 目录；"
        "实际批次目录是 0001~0156，应指向具体批次如 datasets/yolo/0001/data.yaml"
    )
