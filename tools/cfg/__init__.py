"""
tools/cfg
=========

工作流配置加载、变量替换、合并逻辑（仿 ultralytics/cfg）。

公开 API：
- ``load_config(path)``：加载单个 YAML/JSON
- ``flatten_dict(data)``：嵌套字典拍平为 ``prefix.key -> str``
- ``substitute_variables(command, mapping)``：替换命令中的 ``${prefix.key}``
- ``merge_configs(*configs)``：浅合并多个配置 dict（后写覆盖前写）
- ``resolve_config(project_path)``：根据项目路径自动加载
  ``tools/cfg/default.yaml`` + ``tools/cfg/workflow.yaml`` + 项目自己的覆盖文件，
  返回合并后的 dict。
- ``iter_stage_files()``：列出 ``tools/cfg/`` 下所有 ``*.yaml``（按字母序）

``resolve_config`` 还支持在 project 顶层加 ``stages_only: [name1, name2, ...]``
字段：列出的 stage 才会被跑（按顺序从 workflow.yaml 拿完整定义），其它 stage
全部跳过。适合做"任务专项精简工作流"。

典型用法：

    from tools.cfg import load_config, resolve_config

    cfg = load_config(Path("tools/cfg/workflow.yaml"))
    cfg = resolve_config(Path("workflow_config.yaml"))
    cfg = resolve_config(Path("tools/cfg/inherit_yolo.yaml"))   # 走 stages_only
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from typing import Iterable

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# 1. 默认常量
# =============================================================================

CFG_DIR = Path(__file__).resolve().parent          # tools/cfg/
PROJECT_ROOT = CFG_DIR.resolve().parent.parent    # 项目根目录
DEFAULT_CFG_PATH = CFG_DIR / "default.yaml"
WORKFLOW_CFG_PATH = CFG_DIR / "workflow.yaml"      # 系统主工作流（完整 stage 定义）
EXAMPLE_CFG_PATH = CFG_DIR / "workflow.example.yaml"
PROJECT_CFG_PATH = CFG_DIR / "workflow_config.yaml"        # 项目级覆盖入口
PROJECT_EXAMPLE_CFG_PATH = CFG_DIR / "workflow_config.yaml.example"  # 同上示例
DEFAULT_LOG = Path("workflow.log")                 # 默认日志名（相对项目根）

_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")
_LATEST_TIMESTAMP_RE = re.compile(r".*_(\d{8}_\d{6})(?:_\d+)?$")


# =============================================================================
# 2. 加载 / 解析
# =============================================================================


def load_config(path: Path) -> dict:
    """加载单个 YAML 或 JSON 配置文件，返回 dict。

    失败时抛 ``ImportError``（YAML 缺 PyYAML）或 ``ValueError``（格式不支持）。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    # 允许 .yaml.example / .yml.example 这类「示例」后缀按 YAML 解析，
    # 方便直接 --config 预览，不必先复制成 .yaml。
    if suffix == ".example":
        stem = path.stem.lower()
        if stem.endswith(".yaml") or stem.endswith(".yml"):
            suffix = ".yaml"
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # PyYAML 是项目间接依赖
        except ImportError as e:
            raise ImportError(
                "读取 YAML 需要 PyYAML，请安装：pip install pyyaml"
            ) from e
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"不支持的配置文件格式：{suffix}")


def flatten_dict(
    data: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, str]:
    """把嵌套字典拍平为 ``prefix.key -> str`` 映射，用于变量替换。"""
    items: dict[str, str] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep))
        else:
            items[new_key] = str(value)
    return items


def resolve_latest_path(pattern: str) -> str | None:
    """按 glob pattern 查找带时间戳的目录，返回最新一个目录路径；无匹配返回 ``None``。"""
    candidates: list[tuple[str, str]] = []
    for path_str in glob.glob(pattern):
        path = Path(path_str)
        if not path.is_dir():
            continue
        m = _LATEST_TIMESTAMP_RE.match(path.name)
        if not m:
            continue
        candidates.append((m.group(1), path_str))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def substitute_variables(command: str, mapping: dict[str, str]) -> str:
    """把命令中的 ``${prefix.key}`` 占位符替换为 ``mapping`` 里的值；找不到则保留原占位符。

    支持 ``${latest:glob_pattern}`` 特殊变量：匹配到的最新时间戳目录会被替换为目录路径；
    无匹配时打印警告并保留原占位符。
    """
    def _repl(match: re.Match) -> str:
        var = match.group(1)
        if var.startswith("latest:"):
            pattern = var[7:]
            latest = resolve_latest_path(pattern)
            if latest is None:
                print(f"警告：未找到匹配目录，保留占位符 ${{{var}}}")
                return match.group(0)
            return latest
        return mapping.get(var, match.group(0))
    return _PLACEHOLDER_RE.sub(_repl, command)


def merge_configs(*configs: dict) -> dict:
    """浅合并多个 dict；后面的覆盖前面的。"""
    merged: dict = {}
    for cfg in configs:
        if not cfg:
            continue
        for key, value in cfg.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


# =============================================================================
# 3. 项目级配置解析
# =============================================================================


def iter_stage_files() -> list[Path]:
    """列出 ``tools/cfg/`` 下所有 ``*.yaml``，按字母序。"""
    return sorted(CFG_DIR.glob("*.yaml"))


def resolve_config(project_path: Path | None = None) -> dict:
    """根据项目入口文件路径加载完整配置。

    加载顺序（后写覆盖前写）：
    1. ``tools/cfg/default.yaml``（系统默认）
    2. ``tools/cfg/workflow.yaml``（项目工作流基线）
    3. ``project_path``（项目根的覆盖文件，可选）

    Args:
        project_path: 项目根下的覆盖 yaml，``None`` 时只加载前两份。

    Returns:
        合并后的 dict，``stages`` 列表也会被合并。

    stages 合并规则：
    - project 顶层含 ``stages_only`` 字段：只跑列出的 stage（按顺序从
      workflow.yaml 里拿对应完整定义），其它 stage 全部跳过。
    - 否则按 stage name 去重拼接（先出现保留），这是默认行为。
    """
    layers: list[dict] = []
    if DEFAULT_CFG_PATH.exists():
        layers.append(load_config(DEFAULT_CFG_PATH))
    if WORKFLOW_CFG_PATH.exists():
        layers.append(load_config(WORKFLOW_CFG_PATH))
    project_layer: dict = {}
    if project_path is not None:
        project_path = Path(project_path)
        if project_path.exists():
            project_layer = load_config(project_path)
            layers.append(project_layer)

    # dict 浅合并
    merged = merge_configs(*layers)

    # stages 列表需要特殊处理（merge_configs 浅合并会被覆盖，所以手动拼）
    if "stages_only" in project_layer:
        # 按 project 列出的 name 顺序从所有 layer 拿 stage
        stage_pool: dict[str, dict] = {}
        for layer in layers:
            for stage in layer.get("stages", []) or []:
                name = stage.get("name", "")
                if name and name not in stage_pool:
                    stage_pool[name] = stage
        merged["stages"] = [
            dict(stage_pool[name]) for name in project_layer["stages_only"]
            if name in stage_pool
        ]
    else:
        # 字段级合并：项目层（后加载）按 name 覆盖单个键（如 order / enabled /
        # command），其余键（如 workflow.yaml 提供的 command）保留；先出现的
        # name 决定整体顺序。这样示例配置才能给 workflow.yaml 的同名 stage
        # 追加 order / enabled 而不会被忽略。
        stages_pool: dict[str, dict] = {}
        seen_order: list[str] = []
        for layer in layers:
            for stage in layer.get("stages", []) or []:
                name = stage.get("name", "")
                if not name:
                    continue
                if name in stages_pool:
                    stages_pool[name] = {**stages_pool[name], **stage}
                else:
                    stages_pool[name] = dict(stage)
                    seen_order.append(name)
        merged["stages"] = [stages_pool[n] for n in seen_order]
    return merged


__all__ = [
    "CFG_DIR",
    "DEFAULT_CFG_PATH",
    "DEFAULT_LOG",
    "PROJECT_CFG_PATH",
    "PROJECT_EXAMPLE_CFG_PATH",
    "PROJECT_ROOT",
    "WORKFLOW_CFG_PATH",
    "flatten_dict",
    "iter_stage_files",
    "load_config",
    "merge_configs",
    "resolve_config",
    "resolve_latest_path",
    "substitute_variables",
]
