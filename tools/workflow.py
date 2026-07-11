"""
工作流编排器。

读取 YAML/JSON 配置文件，按顺序执行各个阶段。每个阶段调用一个现有工具脚本，
从而把“补充数据 -> 标注 -> 合并 -> 清洗 -> 拆分 -> YOLO -> 训练 -> 自标注 ->
回 LabelMe -> 再清洗 -> 归档”整条链路串起来。

典型用法：

    python tools/workflow.py --config workflow_config.yaml

只预览不执行：

    python tools/workflow.py --config workflow_config.yaml --dry-run

从某个阶段开始执行：

    python tools/workflow.py --config workflow_config.yaml --from-stage train_yolo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re

DEFAULT_CONFIG = Path("workflow_config.yaml")
DEFAULT_LOG = Path("workflow.log")


def load_config(path: Path) -> dict:
    """加载 YAML 或 JSON 配置文件。"""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "读取 YAML 需要 PyYAML，请安装：pip install pyyaml"
            ) from e
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"不支持的配置文件格式：{suffix}")


def flatten_dict(
    data: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, str]:
    """把嵌套字典拍平为 ``prefix.key`` -> str 的映射，用于变量替换。"""
    items: dict[str, str] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep))
        else:
            items[new_key] = str(value)
    return items


_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def substitute_variables(command: str, mapping: dict[str, str]) -> str:
    """将命令中的 ``${prefix.key}`` 占位符替换为实际值。

    支持点号分隔的嵌套键（如 ``${paths.raw_data}``）。未找到时保留原占位符。
    """
    def _repl(match: re.Match) -> str:
        key = match.group(1)
        return mapping.get(key, match.group(0))

    return _PLACEHOLDER_RE.sub(_repl, command)


def run_stage(
    stage: dict,
    mapping: dict[str, str],
    dry_run: bool,
    log_path: Path,
) -> bool:
    """执行单个阶段。

    Returns:
        成功返回 ``True``，失败返回 ``False``。
    """
    name = stage.get("name", "unnamed")
    raw_command = stage.get("command", "")
    if not raw_command:
        print(f"[跳过] 阶段 {name} 没有配置 command。")
        return True

    command = substitute_variables(raw_command, mapping)
    now = datetime.now().isoformat(timespec="seconds")

    print(f"\n[{now}] >>> 阶段：{name}")
    print(f"    {command}")

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{now}] [{name}] {command}\n")

    if dry_run:
        print("    (dry-run，未实际执行)")
        return True

    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"    [失败] 阶段 {name} 返回码 {result.returncode}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{datetime.now().isoformat(timespec='seconds')}] [{name}] FAILED rc={result.returncode}\n")
        return False

    print(f"    [完成] 阶段 {name}")
    return True


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="LabelMe / YOLO 训练数据工作流编排器。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"工作流配置文件（默认：{DEFAULT_CONFIG}）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印每个阶段将要执行的命令，不真正运行。",
    )
    parser.add_argument(
        "--from-stage",
        default="",
        help="从指定名称的阶段开始执行，之前的阶段跳过。",
    )
    parser.add_argument(
        "--to-stage",
        default="",
        help="执行到指定名称的阶段之前停止（不包括该阶段）。",
    )
    return parser


def main() -> int:
    """脚本入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[错误] 配置文件不存在：{args.config}")
        return 1

    config = load_config(args.config)
    mapping = flatten_dict(config)
    mapping["date"] = datetime.now().strftime("%Y%m%d")
    mapping["datetime"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = Path(config.get("log_file", str(DEFAULT_LOG)))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stages = config.get("stages", [])
    if not stages:
        print("[警告] 配置文件中没有定义 stages。")
        return 0

    started = not args.from_stage
    for stage in stages:
        name = stage.get("name", "unnamed")

        if args.to_stage and name == args.to_stage:
            print(f"\n[暂停] 已到达阶段 '{name}' 前，等待人工处理。")
            return 0

        if not stage.get("enabled", True):
            continue

        if not started and args.from_stage:
            if name == args.from_stage:
                started = True
            else:
                print(f"[跳过] {name}")
                continue

        if not run_stage(stage, mapping, args.dry_run, log_path):
            print(f"\n[终止] 阶段 {name} 失败，工作流中断。")
            return 1

    print("\n[完成] 工作流全部阶段执行完毕。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
