"""
工作流编排器（仿 ultralytics 风格的薄编排层）。

只负责"按顺序跑 stage"这一件事；YAML 解析、变量替换、配置合并都搬到了
``tools.cfg``。默认从 ``tools/cfg/workflow_config.yaml`` 加载项目入口，
自动叠加 ``tools/cfg/default.yaml`` + ``tools/cfg/workflow.yaml``。

典型用法：

    # 预览
    python tools/workflow.py --dry-run

    # 真跑第零段
    python tools/workflow.py --from-stage backup_snapshot --to-stage rename_with_labelme_sync

    # 也可显式指定其它入口（如直接用系统主工作流）
    python tools/workflow.py --config tools/cfg/workflow.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.cfg import (
    DEFAULT_LOG,
    PROJECT_CFG_PATH,
    PROJECT_ROOT,
    flatten_dict,
    resolve_config,
    substitute_variables,
)


# =============================================================================
# 0. 子进程 PATH：让 stage 里的 `python` / `yolo` 走到项目自带的 .conda
# =============================================================================
# tools/workflow.py 本身是被 .conda\python.exe 启动的，但 stage command 里的
# `python` / `yolo` 命令走 shell PATH 查找。把 .conda 和 .conda\Scripts
# 显式 prepend 到 os.environ["PATH"]，避免子进程 shell 找不到解释器
# （同时让 ultralytics 装的 `yolo` CLI 也都能找到）。
_CONDA_DIR = PROJECT_ROOT / ".conda"
if _CONDA_DIR.is_dir() and sys.platform == "win32":
    for sub in ("", "Scripts"):
        p = str(_CONDA_DIR / sub) if sub else str(_CONDA_DIR)
        if p not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_CONFIG = PROJECT_CFG_PATH                 # 默认 tools/cfg/workflow_config.yaml
DEFAULT_LOG_FILE = DEFAULT_LOG                    # workflow.log（在 tools/cfg/__init__.py 里定义）


# =============================================================================
# 2. Stage 执行
# =============================================================================


def run_stage(
    stage: dict,
    mapping: dict[str, str],
    dry_run: bool,
    log_path: Path,
) -> bool:
    """执行单个 stage。成功返回 ``True``，失败返回 ``False``。"""
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

    result = subprocess.run(command, shell=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"    [失败] 阶段 {name} 返回码 {result.returncode}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"[{name}] FAILED rc={result.returncode}\n"
            )
        return False

    print(f"    [完成] 阶段 {name}")
    return True


# =============================================================================
# 3. CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="CJET 数据生产工作流编排器。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "配置文件路径；可为项目根的 workflow_config.yaml（兼容老路径，"
            "自动叠加 tools/cfg/default.yaml + tools/cfg/workflow.yaml），"
            f"或直接传 tools/cfg/workflow.yaml。默认：{DEFAULT_CONFIG}"
        ),
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


def run(
    cfg: Path | str | None = None,
    *,
    dry_run: bool = False,
    from_stage: str = "",
    to_stage: str = "",
) -> int:
    """跑工作流（Python API 版，不依赖 sys.argv）。

    等价于命令行：
        python tools/workflow.py --config <cfg> [--dry-run] \
            [--from-stage X] [--to-stage Y]

    ``cfg`` 为 None 时走 ``tools/cfg/workflow_config.yaml`` 默认入口。
    """
    config_path = Path(cfg) if cfg is not None else PROJECT_CFG_PATH
    if not config_path.exists():
        print(f"[错误] 配置文件不存在：{config_path}")
        return 1

    # 走 tools.cfg.resolve_config：自动叠加 default + workflow + 用户配置
    config = resolve_config(config_path)
    mapping = flatten_dict(config)
    mapping["date"] = datetime.now().strftime("%Y%m%d")
    mapping["datetime"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapping["project_root"] = str(PROJECT_ROOT)

    log_path = Path(config.get("log_file", str(DEFAULT_LOG_FILE)))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stages = config.get("stages", [])
    if not stages:
        print("[警告] 配置文件中没有定义 stages。")
        return 0

    started = not from_stage
    for stage in stages:
        name = stage.get("name", "unnamed")

        if to_stage and name == to_stage:
            print(f"\n[暂停] 已到达阶段 '{name}' 前，等待人工处理。")
            return 0

        if not stage.get("enabled", True):
            continue

        if not started and from_stage:
            if name == from_stage:
                started = True
            else:
                print(f"[跳过] {name}")
                continue

        if not run_stage(stage, mapping, dry_run, log_path):
            print(f"\n[终止] 阶段 {name} 失败，工作流中断。")
            return 1

    print("\n[完成] 工作流全部阶段执行完毕。")
    return 0


def main() -> int:
    """CLI 入口（薄壳，参数解析后转给 run()）。"""
    parser = build_parser()
    args = parser.parse_args()
    return run(
        cfg=args.config,
        dry_run=args.dry_run,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
