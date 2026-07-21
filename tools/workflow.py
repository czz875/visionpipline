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

每个 stage 支持两个开关：`enabled: true/false` 控制是否启用；`order: <整数>`
控制运行顺序——相同 order 的 stage 并行同时启动，不同 order 按整数升序顺序执行
（不写 order 则退化为各自独立顺序执行）。详见 tools/cfg/all_modules.yaml.example。
"""

from __future__ import annotations

import argparse
import os
import re
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
# 从 PROJECT_ROOT 向上逐级查找项目自带的 .conda（worktree 嵌套更深也能命中，
# 例如 .worktrees/<name> 下往上两级才是主仓库根的 .conda）。
_conda_dir = None
for _cur in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    _cand = _cur / ".conda"
    if _cand.is_dir():
        _conda_dir = _cand
        break

if _conda_dir is not None and sys.platform == "win32":
    for sub in ("", "Scripts"):
        p = str(_conda_dir / sub) if sub else str(_conda_dir)
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

_OUTPUT_PATH_RE = re.compile(r"^OUTPUT_PATH:(.+)$", re.MULTILINE)


def _extract_output_path(stdout_text: str, command: str) -> str | None:
    """从 stdout 或 command 的 --output 参数提取输出路径。"""
    m = _OUTPUT_PATH_RE.search(stdout_text)
    if m:
        return m.group(1).strip()
    m = re.search(r"--output\s+(\S+)", command)
    if m:
        return m.group(1)
    return None


def _capture_output_var(
    stage: dict,
    stdout_text: str,
    command: str,
    mapping: dict[str, str],
) -> None:
    """若 stage 声明了 output_var，从 stdout 或 command 的 --output 提取路径并写入 mapping。"""
    output_var = stage.get("output_var", "")
    if not output_var:
        return
    name = stage.get("name", "unnamed")
    output_path = _extract_output_path(stdout_text, command)
    if output_path:
        mapping[f"prev.{output_var}"] = output_path
        print(f"    [变量] prev.{output_var} = {output_path}")
    else:
        print(
            f"    [警告] 阶段 {name} 声明了 output_var={output_var}，"
            f"但未提取到输出路径"
        )


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
        _capture_output_var(stage, "", command, mapping)
        return True

    result = subprocess.run(
        command, shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"    {line}")

    if result.returncode != 0:
        print(f"    [失败] 阶段 {name} 返回码 {result.returncode}")
        if result.stderr:
            for line in result.stderr.splitlines():
                print(f"    {line}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"[{name}] FAILED rc={result.returncode}\n"
            )
        return False

    print(f"    [完成] 阶段 {name}")

    _capture_output_var(stage, result.stdout, command, mapping)

    return True


# =============================================================================
# 2b. 排序与并行分组
# =============================================================================


def _order_value(stage: dict, index: int) -> int:
    """取 stage 的 order 字段用于排序；缺失或非整数时退化为列表位置，保序执行。"""
    order = stage.get("order")
    if order is None:
        return index
    try:
        return int(order)
    except (TypeError, ValueError):
        return index


def run_group(
    group: list[dict],
    mapping: dict[str, str],
    dry_run: bool,
    log_path: Path,
) -> bool:
    """执行同一 order 值的阶段组。

    组内启用的阶段 >1 个时并行（``subprocess.Popen`` 同时启动），否则退化为单阶段
    顺序执行。全部成功返回 ``True``。``enabled: false`` 的 stage 会跳过。
    """
    active = [s for s in group if s.get("enabled", True)]
    if not active:
        return True
    if len(active) == 1:
        return run_stage(active[0], mapping, dry_run, log_path)

    names = [s.get("name", "unnamed") for s in active]
    print(f"\n[并行] 同序组同时启动 {len(active)} 个阶段：{', '.join(names)}")
    if dry_run:
        for s in active:
            run_stage(s, mapping, True, log_path)
        return True

    procs: list[tuple[dict, str, subprocess.Popen]] = []
    for s in active:
        command = substitute_variables(s.get("command", ""), mapping)
        now = datetime.now().isoformat(timespec="seconds")
        print(f"\n[{now}] >>> [并行] 阶段：{s.get('name', 'unnamed')}\n    {command}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{now}] [{s.get('name', 'unnamed')}] {command}\n")
        procs.append(
            (
                s,
                command,
                subprocess.Popen(
                    command,
                    shell=True,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
        )

    ok = True
    for s, command, proc in procs:
        stdout, stderr = proc.communicate()
        name = s.get("name", "unnamed")

        if stdout:
            for line in stdout.splitlines():
                print(f"    {line}")

        if proc.returncode != 0:
            print(f"    [失败] 阶段 {name} 返回码 {proc.returncode}")
            if stderr:
                for line in stderr.splitlines():
                    print(f"    {line}")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"[{name}] FAILED rc={proc.returncode}\n"
                )
            ok = False
        else:
            print(f"    [完成] 阶段 {name}")
            _capture_output_var(s, stdout, command, mapping)
    return ok


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
    config: dict | None = None,
    overrides: dict[str, str] | None = None,
    **kwargs: str,
) -> int:
    """跑工作流（Python API 版，不依赖 sys.argv）。

    等价于命令行：
        python tools/workflow.py --config <cfg> [--dry-run] \
            [--from-stage X] [--to-stage Y]

    ``cfg`` 为 None 时走 ``tools/cfg/workflow_config.yaml`` 默认入口。

    ``config`` 为已解析好的 dict 时直接用它（跳过文件加载），便于在 Python 里
    动态改写 stages / paths / parameters 后再跑——这就是「参数覆盖 yaml」的入口。
    ``overrides`` 为 ``${prefix.key}`` → 新值的映射，会覆盖 yaml 同名变量，
    等价于不改文件、只在 Python 里临时改某个参数（见 ``src/main.py`` 示例）。

    任意 ``paths.<key>`` 都可当成命名参数直接传，例如 ``input_dir=...``、
    ``output_dir=...``、``encrypt_in=...``、``decrypt_out=...``，传了就等价于在
    ``overrides`` 里写 ``paths.<key>``（见 ``src/clean_orphans.py`` 等示例）。
    """
    if config is not None:
        resolved = config
    else:
        config_path = Path(cfg) if cfg is not None else PROJECT_CFG_PATH
        if not config_path.exists():
            print(f"[错误] 配置文件不存在：{config_path}")
            return 1
        resolved = resolve_config(config_path)

    mapping = flatten_dict(resolved)
    mapping["date"] = datetime.now().strftime("%Y%m%d")
    mapping["datetime"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapping["project_root"] = str(PROJECT_ROOT)
    if overrides:
        # 变量层覆盖：覆盖 yaml 里的 ${prefix.key}，实现 Python 内改参。
        mapping.update({str(k): str(v) for k, v in overrides.items()})

    # 命名参数便捷写法：把 input_dir / output_dir / encrypt_in ... 等透传成
    # paths.<key> 覆盖，等价于在 overrides 里写对应键。
    for key, val in kwargs.items():
        mapping[f"paths.{key}"] = str(val)

    log_path = Path(resolved.get("log_file", str(DEFAULT_LOG_FILE)))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stages = resolved.get("stages", [])
    if not stages:
        print("[警告] 配置文件中没有定义 stages。")
        return 0

    # 按 order 分组：order 相同 → 同一组，组内并行；组间按 order 升序顺序执行。
    # 未写 order 的 stage 退化成「列表位置」，即各自独立成组、保序顺序执行。
    ordered = sorted(
        enumerate(stages),
        key=lambda kv: (_order_value(kv[1], kv[0]), kv[0]),
    )
    groups: list[list[dict]] = []
    for _idx, stage in ordered:
        if (
            groups
            and groups[-1][-1].get("order") is not None
            and stage.get("order") is not None
            and int(groups[-1][-1]["order"]) == int(stage["order"])
        ):
            groups[-1].append(stage)
        else:
            groups.append([stage])

    started = not from_stage
    for group in groups:
        names = [s.get("name", "") for s in group]

        if to_stage and to_stage in names:
            print(f"\n[暂停] 已到达阶段 '{to_stage}' 前，等待人工处理。")
            return 0

        if from_stage and not started:
            if from_stage in names:
                started = True
            else:
                print(f"[跳过] 同序组：{', '.join(names)}")
                continue

        if not started:
            continue

        if not run_group(group, mapping, dry_run, log_path):
            print(f"\n[终止] 同序组内某阶段失败，工作流中断。")
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
