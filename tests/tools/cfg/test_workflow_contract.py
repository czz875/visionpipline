"""工作流配置合并与断点续跑的回归测试。

覆盖审查发现的契约脱节：
- ``stages_only`` 下项目层对同名 stage 的字段覆盖必须生效；
- ``--from-stage`` / ``--to-stage`` 拼错时要报错，不能静默空跑；
- 默认入口在 ``tools/cfg/workflow_config.yaml`` 缺失时要回退。
"""

from pathlib import Path

import pytest
import yaml

from tools.cfg import resolve_config
from tools.workflow import run


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_stages_only_keeps_project_level_stage_overrides(tmp_path: Path) -> None:
    """项目层在 stages_only 模式下也必须能覆盖 workflow.yaml 的同名 stage。"""
    project_cfg = _write_yaml(
        tmp_path / "project.yaml",
        {
            "stages_only": ["inherit_dataset"],
            "parameters": {"apply_flag": ""},
            "stages": [
                {
                    "name": "inherit_dataset",
                    "command": (
                        "python tools/merge/inherit_dataset.py "
                        "--source ${paths.autolabel} ${parameters.apply_flag}"
                    ),
                }
            ],
        },
    )

    resolved = resolve_config(project_cfg)

    assert [s["name"] for s in resolved["stages"]] == ["inherit_dataset"]
    assert "${parameters.apply_flag}" in resolved["stages"][0]["command"]
    # 项目层覆盖生效，不应带上 workflow.yaml 里的 --apply
    assert "--apply" not in resolved["stages"][0]["command"]


def test_stages_only_rejects_unknown_stage_name(tmp_path: Path) -> None:
    project_cfg = _write_yaml(
        tmp_path / "project.yaml",
        {"stages_only": ["不存在的阶段"]},
    )

    with pytest.raises(KeyError, match="不存在的阶段"):
        resolve_config(project_cfg)


def test_from_stage_typo_returns_error_instead_of_silent_success(
    tmp_path: Path, capsys
) -> None:
    """--from-stage 拼错时必须返回非 0，而不是「全部跳过 + 完成」。"""
    config = {
        "stages": [
            {"name": "a", "command": "echo a"},
            {"name": "b", "command": "echo b"},
        ],
        "log_file": str(tmp_path / "workflow.log"),
    }

    rc = run(config=config, dry_run=True, from_stage="typo")

    assert rc == 1
    assert "--from-stage" in capsys.readouterr().out


def test_to_stage_typo_returns_error(tmp_path: Path) -> None:
    config = {
        "stages": [{"name": "a", "command": "echo a"}],
        "log_file": str(tmp_path / "workflow.log"),
    }

    assert run(config=config, dry_run=True, to_stage="typo") == 1


def test_to_stage_stops_before_named_stage(tmp_path: Path) -> None:
    config = {
        "stages": [
            {"name": "a", "command": "echo a"},
            {"name": "manual_review", "command": "echo review"},
            {"name": "b", "command": "echo b"},
        ],
        "log_file": str(tmp_path / "workflow.log"),
    }

    assert run(config=config, dry_run=True, to_stage="manual_review") == 0
    log = (tmp_path / "workflow.log").read_text(encoding="utf-8")
    assert "echo a" in log
    assert "echo review" not in log
    assert "echo b" not in log


def test_unresolved_variable_fails_instead_of_passing_literal(tmp_path: Path) -> None:
    """未知变量必须报错，不能把 ${paths.nope} 原样交给子进程。"""
    config = {
        "stages": [{"name": "a", "command": "echo ${paths.nope}"}],
        "log_file": str(tmp_path / "workflow.log"),
    }

    assert run(config=config, dry_run=True) == 1
