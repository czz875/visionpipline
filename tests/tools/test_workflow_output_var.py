import sys
from pathlib import Path

from tools import workflow


def test_run_stage_captures_output_path(tmp_path):
    log_path = tmp_path / "workflow.log"
    stage = {
        "name": "fake_stage",
        "command": f"{sys.executable} -c \"print('OUTPUT_PATH:/tmp/fake_output')\"",
        "output_var": "fake_dir",
    }
    mapping = {}
    ok = workflow.run_stage(stage, mapping, dry_run=False, log_path=log_path)
    assert ok is True
    assert mapping.get("prev.fake_dir") == "/tmp/fake_output"


def test_run_stage_uses_output_param_as_fallback(tmp_path):
    log_path = tmp_path / "workflow.log"
    stage = {
        "name": "fake_stage",
        # 使用 echo 保证命令一定成功，同时让 --output 参数出现在 command 中，
        # 用于验证 stdout 没有 OUTPUT_PATH 时从 command 回退提取路径。
        "command": "echo no-output-path --output datasets/01_annotated",
        "output_var": "annotated_dir",
    }
    mapping = {}
    ok = workflow.run_stage(stage, mapping, dry_run=False, log_path=log_path)
    assert ok is True
    assert mapping.get("prev.annotated_dir") == "datasets/01_annotated"


def test_run_stage_captures_output_path_in_dry_run(tmp_path):
    log_path = tmp_path / "workflow.log"
    stage = {
        "name": "dry_stage",
        "command": "python tools/annotate/auto.py --output datasets/01_annotated",
        "output_var": "annotated_dir",
    }
    mapping = {}
    ok = workflow.run_stage(stage, mapping, dry_run=True, log_path=log_path)
    assert ok is True
    assert mapping.get("prev.annotated_dir") == "datasets/01_annotated"


def test_run_keeps_stages_without_order_sequential(tmp_path, capsys):
    log_path = tmp_path / "workflow.log"
    config = {
        "log_file": str(log_path),
        "stages": [
            {"name": "first", "command": "echo first"},
            {"name": "second", "command": "echo second"},
        ],
    }
    rc = workflow.run(config=config, dry_run=True)
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[并行]" not in captured
    assert ">>> 阶段：first" in captured
    assert ">>> 阶段：second" in captured
