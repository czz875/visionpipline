import sys
from pathlib import Path

from tools import workflow


def test_run_stage_realtime_forwards_stderr_and_captures_stdout(
    tmp_path, capfd
):
    child_script = tmp_path / "child_stage.py"
    child_script.write_text(
        "import sys\n"
        'print("progress 1/1", file=sys.stderr, flush=True)\n'
        'print("OUTPUT_PATH:datasets/batch_test/01_annotated", flush=True)\n',
        encoding="utf-8",
    )
    log_path = tmp_path / "workflow.log"
    stage = {
        "name": "realtime_stage",
        "command": f'"{sys.executable}" "{child_script}"',
        "output_var": "annotated_output",
    }
    mapping = {}

    ok = workflow.run_stage(stage, mapping, dry_run=False, log_path=log_path)

    captured = capfd.readouterr()
    assert ok is True
    assert "progress 1/1" in captured.err
    assert "OUTPUT_PATH:datasets/batch_test/01_annotated" in captured.out
    assert (
        mapping["prev.annotated_output"]
        == "datasets/batch_test/01_annotated"
    )


def test_run_stage_realtime_forwards_failure_stderr(tmp_path, capfd):
    child_script = tmp_path / "failed_stage.py"
    child_script.write_text(
        "import sys\n"
        'print("failure detail", file=sys.stderr, flush=True)\n'
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "workflow.log"
    stage = {
        "name": "failed_stage",
        "command": f'"{sys.executable}" "{child_script}"',
    }

    ok = workflow.run_stage(stage, {}, dry_run=False, log_path=log_path)

    captured = capfd.readouterr()
    assert ok is False
    assert "failure detail" in captured.err
    assert "FAILED rc=3" in log_path.read_text(encoding="utf-8")


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
