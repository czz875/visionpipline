import sys
from pathlib import Path

import pytest

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


def test_output_path_binding_is_passed_to_the_next_stage(tmp_path):
    """准备阶段必须读取本轮实际批次，不能再读配置中的逻辑目录。"""
    logical = tmp_path / "datasets" / "01_annotated"
    actual = tmp_path / "datasets" / "batch_20261105_120000" / "01_annotated"
    actual.mkdir(parents=True)
    producer = tmp_path / "producer.py"
    producer.write_text(f"print({('OUTPUT_PATH:' + str(actual))!r})\n", encoding="utf-8")
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from pathlib import Path\nimport sys\n"
        "(Path(sys.argv[1]) / 'consumed.txt').write_text('current batch')\n",
        encoding="utf-8",
    )
    config = {
        "paths": {"annotated": str(logical)},
        "log_file": str(tmp_path / "workflow.log"),
        "stages": [
            {"name": "annotate", "command": f'"{sys.executable}" "{producer}"',
             "output_var": "paths.annotated"},
            {"name": "clean", "command": f'"{sys.executable}" "{consumer}" "${{paths.annotated}}"'},
        ],
    }

    assert workflow.run(config=config) == 0
    assert (actual / "consumed.txt").read_text() == "current batch"
    assert not logical.exists()


def test_resume_restores_skipped_producer_path_binding(tmp_path):
    """跳过标注后恢复清洗时，沿用该阶段最近一个已有批次。"""
    logical = tmp_path / "datasets" / "01_annotated"
    actual = tmp_path / "datasets" / "batch_20261105_120000" / "01_annotated"
    actual.mkdir(parents=True)
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from pathlib import Path\nimport sys\n"
        "(Path(sys.argv[1]) / 'resumed.txt').write_text('resumed')\n",
        encoding="utf-8",
    )
    config = {
        "paths": {"annotated": str(logical)},
        "log_file": str(tmp_path / "workflow.log"),
        "stages": [
            {"name": "annotate", "command": "exit 99", "output_var": "paths.annotated"},
            {"name": "clean", "command": f'"{sys.executable}" "{consumer}" "${{paths.annotated}}"'},
        ],
    }

    assert workflow.run(config=config, from_stage="clean") == 0
    assert (actual / "resumed.txt").read_text() == "resumed"


def test_dry_run_binding_keeps_quoted_output_paths_intact(tmp_path):
    output = tmp_path / "中文目录 带空格"
    mapping = {}
    stage = {
        "name": "annotate", "output_var": "paths.annotated",
        "command": f'echo --output "{output}"',
    }

    assert workflow.run_stage(stage, mapping, True, tmp_path / "workflow.log")
    assert mapping["paths.annotated"] == str(output)


@pytest.mark.parametrize("project_name", ["project", "project with spaces"])
def test_resume_resolves_relative_paths_against_project_root(tmp_path, monkeypatch, project_name):
    """Python API 从其它目录调用时，与子进程 cwd 使用同一个路径基准。"""
    project = tmp_path / project_name
    actual = project / "datasets" / "batch_20261105_120000" / "01_annotated"
    actual.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(workflow, "PROJECT_ROOT", project)
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from pathlib import Path\nimport sys\n"
        "(Path(sys.argv[1]) / 'resumed.txt').write_text('resumed')\n",
        encoding="utf-8",
    )
    config = {
        "paths": {"annotated": "datasets/01_annotated"},
        "log_file": str(tmp_path / "workflow.log"),
        "stages": [
            {"name": "annotate", "command": "exit 99", "output_var": "paths.annotated"},
            {"name": "clean", "command": f'"{sys.executable}" "{consumer}" ${{paths.annotated}}'},
        ],
    }

    assert workflow.run(config=config, from_stage="clean") == 0
    assert (actual / "resumed.txt").read_text() == "resumed"


def test_resume_does_not_restore_a_disabled_producer(tmp_path):
    logical = tmp_path / "datasets" / "01_annotated"
    logical.mkdir(parents=True)
    history = tmp_path / "datasets" / "batch_20261105_120000" / "01_annotated"
    history.mkdir(parents=True)
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from pathlib import Path\nimport sys\n"
        "(Path(sys.argv[1]) / 'consumed.txt').write_text('selected')\n", encoding="utf-8",
    )
    config = {
        "paths": {"annotated": str(logical)}, "log_file": str(tmp_path / "workflow.log"),
        "stages": [
            {"name": "annotate", "enabled": False, "command": "exit 99", "output_var": "paths.annotated"},
            {"name": "clean", "command": f'"{sys.executable}" "{consumer}" "${{paths.annotated}}"'},
        ],
    }
    assert workflow.run(config=config, from_stage="clean") == 0
    assert (logical / "consumed.txt").is_file()
    assert not (history / "consumed.txt").exists()
