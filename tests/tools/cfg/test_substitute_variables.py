from pathlib import Path

from tools.cfg import resolve_latest_path, substitute_variables


def test_resolve_latest_path_returns_newest_timestamped_dir(tmp_path):
    (tmp_path / "auto_annotate_20260720_120000").mkdir()
    (tmp_path / "auto_annotate_20260721_153022").mkdir()
    (tmp_path / "other_20260722_000000").mkdir()

    result = resolve_latest_path(str(tmp_path / "auto_annotate_*"))
    assert Path(result).name == "auto_annotate_20260721_153022"


def test_substitute_variables_latest(tmp_path):
    (tmp_path / "auto_annotate_20260721_153022").mkdir()
    pattern = str(tmp_path / "auto_annotate_*")
    command = f"--input ${{latest:{pattern}}}"
    result = substitute_variables(command, {})
    assert "auto_annotate_20260721_153022" in result
    assert "${latest:" not in result


def test_substitute_variables_latest_no_match_keeps_placeholder():
    command = "--input ${latest:/nonexistent/path/*}"
    result = substitute_variables(command, {})
    assert result == command
