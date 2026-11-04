from pathlib import Path

import pytest

from tools.cfg import resolve_latest_path, substitute_variables


def test_resolve_latest_path_returns_newest_timestamped_dir(tmp_path):
    (tmp_path / "auto_annotate_20260720_120000").mkdir()
    (tmp_path / "auto_annotate_20260721_153022").mkdir()
    (tmp_path / "other_20260722_000000").mkdir()

    result = resolve_latest_path(str(tmp_path / "auto_annotate_*"))
    assert Path(result).name == "auto_annotate_20260721_153022"


def test_resolve_latest_path_matches_timestamp_then_tag(tmp_path):
    """archive.py 生成的 `<时间戳>_<tag>` 目录也要能被识别。"""
    (tmp_path / "20261020_120000_cjet_dataset").mkdir()
    (tmp_path / "20261021_153022_cjet_dataset").mkdir()

    result = resolve_latest_path(str(tmp_path / "*"))

    assert Path(result).name == "20261021_153022_cjet_dataset"


def test_resolve_latest_path_resolves_relative_pattern_against_project_root(
    tmp_path, monkeypatch
):
    """相对 pattern 以项目根为基准，与子进程工作目录保持一致。"""
    import tools.cfg as cfg_module

    (tmp_path / "run_20260721_153022").mkdir()
    monkeypatch.setattr(cfg_module, "PROJECT_ROOT", tmp_path)

    result = resolve_latest_path("run_*")

    assert Path(result).name == "run_20260721_153022"


def test_substitute_variables_latest(tmp_path):
    (tmp_path / "auto_annotate_20260721_153022").mkdir()
    pattern = str(tmp_path / "auto_annotate_*")
    command = f"--input ${{latest:{pattern}}}"
    result = substitute_variables(command, {})
    assert "auto_annotate_20260721_153022" in result
    assert "${latest:" not in result


def test_substitute_variables_latest_supports_nested_variable(tmp_path):
    """`${latest:${paths.archive}/*}` 里的内层变量要先解析。"""
    archive = tmp_path / "archive"
    (archive / "20261021_153022_tag").mkdir(parents=True)

    result = substitute_variables(
        "--input ${latest:${paths.archive}/*}", {"paths.archive": str(archive)}
    )

    assert result == "--input " + str(archive / "20261021_153022_tag")


def test_substitute_variables_latest_no_match_raises():
    """无匹配目录时必须报错，不能把占位符原样交给子进程。"""
    with pytest.raises(KeyError, match="无法解析的变量"):
        substitute_variables("--input ${latest:/nonexistent/path/*}", {})


def test_substitute_variables_unknown_variable_raises():
    with pytest.raises(KeyError, match=r"\$\{paths\.nope\}"):
        substitute_variables("--input ${paths.nope}", {})


def test_substitute_variables_replaces_known_variables():
    result = substitute_variables(
        "python a.py --in ${paths.a} --out ${paths.b}",
        {"paths.a": "datasets/a", "paths.b": "datasets/b"},
    )
    assert result == "python a.py --in datasets/a --out datasets/b"
