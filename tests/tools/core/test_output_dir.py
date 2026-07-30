import re
from pathlib import Path

from tools.core.output_dir import (
    build_batch_stage_dir,
    build_related_batch_stage_dir,
    build_timestamped_output_dir,
    resolve_latest_batch_stage_dir,
)


def test_build_timestamped_output_dir_creates_timestamped_subdir(tmp_path):
    base = tmp_path / "out"
    result = build_timestamped_output_dir(base, "auto_annotate")
    assert result.parent == base
    assert re.match(r"auto_annotate_\d{8}_\d{6}", result.name)
    assert result.exists()


def test_build_timestamped_output_dir_avoids_collision(tmp_path):
    base = tmp_path / "out"
    first = build_timestamped_output_dir(base, "auto_annotate")
    second = build_timestamped_output_dir(base, "auto_annotate")
    assert second.name.startswith(first.name)
    assert second != first
    assert re.search(r"_\d{3}$", second.name)


def test_build_batch_stage_dir_creates_stage_in_timestamped_batch(tmp_path):
    result = build_batch_stage_dir(tmp_path / "01_annotated")

    assert result.parent.parent == tmp_path
    assert result.name == "01_annotated"
    assert re.fullmatch(r"batch_\d{8}_\d{6}(?:_\d{3})?", result.parent.name)
    assert result.is_dir()


def test_resolve_latest_batch_stage_dir_ignores_unrelated_dirs(tmp_path):
    old = tmp_path / "batch_20260729_100000" / "01_annotated"
    latest = tmp_path / "batch_20260729_110000_001" / "01_annotated"
    unrelated = tmp_path / "batch_backup" / "01_annotated"
    for stage_dir in (old, latest, unrelated):
        stage_dir.mkdir(parents=True)

    result = resolve_latest_batch_stage_dir(tmp_path / "01_annotated")

    assert result == latest


def test_resolve_latest_batch_stage_dir_returns_logical_path_without_batch(tmp_path):
    logical = tmp_path / "01_annotated"

    assert resolve_latest_batch_stage_dir(logical) == logical


def test_build_related_batch_stage_dir_creates_sibling_in_same_batch(tmp_path):
    source = tmp_path / "batch_20260729_110000" / "01_annotated"
    source.mkdir(parents=True)

    result = build_related_batch_stage_dir(source, tmp_path / "02_encrypted")

    assert result == source.parent / "02_encrypted"
    assert result.is_dir()


def test_resolve_latest_batch_stage_dir_sorts_large_sequence_numerically(tmp_path):
    sequence_999 = tmp_path / "batch_20260729_110000_999" / "01_annotated"
    sequence_1000 = tmp_path / "batch_20260729_110000_1000" / "01_annotated"
    sequence_999.mkdir(parents=True)
    sequence_1000.mkdir(parents=True)

    result = resolve_latest_batch_stage_dir(tmp_path / "01_annotated")

    assert result == sequence_1000


def test_resolve_latest_batch_stage_dir_skips_latest_batch_without_stage(tmp_path):
    older_stage = tmp_path / "batch_20260729_100000" / "01_annotated"
    latest_batch = tmp_path / "batch_20260729_110000"
    older_stage.mkdir(parents=True)
    latest_batch.mkdir()

    result = resolve_latest_batch_stage_dir(tmp_path / "01_annotated")

    assert result == older_stage


def test_resolve_latest_batch_stage_dir_keeps_existing_batch_path(tmp_path):
    stage_dir = tmp_path / "batch_20260729_110000_1000" / "01_annotated"

    assert resolve_latest_batch_stage_dir(stage_dir) == stage_dir


def test_full_batch_path_chain_uses_single_batch_root(tmp_path):
    annotated = build_batch_stage_dir(tmp_path / "01_annotated")
    merged = resolve_latest_batch_stage_dir(tmp_path / "01_annotated")
    encrypted = build_related_batch_stage_dir(merged, tmp_path / "02_encrypted")
    grouped = build_related_batch_stage_dir(encrypted, tmp_path / "03_grouped")

    assert merged == annotated
    assert annotated.parent == encrypted.parent == grouped.parent
    assert [annotated.name, encrypted.name, grouped.name] == [
        "01_annotated",
        "02_encrypted",
        "03_grouped",
    ]
