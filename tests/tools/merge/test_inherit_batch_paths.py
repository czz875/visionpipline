from __future__ import annotations

import sys
from pathlib import Path

from tools.merge import inherit_dataset


def test_resolve_group_dirs_uses_same_batch(tmp_path: Path) -> None:
    """逻辑输入应解析到最新批次，并在同批次创建分组目录。"""
    encrypted_dir = tmp_path / "batch_20260729_120000" / "02_encrypted"
    encrypted_dir.mkdir(parents=True)

    source_dir, target_dir = inherit_dataset._resolve_group_dirs(
        tmp_path / "02_encrypted",
        tmp_path / "03_grouped",
    )

    assert source_dir == encrypted_dir
    assert target_dir == encrypted_dir.parent / "03_grouped"
    assert target_dir.is_dir()


def test_resolve_group_dirs_creates_batch_for_standalone_source(tmp_path: Path) -> None:
    """普通独立输入应为分组输出创建新的时间戳批次。"""
    source_dir = tmp_path / "standalone_source"
    source_dir.mkdir()

    actual_source, target_dir = inherit_dataset._resolve_group_dirs(
        source_dir,
        tmp_path / "03_grouped",
    )

    assert actual_source == source_dir
    assert target_dir.parent.parent == tmp_path
    assert target_dir.parent.name.startswith("batch_")
    assert target_dir.name == "03_grouped"
    assert target_dir.is_dir()


def test_resolve_group_dirs_does_not_create_batch_for_invalid_source(tmp_path: Path) -> None:
    """输入无效时不得创建会被后续流程误选的空批次。"""
    logical_source = tmp_path / "02_encrypted"
    logical_target = tmp_path / "03_grouped"

    source_dir, target_dir = inherit_dataset._resolve_group_dirs(
        logical_source,
        logical_target,
    )

    assert source_dir == logical_source
    assert target_dir == logical_target
    assert list(tmp_path.glob("batch_*")) == []


def test_resolve_group_dirs_does_not_create_batch_for_file_source(tmp_path: Path) -> None:
    """普通文件不是有效源目录，不得因此创建批次输出。"""
    source_file = tmp_path / "sample.png"
    source_file.write_bytes(b"image")
    logical_target = tmp_path / "03_grouped"

    source_dir, target_dir = inherit_dataset._resolve_group_dirs(
        source_file,
        logical_target,
    )

    assert source_dir == source_file
    assert target_dir == logical_target
    assert list(tmp_path.glob("batch_*")) == []


def test_main_groups_files_into_same_batch(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """命令行入口应把文件复制到当前批次的分组阶段。"""
    encrypted_dir = tmp_path / "batch_20260729_120000" / "02_encrypted"
    encrypted_dir.mkdir(parents=True)
    (encrypted_dir / "sample.png").write_bytes(b"encrypted-image")
    (encrypted_dir / "sample.json").write_bytes(b'{"shapes": []}')

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inherit_dataset.py",
            "--source",
            str(tmp_path / "02_encrypted"),
            "--target",
            str(tmp_path / "03_grouped"),
            "--batch-size",
            "1000",
            "--no-classify",
            "--recursive",
            "--apply",
        ],
    )

    result = inherit_dataset.main()

    grouped_dir = encrypted_dir.parent / "03_grouped"
    assert result == 0
    assert (grouped_dir / "0001" / "sample.png").read_bytes() == b"encrypted-image"
    assert (grouped_dir / "0001" / "sample.json").read_bytes() == b'{"shapes": []}'
    assert f"OUTPUT_PATH:{grouped_dir.resolve()}" in capsys.readouterr().out


def test_main_invalid_start_batches_does_not_create_output_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """参数校验失败时不得提前创建任何输出目录。"""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "sample.png").write_bytes(b"image")
    (source_dir / "sample.json").write_bytes(b'{"shapes": []}')

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inherit_dataset.py",
            "--source",
            str(source_dir),
            "--target",
            str(tmp_path / "03_grouped"),
            "--start-batches",
            "bad",
        ],
    )

    result = inherit_dataset.main()

    assert result == 1
    assert list(tmp_path.glob("batch_*")) == []
    assert not (tmp_path / "03_grouped").exists()
