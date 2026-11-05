"""备份前必须校验完整源列表，不能遗漏源目录后仍报告成功。"""

from pathlib import Path
import tarfile

import pytest

from tools.backup.snapshot import _build_tar, main, snapshot_sources


def test_missing_source_fails_before_writing_any_archive(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "sample.txt").write_text("payload", encoding="utf-8")
    output = tmp_path / "backups"

    with pytest.raises(FileNotFoundError, match="源目录"):
        snapshot_sources([valid, tmp_path / "missing"], output, dry_run=False)

    assert not output.exists()


def test_cli_missing_source_returns_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", [
        "snapshot", "--sources", str(tmp_path / "missing"),
        "--output-dir", str(tmp_path / "out"), "--apply",
    ])
    assert main() == 1


def test_output_inside_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="源目录"):
        snapshot_sources([source], source / "backup", dry_run=False)
    assert not (source / "backup").exists()


def test_same_named_sources_are_rejected_before_writing(tmp_path: Path) -> None:
    sources = [tmp_path / "a" / "data", tmp_path / "b" / "data"]
    for source in sources:
        source.mkdir(parents=True)
    with pytest.raises(ValueError, match="同名"):
        snapshot_sources(sources, tmp_path / "backup", dry_run=False)
    assert not (tmp_path / "backup").exists()


def test_backup_remains_readable_with_relative_member_names(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.txt").write_text("payload", encoding="utf-8")
    plans = snapshot_sources([source], tmp_path / "backup", dry_run=False)
    with tarfile.open(plans[0][1], "r:gz") as archive:
        member = archive.extractfile("source/sample.txt")
        assert member is not None
        assert member.read() == b"payload"


def test_failed_write_does_not_leave_a_partial_backup(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "backup"

    def fail_read(*args, **kwargs):
        raise OSError("读取源文件失败")

    monkeypatch.setattr(tarfile.TarFile, "add", fail_read)
    with pytest.raises(OSError, match="读取源文件失败"):
        snapshot_sources([source], output, dry_run=False)
    assert not list(output.glob("*.tar.gz"))


def test_exclusive_create_never_removes_an_existing_backup(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    existing = tmp_path / "existing.tar.gz"
    existing.write_bytes(b"old backup")
    with pytest.raises(FileExistsError):
        _build_tar(source, existing, level=6, dry_run=False)
    assert existing.read_bytes() == b"old backup"
