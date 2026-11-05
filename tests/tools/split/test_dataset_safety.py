"""拆分工具必须保留旧数据并阻止 A/B 样本混合。"""

from pathlib import Path

import pytest

from tools.split.dataset import main, split_dataset


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for i in range(10):
        (source / f"{i}.png").write_bytes(f"image-{i}".encode())
        (source / f"{i}.json").write_text("{}", encoding="utf-8")
    return source


def test_rerun_rejects_nonempty_outputs_before_copying(tmp_path: Path) -> None:
    source = _source(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    assert split_dataset(source, a, b, 0.3, 42) == (3, 7)
    before = {p: p.read_bytes() for d in (a, b) for p in d.rglob("*") if p.is_file()}

    with pytest.raises(ValueError, match="非空"):
        split_dataset(source, a, b, 0.3, 43)

    after = {p: p.read_bytes() for d in (a, b) for p in d.rglob("*") if p.is_file()}
    assert after == before
    assert not {p.name for p in a.glob("*.png")} & {p.name for p in b.glob("*.png")}


@pytest.mark.parametrize("layout", ["same", "nested", "inside_source", "contains_source"])
def test_overlapping_paths_are_rejected_without_writes(tmp_path: Path, layout: str) -> None:
    source = _source(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    if layout == "same":
        b = a
    elif layout == "nested":
        b = a / "nested"
    elif layout == "inside_source":
        a = source / "nested"
    else:
        a = tmp_path

    with pytest.raises(ValueError, match="重叠"):
        split_dataset(source, a, b, 0.3, 42)

    assert len(list(source.glob("*.png"))) == 10
    assert not b.exists()


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_empty_input_returns_failure(tmp_path: Path, monkeypatch, exists: bool) -> None:
    source = tmp_path / "source"
    if exists:
        source.mkdir()
    monkeypatch.setattr("sys.argv", [
        "split", "--input", str(source),
        "--output-a", str(tmp_path / "a"), "--output-b", str(tmp_path / "b"),
    ])

    assert main() == 1
    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


@pytest.mark.parametrize("exists", [False, True])
def test_invalid_input_does_not_create_default_outputs(tmp_path: Path, monkeypatch, exists: bool) -> None:
    source = tmp_path / "source"
    if exists:
        source.mkdir()
    base = tmp_path / "default_outputs"
    monkeypatch.setattr("tools.split.dataset.DEFAULT_OUTPUT_DIR", base)
    monkeypatch.setattr("sys.argv", ["split", "--input", str(source)])

    assert main() == 1
    assert not base.exists()


def test_invalid_explicit_output_does_not_create_the_default_other_half(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    base = tmp_path / "default_outputs"
    monkeypatch.setattr("tools.split.dataset.DEFAULT_OUTPUT_DIR", base)
    monkeypatch.setattr("sys.argv", ["split", "--input", str(source), "--output-b", str(existing)])

    assert main() == 1
    assert not base.exists()
    assert (existing / "keep.txt").read_text() == "keep"


def test_blocked_output_parent_is_checked_before_creating_other_half(tmp_path: Path) -> None:
    source = _source(tmp_path)
    blocker = tmp_path / "file"
    blocker.write_bytes(b"keep")
    with pytest.raises(ValueError, match="目录"):
        split_dataset(source, tmp_path / "a", blocker / "b", 0.3, 42)
    assert not (tmp_path / "a").exists()
    assert blocker.read_bytes() == b"keep"


def test_default_outputs_inside_input_are_rejected_before_creation(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    base = source / "split"
    monkeypatch.setattr("tools.split.dataset.DEFAULT_OUTPUT_DIR", base)
    monkeypatch.setattr("sys.argv", ["split", "--input", str(source)])
    assert main() == 1
    assert not base.exists()
