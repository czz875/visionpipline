"""公共写入应隔离硬链接，且在发布失败时保留原文件。"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.core import read_image, save_labelme, write_image


@pytest.mark.parametrize("kind", ["json", "image"])
def test_write_replaces_only_the_requested_hardlink(tmp_path: Path, kind: str) -> None:
    target = tmp_path / ("sample.json" if kind == "json" else "sample.png")
    if kind == "json":
        target.write_text('{"imagePath":"old.png","shapes":[]}', encoding="utf-8")
    else:
        write_image(target, np.zeros((8, 8, 3), dtype=np.uint8))
    backup = tmp_path / ("backup.json" if kind == "json" else "backup.png")
    os.link(target, backup)
    original = backup.read_bytes()

    if kind == "json":
        save_labelme({"imagePath": "新图片.png", "shapes": []}, target)
        assert json.loads(target.read_text(encoding="utf-8"))["imagePath"] == "新图片.png"
    else:
        write_image(target, np.full((8, 8, 3), 255, dtype=np.uint8))
        assert np.all(read_image(target) == 255)

    assert backup.read_bytes() == original
    assert not target.samefile(backup)


@pytest.mark.parametrize("kind", ["json", "image"])
def test_failed_publish_preserves_old_file_and_removes_temporary_file(tmp_path: Path, monkeypatch, kind: str) -> None:
    target = tmp_path / ("sample.json" if kind == "json" else "sample.png")
    target.write_bytes(b"original content")

    def fail_publish(*args, **kwargs):
        raise OSError("模拟发布失败")

    monkeypatch.setattr(os, "replace", fail_publish)
    with pytest.raises(OSError, match="模拟发布失败"):
        if kind == "json":
            save_labelme({"shapes": []}, target)
        else:
            write_image(target, np.zeros((8, 8, 3), dtype=np.uint8))

    assert target.read_bytes() == b"original content"
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("kind", ["json", "image"])
def test_partial_temporary_write_does_not_damage_the_old_target(tmp_path: Path, monkeypatch, kind: str) -> None:
    target = tmp_path / ("sample.json" if kind == "json" else "sample.png")
    target.write_bytes(b"original content")
    real_fdopen = os.fdopen

    class FailingStream:
        def __init__(self, descriptor, mode):
            self.stream = real_fdopen(descriptor, mode)

        def __enter__(self):
            return self

        def write(self, data):
            self.stream.write(memoryview(data).cast("B")[:5])
            raise OSError("模拟临时写盘中断")

        def __exit__(self, *args):
            self.stream.close()

    monkeypatch.setattr(os, "fdopen", FailingStream)
    with pytest.raises(OSError, match="模拟临时写盘中断"):
        if kind == "json":
            save_labelme({"shapes": []}, target)
        else:
            write_image(target, np.zeros((8, 8, 3), dtype=np.uint8))
    assert target.read_bytes() == b"original content"
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("kind", ["json", "image"])
def test_writing_through_a_symlink_keeps_the_link_and_updates_its_target(tmp_path: Path, kind: str) -> None:
    target = tmp_path / ("actual.json" if kind == "json" else "actual.png")
    target.write_bytes(b"old")
    alias = tmp_path / ("alias.json" if kind == "json" else "alias.png")
    os.symlink(target, alias)
    if kind == "json":
        save_labelme({"imagePath": "new.png"}, alias)
        assert json.loads(target.read_text(encoding="utf-8"))["imagePath"] == "new.png"
    else:
        write_image(alias, np.full((8, 8, 3), 255, dtype=np.uint8))
        assert np.all(read_image(target) == 255)
    assert alias.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX 文件权限与 umask 语义")
def test_atomic_replacement_keeps_existing_file_permissions(tmp_path: Path) -> None:
    import stat

    target = tmp_path / "sample.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o664)
    save_labelme({"shapes": []}, target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o664


@pytest.mark.skipif(os.name == "nt", reason="POSIX 文件权限与 umask 语义")
def test_new_atomic_file_uses_normal_create_permissions(tmp_path: Path) -> None:
    import stat

    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text("{}", encoding="utf-8")
    target = tmp_path / "sample.json"
    save_labelme({"shapes": []}, target)
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(ordinary.stat().st_mode)
