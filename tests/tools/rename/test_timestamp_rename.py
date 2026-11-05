"""批量改名必须在规划阶段预留完整的图片与 JSON 目标。"""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from tools.rename.timestamp_rename import rename_by_timestamp


@pytest.mark.parametrize("sync", [False, True])
def test_planned_targets_are_unique_and_execution_preserves_pairs(tmp_path: Path, sync: bool) -> None:
    timestamp = 1700000000.0
    prefix = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    images = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / f"{prefix}_000001.png"]
    before = {}
    for i, image in enumerate(images):
        before[image] = f"image-{i}".encode()
        image.write_bytes(before[image])
        os.utime(image, (timestamp + i * 0.1, timestamp + i * 0.1))
        if sync:
            image.with_suffix(".json").write_text(
                json.dumps({"imagePath": image.name, "origin": image.name, "shapes": []}), encoding="utf-8",
            )

    preview, errors = rename_by_timestamp(tmp_path, dry_run=True, labelme_sync=sync)
    assert errors == []
    assert len({new for old, new in preview}) == 3
    assert {p: p.read_bytes() for p in images} == before

    plans, errors = rename_by_timestamp(tmp_path, dry_run=False, labelme_sync=sync)
    assert errors == []
    assert plans == preview
    assert len(list(tmp_path.glob("*.png"))) == 3
    for old, new in plans:
        assert new.read_bytes() == before[old]
        if sync:
            data = json.loads(new.with_suffix(".json").read_text(encoding="utf-8"))
            assert data["imagePath"] == new.name
            assert data["origin"] == old.name


def test_json_target_collision_changes_the_image_target_together(tmp_path: Path) -> None:
    timestamp = 1700000000.0
    prefix = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    source = tmp_path / "a.png"
    source.write_bytes(b"image")
    os.utime(source, (timestamp, timestamp))
    source.with_suffix(".json").write_text('{"imagePath":"a.png","shapes":[]}', encoding="utf-8")
    existing = tmp_path / f"{prefix}_000001.json"
    existing.write_text("existing orphan annotation", encoding="utf-8")

    plans, errors = rename_by_timestamp(tmp_path, dry_run=False, labelme_sync=True)

    assert errors == []
    assert len(plans) == 1
    target = plans[0][1]
    assert target.with_suffix(".json") != existing
    assert existing.read_text() == "existing orphan annotation"
    data = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["imagePath"] == target.name
    assert not list(tmp_path.glob("*__dup*.json"))
