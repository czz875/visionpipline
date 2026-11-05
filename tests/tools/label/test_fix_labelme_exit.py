"""修复失败必须让工作流收到非零退出码。"""

from pathlib import Path

import pytest
from PIL import Image

from tools.label.fix_labelme import main


@pytest.mark.parametrize("apply", [False, True])
def test_invalid_json_returns_failure_and_keeps_original(tmp_path: Path, monkeypatch, capsys, apply: bool) -> None:
    invalid = tmp_path / "bad.json"
    invalid.write_text("{ broken", encoding="utf-8")
    argv = ["fix_labelme", "--root", str(tmp_path)]
    if apply:
        argv.append("--apply")
    monkeypatch.setattr("sys.argv", argv)

    assert main() == 1
    assert "失败: 1" in capsys.readouterr().out
    assert invalid.read_text() == "{ broken"


def test_successful_repair_still_returns_success(tmp_path: Path, monkeypatch) -> None:
    import json

    Image.new("RGB", (20, 20)).save(tmp_path / "sample.png")
    (tmp_path / "sample.json").write_text('{"shapes":[]}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["fix_labelme", "--root", str(tmp_path), "--apply"])

    assert main() == 0
    data = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert data["imagePath"] == "sample.png"
    assert (data["imageWidth"], data["imageHeight"]) == (20, 20)
