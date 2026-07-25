import re
from pathlib import Path

from tools.core.output_dir import build_timestamped_output_dir


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
