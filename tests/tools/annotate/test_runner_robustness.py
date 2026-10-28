"""标注循环的健壮性回归测试。

覆盖：
- 单张图写盘失败（磁盘满、路径过长、文件被占用）不能中断整批；
- 坏 JSON（语法错误、被截断）不能中断整批，且已就地改写的图片要如实告警。
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.annotate.runners.common import iter_annotated_images


def _make_image(path: Path) -> Path:
    Image.new("RGB", (100, 80), color="white").save(path)
    return path


def test_write_failure_does_not_abort_batch(tmp_path: Path, monkeypatch) -> None:
    """单张写盘失败应跳过并计数，其余图片照常处理。"""
    src = tmp_path / "src"
    src.mkdir()
    images = [_make_image(src / f"img{i}.png") for i in range(3)]
    out = tmp_path / "out"

    import tools.annotate.runners.common as common

    real_write = common.write_image

    def flaky_write(path, image, params=None):
        if Path(path).name == "img1.png":
            raise OSError("磁盘已满")
        return real_write(path, image, params)

    monkeypatch.setattr(common, "write_image", flaky_write)
    failures: list[Path] = []

    def build_sources(image, image_path):
        from tools.annotate.runners.common import BoxSource

        return [
            BoxSource(
                kept=np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
                removed=np.empty((0, 4), dtype=np.float32),
                kept_labels=["face"],
            )
        ]

    results = list(
        iter_annotated_images(
            images,
            build_sources,
            use_mosaic=False,
            mosaic_block=16,
            dry_run=False,
            output=out,
            failures=failures,
        )
    )

    # 失败图被跳过（不 yield），其余两张正常产出，循环不中断
    assert len(results) == 2, f"期望 2 张成功，实际 {len(results)}"
    assert len(failures) == 1 and failures[0].name == "img1.png", (
        f"失败清单不对：{failures}"
    )
    # 好图应正常产出
    assert (out / "img0.json").exists()
    assert (out / "img2.json").exists()
    # 失败的那张不该留下 JSON（避免「有 JSON 无图」）
    assert not (out / "img1.json").exists(), "写图失败却写了 JSON"


def test_broken_json_does_not_abort_mosaic(tmp_path: Path, monkeypatch) -> None:
    """坏 JSON 应跳过该图并告警，不能中断整批。"""
    from tools.annotate.runners import mosaic as mosaic_mod

    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        _make_image(src / f"img{i}.png")
        data = {
            "shapes": [
                {"label": "face", "shape_type": "rectangle", "points": [[10, 10], [50, 50]]}
            ],
            "imagePath": f"img{i}.png",
            "imageWidth": 100,
            "imageHeight": 80,
        }
        (src / f"img{i}.json").write_text(json.dumps(data), encoding="utf-8")
    # 把中间那个 JSON 写成坏语法
    (src / "img1.json").write_text("{ bad", encoding="utf-8")

    class _Args:
        source = str(src)
        recursive = False
        blackout = False
        mosaic_min_ratio = 0.5
        onnx_min_ratio = 0.01
        mosaic_labels = None
        mosaic_block = 16
        dry_run = False

    rc = mosaic_mod.run_mosaic(_Args())

    # 坏 JSON 被跳过，其余两张正常处理，返回码反映有失败
    assert (src / "img0.json").exists()
    assert (src / "img2.json").exists()
    assert rc == 1, "有失败图时应返回非 0"
    # 坏 JSON 保持原样，不能被覆盖成合法 JSON
    assert (src / "img1.json").read_text(encoding="utf-8") == "{ bad"
