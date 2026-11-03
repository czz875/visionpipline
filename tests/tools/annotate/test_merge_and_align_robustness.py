"""标注合并与标签对齐的回归测试。

覆盖两类「静默出错」的缺陷：
- ``tools/annotate/merge.py`` 在目录不存在 / 无 JSON 时返回 0，定时任务误判成功；
  且任一 shape 缺 ``label`` 会让 ``KeyError`` 终止整个线程池；
- ``tools/label/align_labelme.py`` 在同名 PNG 有多张时任意挑一个，把标注绑到
  错误的图片上。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.annotate.merge import process_json_dir_mt, process_labelme_json
from tools.label.align_labelme import plan_alignments


def _merge_args(**overrides) -> argparse.Namespace:
    base = dict(
        face_label="face",
        # right_cut_ratio=0 → 不裁剪任何右侧区域，避免测试被裁剪规则干扰
        right_cut_ratio=0.0,
        min_face_width=0,
        min_face_height=0,
        merge_distance_x=100,
        merge_distance_y=200,
        max_merge_width=600,
        max_merge_height=600,
        max_merge_count=0,
        max_workers=1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_labelme(path: Path, shapes: list[dict], width: int = 100, height: int = 100) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "5.3.1",
                "imagePath": path.with_suffix(".png").name,
                "imageWidth": width,
                "imageHeight": height,
                "shapes": shapes,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _rect(label: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    return {
        "label": label,
        "shape_type": "rectangle",
        "points": [[x1, y1], [x2, y2]],
    }


def test_merge_missing_dir_returns_negative(tmp_path: Path) -> None:
    """目录不存在时返回 -1，由 main 转成非 0 退出码。"""
    assert process_json_dir_mt(tmp_path / "nope", _merge_args()) == -1


def test_merge_empty_dir_returns_negative(tmp_path: Path) -> None:
    """目录存在但没有 JSON 时同样不能报成功。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert process_json_dir_mt(empty, _merge_args()) == -1


def test_merge_skips_shape_without_label(tmp_path: Path) -> None:
    """缺 label 的坏 shape 要被跳过，不能抛 KeyError 中断整批。"""
    json_path = tmp_path / "x.json"
    _write_labelme(
        json_path,
        [
            {"shape_type": "rectangle", "points": [[0, 0], [10, 10]]},  # 无 label
            _rect("face", 0, 0, 20, 20),
        ],
    )

    error = process_labelme_json(json_path, _merge_args())

    assert error is None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # 合法的 face 框仍然保留
    assert [s["label"] for s in data["shapes"]] == ["face"]


def test_plan_alignments_does_not_guess_between_duplicate_pngs(tmp_path: Path) -> None:
    """同名 PNG 有多张且都不在 JSON 同目录时，不做猜测，计入 ambiguous。"""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    (tmp_path / "a" / "same.png").write_bytes(b"A")
    (tmp_path / "b" / "same.png").write_bytes(b"B")
    # JSON 放在第三个目录，指向的 same.png 既不在同目录、也无法唯一定位
    json_path = tmp_path / "c" / "other.json"
    json_path.write_text(
        json.dumps({"imagePath": "same.png", "shapes": []}), encoding="utf-8"
    )

    pngs = [tmp_path / "a" / "same.png", tmp_path / "b" / "same.png"]
    plans, ambiguous = plan_alignments(pngs, [json_path])

    assert plans == []
    assert ambiguous == [json_path]


def test_plan_alignments_prefers_same_dir_when_duplicated(tmp_path: Path) -> None:
    """同名 PNG 多张时，同目录那张是确定匹配，应正常对齐。"""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "same.png").write_bytes(b"A")
    (tmp_path / "b" / "same.png").write_bytes(b"B")
    json_path = tmp_path / "b" / "old_name.json"
    json_path.write_text(
        json.dumps({"imagePath": "same.png", "shapes": []}), encoding="utf-8"
    )

    pngs = [tmp_path / "a" / "same.png", tmp_path / "b" / "same.png"]
    plans, ambiguous = plan_alignments(pngs, [json_path])

    assert ambiguous == []
    assert len(plans) == 1
    json_old, json_new, png, new_name = plans[0]
    assert json_old == json_path
    assert png == tmp_path / "b" / "same.png"
    assert json_new == tmp_path / "b" / "same.json"
    assert new_name == "same.png"
