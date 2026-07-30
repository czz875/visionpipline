"""标注、合并与打码阶段的批次路径测试。"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tools.annotate import auto, merge
from tools.annotate.defaults import DEFAULT_OUTPUT_DIR
from tools.annotate.parser import build_parser
from tools.annotate.runners import multi as multi_runner
from tools.annotate.runners import onnx as onnx_runner


def _make_args(tmp_path: Path, **overrides: object) -> Namespace:
    """构造路径解析所需的最小参数。"""
    values = {
        "output": tmp_path / "01_annotated",
        "source": tmp_path / "source",
        "mosaic_existing": False,
        "detectors_config": None,
        "model_type": "onnx",
        "dry_run": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _make_onnx_args(tmp_path: Path, **overrides: object) -> Namespace:
    """构造 ONNX runner 所需参数。"""
    values = {
        "source": tmp_path / "input",
        "output": Path("datasets/01_annotated"),
        "reannotate": False,
        "onnx_model": tmp_path / "fake.onnx",
        "onnx_label": "face",
        "onnx_conf": 0.25,
        "onnx_normalize": True,
        "onnx_transpose": False,
        "onnx_score_indices": "4",
        "onnx_min_ratio": 0.01,
        "sam_model": None,
        "sam_prompt": "hand",
        "sam_label": "hand",
        "sam_conf": 0.25,
        "sam_min_ratio": 0.01,
        "device": "",
        "mosaic": False,
        "blackout": False,
        "no_blackout": False,
        "mosaic_block": 16,
        "dry_run": False,
        "recursive": True,
        "input_root": tmp_path / "behavior",
        "start_batch": 1,
        "end_batch": 1,
        "keep_labels": None,
        "keep_min_ratio": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _make_multi_args(tmp_path: Path, cfg_path: Path, **overrides: object) -> Namespace:
    """构造多检测器 runner 所需参数。"""
    values = {
        "detectors_config": cfg_path,
        "source": tmp_path / "fallback_input",
        "output": None,
        "recursive": True,
        "mosaic_block": 16,
        "onnx_min_ratio": 0.01,
        "device": "",
        "dry_run": True,
    }
    values.update(overrides)
    return Namespace(**values)


def _patch_onnx_runtime(monkeypatch, image_path: Path) -> None:
    """替换模型与图片迭代，专注验证输出目录准备。"""
    monkeypatch.setattr(onnx_runner, "list_images", lambda *args, **kwargs: [image_path])
    monkeypatch.setattr(onnx_runner, "OnnxDetector", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        onnx_runner,
        "iter_annotated_images",
        lambda *args, **kwargs: iter(()),
    )


def _patch_multi_runtime(monkeypatch, image_path: Path) -> None:
    """替换检测器与图片迭代，避免测试加载真实模型。"""
    monkeypatch.setattr(multi_runner, "list_images", lambda *args, **kwargs: [image_path])
    monkeypatch.setattr(
        multi_runner,
        "_build_detectors",
        lambda *args, **kwargs: [Namespace(name="fake")],
    )
    monkeypatch.setattr(
        multi_runner,
        "iter_annotated_images",
        lambda *args, **kwargs: iter(()),
    )


def test_onnx_auto_defers_standard_output_to_runner(tmp_path: Path, monkeypatch) -> None:
    """ONNX 标注应保留逻辑输出路径，由 runner 延迟创建批次。"""
    logical_dir = tmp_path / "01_annotated"
    args = _make_args(tmp_path, output=logical_dir, dry_run=False)

    def fail_if_called(path: Path) -> Path:
        raise AssertionError(f"auto 不应创建 ONNX 批次：{path}")

    auto._resolve_batch_paths(args)

    assert args.output == logical_dir


def test_mosaic_reuses_latest_annotated_batch(tmp_path: Path, monkeypatch) -> None:
    """已有标注打码应复用最新批次，并统一输入输出目录。"""
    logical_dir = tmp_path / "01_annotated"
    expected = tmp_path / "batch_20260729_120000" / "01_annotated"
    args = _make_args(
        tmp_path,
        output=None,
        source=logical_dir,
        mosaic_existing=True,
    )
    monkeypatch.setattr(
        auto,
        "resolve_latest_batch_stage_dir",
        lambda path: expected,
    )

    auto._resolve_batch_paths(args)

    assert args.source == expected
    assert args.output == expected


def test_merge_reuses_latest_annotated_batch(tmp_path: Path, monkeypatch) -> None:
    """合并阶段应把逻辑目录解析为最新批次中的标注目录。"""
    logical_dir = tmp_path / "01_annotated"
    expected = tmp_path / "batch_20260729_120000" / "01_annotated"
    monkeypatch.setattr(
        merge,
        "resolve_latest_batch_stage_dir",
        lambda path: expected,
    )

    assert merge._resolve_json_dir(logical_dir) == expected


def test_custom_output_keeps_existing_semantics(tmp_path: Path, monkeypatch) -> None:
    """自定义输出名称不应创建总批次目录。"""
    custom_output = tmp_path / "custom_output"
    args = _make_args(tmp_path, output=custom_output)

    auto._resolve_batch_paths(args)

    assert args.output == custom_output


def test_onnx_dry_run_without_output_creates_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """解析器默认输出标准逻辑目录，预览不应创建任何目录。"""
    monkeypatch.chdir(tmp_path)
    parsed = build_parser().parse_args([])
    args = _make_args(tmp_path, output=parsed.output, dry_run=True)

    auto._resolve_batch_paths(args)

    assert args.output == DEFAULT_OUTPUT_DIR
    assert list(tmp_path.iterdir()) == []


def test_multi_auto_leaves_yaml_paths_for_runner(tmp_path: Path, monkeypatch) -> None:
    """多检测器模式不得在 auto 中创建或改写 YAML 路径。"""
    output = tmp_path / "yaml_output"
    args = _make_args(tmp_path, output=output, detectors_config="detectors.yaml")

    auto._resolve_batch_paths(args)

    assert args.output == output


def test_onnx_invalid_source_checks_before_model_and_creates_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """无效输入必须在加载模型和创建批次前返回。"""
    monkeypatch.chdir(tmp_path)
    args = _make_onnx_args(tmp_path, source=tmp_path / "missing")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("无效输入不应初始化 ONNX 检测器")

    monkeypatch.setattr(onnx_runner, "OnnxDetector", fail_if_called)

    assert onnx_runner.run_onnx(args) == 1
    assert list(tmp_path.glob("batch_*")) == []


def test_onnx_empty_source_checks_before_model_and_creates_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """空输入目录必须在加载模型和创建批次前返回。"""
    source = tmp_path / "input"
    source.mkdir()
    args = _make_onnx_args(tmp_path, source=source)

    monkeypatch.setattr(onnx_runner, "list_images", lambda *args, **kwargs: [])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("空输入不应初始化 ONNX 检测器")

    monkeypatch.setattr(onnx_runner, "OnnxDetector", fail_if_called)

    assert onnx_runner.run_onnx(args) == 1
    assert list(tmp_path.glob("batch_*")) == []


def test_onnx_apply_creates_standard_batch_and_updates_args(
    tmp_path: Path, monkeypatch
) -> None:
    """ONNX 实际写盘应在准备成功后创建标准批次并回写参数。"""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input"
    source.mkdir()
    args = _make_onnx_args(tmp_path, source=source)
    _patch_onnx_runtime(monkeypatch, source / "a.png")

    assert onnx_runner.run_onnx(args) == 0

    assert args.output.is_dir()
    assert args.output.name == "01_annotated"
    assert args.output.parent.name.startswith("batch_")


def test_onnx_custom_same_basename_is_not_batchified(
    tmp_path: Path, monkeypatch
) -> None:
    """非项目标准路径即使同名，也只创建自定义输出目录。"""
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "custom_job" / "01_annotated"
    args = _make_onnx_args(tmp_path, source=source, output=output)
    _patch_onnx_runtime(monkeypatch, source / "a.png")

    assert onnx_runner.run_onnx(args) == 0

    assert args.output == output
    assert output.is_dir()
    assert list(output.parent.glob("batch_*")) == []


def _write_multi_config(cfg_path: Path, source: Path, output: Path | str) -> None:
    """写入最小多检测器测试配置。"""
    cfg_path.write_text(
        f'source: "{source.as_posix()}"\n'
        f'output: "{Path(output).as_posix()}"\n'
        "detectors:\n"
        "  - type: onnx\n"
        "    model: fake.onnx\n",
        encoding="utf-8",
    )


def test_multi_yaml_standard_output_dry_run_creates_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """多检测器 YAML 标准输出在预览时不得创建目录。"""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input"
    source.mkdir()
    cfg_path = tmp_path / "detectors.yaml"
    _write_multi_config(cfg_path, source, Path("datasets/01_annotated"))
    args = _make_multi_args(tmp_path, cfg_path, dry_run=True)
    _patch_multi_runtime(monkeypatch, source / "a.png")

    assert multi_runner.run_multi(args) == 0

    assert args.source == source
    assert args.output == Path("datasets/01_annotated")
    assert not (tmp_path / "datasets").exists()


def test_multi_yaml_standard_output_apply_creates_batch(
    tmp_path: Path, monkeypatch
) -> None:
    """多检测器 YAML 标准输出实际写盘时应创建批次并回写参数。"""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input"
    source.mkdir()
    cfg_path = tmp_path / "detectors.yaml"
    _write_multi_config(cfg_path, source, Path("datasets/01_annotated"))
    args = _make_multi_args(tmp_path, cfg_path, dry_run=False)
    _patch_multi_runtime(monkeypatch, source / "a.png")

    assert multi_runner.run_multi(args) == 0

    assert args.source == source
    assert args.output.is_dir()
    assert args.output.name == "01_annotated"
    assert args.output.parent.name.startswith("batch_")


def test_multi_yaml_custom_same_basename_is_not_batchified(
    tmp_path: Path, monkeypatch
) -> None:
    """多检测器 YAML 自定义同名输出不得创建批次。"""
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "custom_job" / "01_annotated"
    cfg_path = tmp_path / "detectors.yaml"
    _write_multi_config(cfg_path, source, output)
    args = _make_multi_args(tmp_path, cfg_path, dry_run=False)
    _patch_multi_runtime(monkeypatch, source / "a.png")

    assert multi_runner.run_multi(args) == 0

    assert args.output == output
    assert output.is_dir()
    assert list(output.parent.glob("batch_*")) == []
