from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 允许以 `python tests/test_reannotate_face_hand_onnx.py` 直接运行 / pytest 发现。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 本测试专门验证 onnxruntime 能否在 CPU / GPU(CUDA) 上加载并真正执行推理；
# 环境缺少 onnxruntime 时整体跳过，避免误报。
ort = pytest.importorskip("onnxruntime")

# 复用项目内「人脸/手势重标注」ONNX 的 provider 选择逻辑，对齐真实运行环境。
from tools.annotate.reannotate_face_hand_onnx import (  # noqa: E402
    get_available_execution_providers,
    resolve_execution_providers,
)

# =============================================================================
# 默认参数
# =============================================================================

# onnxruntime 包内自带的极小示例模型（矩阵乘法，无需外部权重文件）。
DEFAULT_SAMPLE_MODEL = "mul_1.onnx"


# =============================================================================
# 辅助函数
# =============================================================================


def _sample_model_path() -> Path:
    """取 onnxruntime 包内自带的示例模型路径，缺失时跳过。"""
    path = Path(ort.__file__).resolve().parent / "datasets" / DEFAULT_SAMPLE_MODEL
    if not path.exists():
        pytest.skip(f"未找到 onnxruntime 自带示例模型：{path}")
    return path


def _run_sample(session: ort.InferenceSession) -> np.ndarray:
    """用全 1 输入跑一次推理，返回第一个输出。"""
    feed = {}
    for inp in session.get_inputs():
        shape = [
            1 if (d is None or not str(d).isdigit()) else int(d) for d in inp.shape
        ]
        feed[inp.name] = np.ones(shape, dtype=np.float32)
    return session.run(None, feed)[0]


# =============================================================================
# 测试
# =============================================================================


def test_onnxruntime_importable() -> None:
    """onnxruntime 可被正常导入。"""
    assert ort is not None


def test_cpu_execution_provider_available() -> None:
    """CPU 提供方始终可用。"""
    assert "CPUExecutionProvider" in ort.get_available_providers()


def test_onnx_runs_on_cpu() -> None:
    """ONNX 模型能在 CPUExecutionProvider 上完成一次真实推理。"""
    session = ort.InferenceSession(
        str(_sample_model_path()), providers=["CPUExecutionProvider"]
    )
    assert "CPUExecutionProvider" in session.get_providers()
    out = _run_sample(session)
    assert out is not None
    assert tuple(out.shape) == (3, 2)


def test_onnx_runs_on_gpu_when_available() -> None:
    """有 CUDA 时 ONNX 能在 GPU 上推理；无 GPU 则跳过。"""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("当前环境未检测到 CUDAExecutionProvider，跳过 GPU 推理测试")
    session = ort.InferenceSession(
        str(_sample_model_path()),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert "CUDAExecutionProvider" in session.get_providers()
    out = _run_sample(session)
    assert out is not None
    assert tuple(out.shape) == (3, 2)


def test_resolve_execution_providers_prefers_cuda() -> None:
    """项目的 provider 选择逻辑：有 CUDA 优先，否则回退 CPU。

    用真实运行环境的可用 provider 列表验证，而非 mock。
    """
    available = ort.get_available_providers()
    resolved = resolve_execution_providers(available)
    if "CUDAExecutionProvider" in available:
        assert resolved[0] == "CUDAExecutionProvider"
    else:
        assert resolved == ["CPUExecutionProvider"]


def test_get_available_execution_providers_real() -> None:
    """项目的 provider 获取封装能在真实 onnxruntime 上工作。"""
    providers = get_available_execution_providers(ort)
    assert "CPUExecutionProvider" in providers
