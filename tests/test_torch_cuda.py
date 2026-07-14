from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 允许以 `python tests/test_torch_cuda.py` 直接运行 / pytest 发现。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402


def test_torch_importable() -> None:
    """torch 可被正常导入。"""
    assert torch is not None


def test_torch_version_present() -> None:
    """torch 版本号存在且非空。"""
    assert getattr(torch, "__version__", "") != ""


def test_cuda_is_available() -> None:
    """CUDA 对 torch 可用。

    若运行环境无 GPU / 未安装 CUDA 版 torch，跳过而非失败。
    """
    if not torch.cuda.is_available():
        pytest.skip("当前环境未检测到可用 CUDA，跳过 GPU 相关断言")
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() >= 1


def test_cuda_tensor_computation() -> None:
    """可在 CUDA 上完成一次基本张量运算。"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境未检测到可用 CUDA，跳过 GPU 运算测试")
    device = torch.device("cuda")
    a = torch.randn(3, 3, device=device)
    b = torch.randn(3, 3, device=device)
    c = a @ b
    assert c.device.type == "cuda"
    assert c.shape == (3, 3)
