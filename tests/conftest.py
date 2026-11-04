"""pytest 全局配置与共享 fixture。

提供三样东西：

1. **marker 体系**：把「依赖真实数据集 / 依赖 GPU / 依赖模型权重 / 慢」这几类
   外部依赖显式标记出来。没有真实 ``datasets/`` 的环境（如 CI、别人的机器）
   可以用 ``-m "not requires_dataset"`` 精确裁剪；本机确认守护测试确实跑到了
   时可以用 ``-m requires_dataset`` 单独跑一遍——避免「静默 skip」让人误以为
   守护测试已经覆盖。

2. **中文路径 fixture**：本项目大量处理 Windows 中文路径，统一提供
   ``chinese_dir`` fixture，避免每个测试各写一遍。

3. **仓库根路径常量**：测试里定位 ``tools/cfg/``、``datasets/`` 等目录时用同一个来源。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = REPO_ROOT / "datasets"
YOLO_ROOT = DATASETS_ROOT / "yolo"
CFG_DIR = REPO_ROOT / "tools" / "cfg"


def pytest_configure(config: pytest.Config) -> None:
    """注册自定义 marker，避免 pytest 因未知 marker 报警告。"""
    config.addinivalue_line(
        "markers", "requires_dataset: 需要仓库内真实 datasets/ 才能运行"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: 需要 CUDA GPU 才能运行"
    )
    config.addinivalue_line(
        "markers", "requires_weights: 需要本地模型权重才能运行"
    )
    config.addinivalue_line("markers", "slow: 耗时较长的用例")


@pytest.fixture
def repo_root() -> Path:
    """仓库根目录。"""
    return REPO_ROOT


@pytest.fixture
def chinese_dir(tmp_path: Path) -> Path:
    """带中文与空格的目录，用于验证 Windows 中文路径下的读写。"""
    d = tmp_path / "中文目录 带空格"
    d.mkdir()
    return d


@pytest.fixture
def skip_without_dataset() -> None:
    """没有真实 datasets/ 时跳过，并说明原因。"""
    if not DATASETS_ROOT.is_dir():
        pytest.skip("需要仓库内真实 datasets/ 才能运行")
