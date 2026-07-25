"""
tools/annotate/auto.py

统一标注入口，只做命令行解析与模式分发：

- ``--detectors-config`` → ``tools.annotate.runners.multi``
- ``--model-type onnx`` → ``tools.annotate.runners.onnx``
- ``--model-type yolo|sam3|detr`` → ``tools.annotate.runners.supervision``
- ``--mosaic-existing`` → ``tools.annotate.runners.mosaic``

具体实现已拆分到 ``tools/annotate/runners/`` 各模块与 ``tools/annotate/parser.py``。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 允许以 `python tools/annotate/auto.py` 直接运行。
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# 默认参数（集中放文件顶部）
DEFAULT_OUTPUT_DIR = Path("datasets/01_annotated")


def ensure_local_supervision_import() -> None:
    """将仓库 ``src/`` 插入 ``sys.path`` 前部，优先使用本地开发版 supervision。

    在脚本入口处调用一次即可，避免安装版覆盖仓库内的 ``as_labelme``、
    ``as_coco`` 等较新接口。
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _resolve_output_dir(args: Any) -> Any:
    """解析输出目录。

    当 ``args.output`` 显式传入时直接使用该路径（创建父目录）；
    仅当 ``args.output`` 为 ``None`` 时，才在默认目录下追加
    ``<prefix>_YYYYMMDD_HHMMSS`` 时间戳子目录。
    """
    from tools.core import build_timestamped_output_dir

    explicit_output = getattr(args, "output", None)
    if explicit_output is not None:
        path = Path(explicit_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if getattr(args, "mosaic_existing", False):
        prefix = "mosaic"
    elif getattr(args, "detectors_config", None):
        prefix = "multi"
    elif getattr(args, "model_type", None):
        prefix = str(args.model_type)
    else:
        prefix = "annotate"
    return build_timestamped_output_dir(DEFAULT_OUTPUT_DIR, prefix)


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，解析参数并分发给对应 runner。"""
    from tools.annotate.parser import build_parser
    from tools.annotate.runners.mosaic import run_mosaic
    from tools.annotate.runners.multi import run_multi
    from tools.annotate.runners.onnx import run_onnx
    from tools.annotate.runners.supervision import run_supervision

    parser = build_parser()
    parser.set_defaults(output=None)
    args = parser.parse_args(argv)
    args.output = _resolve_output_dir(args)

    if args.mosaic_existing:
        ret = run_mosaic(args)
    elif args.detectors_config:
        ret = run_multi(args)
    elif args.model_type == "onnx":
        ret = run_onnx(args)
    else:
        ensure_local_supervision_import()
        ret = run_supervision(args)

    if args.output is not None:
        print(f"OUTPUT_PATH:{args.output.resolve()}")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
