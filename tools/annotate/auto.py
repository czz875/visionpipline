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

from tools.core import resolve_latest_batch_stage_dir


def ensure_local_supervision_import() -> None:
    """将仓库 ``src/`` 插入 ``sys.path`` 前部，优先使用本地开发版 supervision。

    在脚本入口处调用一次即可，避免安装版覆盖仓库内的 ``as_labelme``、
    ``as_coco`` 等较新接口。
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _resolve_batch_paths(args: Any) -> None:
    """解析当前模式使用的输入与输出批次目录。"""
    if getattr(args, "mosaic_existing", False):
        path = resolve_latest_batch_stage_dir(Path(args.source))
        args.source = path
        args.output = path
        return

    if getattr(args, "detectors_config", None):
        return

    if getattr(args, "reannotate", False):
        return

    path = Path(args.output)
    if not (getattr(args, "dry_run", False) and getattr(args, "model_type", None) == "onnx"):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output = path


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，解析参数并分发给对应 runner。"""
    from tools.annotate.parser import build_parser
    from tools.annotate.runners.mosaic import run_mosaic
    from tools.annotate.runners.multi import run_multi
    from tools.annotate.runners.onnx import run_onnx
    from tools.annotate.runners.supervision import run_supervision

    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_batch_paths(args)

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
