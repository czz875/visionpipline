"""
tools/backup/snapshot.py
把指定目录打成一个带时间戳的 ``.tar.gz`` 备份。

行为：
- 走标准库 ``tarfile``，不引入新依赖；
- 默认 ``--dry-run``，加 ``--apply`` 才会真打；
- 多个源目录可以一次打完，文件名带统一时间戳；
- 输出文件名形如 ``<src-stem>_<YYYYMMDD_HHMMSS>.tar.gz``。

典型用法：

    .conda\\python.exe tools\\backup\\snapshot.py ^
        --sources datasets\\autolabel,datasets\\behavior ^
        --output-dir C:\\Users\\EDY\\Pictures

    .conda\\python.exe tools\\backup\\snapshot.py ^
        --sources datasets\\autolabel,datasets\\behavior ^
        --output-dir C:\\Users\\EDY\\Pictures ^
        --apply
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# 1. 默认参数
# =============================================================================

DEFAULT_SOURCES: tuple[str, ...] = (
    r"datasets\autolabel",
    r"datasets\behavior",
)
DEFAULT_OUTPUT_DIR = r"C:\Users\EDY\Pictures"
DEFAULT_LEVEL = 6                      # gzip 压缩等级 0-9
DEFAULT_DRY_RUN = True


# =============================================================================
# 2. 核心逻辑
# =============================================================================


def _make_tar_name(src: Path, now: datetime) -> str:
    """生成 ``<stem>_<YYYYMMDD_HHMMSS>.tar.gz`` 形式的备份名。"""
    return f"{src.name}_{now.strftime('%Y%m%d_%H%M%S')}.tar.gz"


def _build_tar(
    src: Path,
    dst_tar: Path,
    *,
    level: int,
    dry_run: bool,
    pbar: tqdm | None = None,
) -> Path:
    """把 ``src`` 打成 ``dst_tar``。``dry_run=True`` 时只返回目标路径不写盘。

    打包过程中通过回调更新 ``pbar``，让用户看到文件级进度。
    """
    if dry_run:
        return dst_tar
    dst_tar.parent.mkdir(parents=True, exist_ok=True)

    def _track(_tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
        if pbar is not None:
            pbar.update(1)
        return _tarinfo

    # 先统计文件总数，仅用于 progress 总量；不统计不会影响正确性
    total = sum(1 for _ in src.rglob("*") if _.is_file())
    if pbar is not None:
        pbar.total = (pbar.total or 0) + total
        pbar.refresh()

    with tarfile.open(dst_tar, "w:gz", compresslevel=level) as tar:
        tar.add(src, arcname=src.name, filter=_track)
    return dst_tar


def snapshot_sources(
    sources: list[Path],
    output_dir: Path,
    *,
    level: int = DEFAULT_LEVEL,
    dry_run: bool = DEFAULT_DRY_RUN,
) -> list[tuple[Path, Path]]:
    """批量打包。返回 ``[(src, dst_tar), ...]`` 计划列表。"""
    output_dir = output_dir.resolve()
    now = datetime.now()
    plans: list[tuple[Path, Path]] = []
    for src in sources:
        src = src.resolve()
        if not src.exists():
            print(f"[警告] 源目录不存在，跳过：{src}")
            continue
        if not src.is_dir():
            print(f"[警告] 源路径不是目录，跳过：{src}")
            continue
        dst = output_dir / _make_tar_name(src, now)
        plans.append((src, dst))

    # 总文件数先预估，用于 progress bar 的总长度
    total_files = 0
    for src, _dst in plans:
        if src.exists():
            total_files += sum(1 for _ in src.rglob("*") if _.is_file())

    with tqdm(
        total=total_files,
        desc="打包",
        unit="file",
        disable=dry_run,
    ) as pbar:
        for src, dst in plans:
            pbar.set_description(f"打包 {src.name}")
            _build_tar(src, dst, level=level, dry_run=dry_run, pbar=pbar)
    return plans


# =============================================================================
# 3. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把指定目录打成 .tar.gz 备份（默认 dry-run）。",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCES),
        help=(
            "要备份的源目录，逗号分隔。"
            f"默认：{','.join(DEFAULT_SOURCES)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"备份输出目录（默认：{DEFAULT_OUTPUT_DIR}）。",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=DEFAULT_LEVEL,
        help=f"gzip 压缩等级 0-9（默认 {DEFAULT_LEVEL}）。",
    )
    parser.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="真正执行打包（默认是 dry-run 预览）。",
    )
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    sources = [Path(s.strip()) for s in args.sources.split(",") if s.strip()]
    if not sources:
        print("[错误] --sources 不能为空。")
        return 1

    plans = snapshot_sources(
        sources,
        args.output_dir,
        level=args.level,
        dry_run=args.dry_run,
    )

    mode = "预览" if args.dry_run else "已生成"
    print(f"[{mode}] 共 {len(plans)} 个备份")
    for src, dst in plans:
        print(f"  {src}  ->  {dst}")
    if args.dry_run:
        print("\n（这是 dry-run 预览，加上 --apply 才会真正打包）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
