"""共享文件写入：原子替换旧文件，或排他移动到新路径。"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from uuid import uuid4


def atomic_write_bytes(path: Path, data: bytes | memoryview) -> None:
    """在同目录完成写入后原子替换目标，失败时保留旧文件并清理本次临时文件。

    调用方负责创建父目录。替换独立目录项，不修改其他硬链接的内容。
    """
    # 保留原 API 追随符号链接的行为，同时以替换目标目录项隔离其他硬链接。
    path = Path(path).resolve()
    try:
        previous_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        previous_mode = None
    if previous_mode is not None and not os.access(path, os.W_OK):
        raise PermissionError(f"目标文件不可写：{path}")
    temporary = path.parent / f".write_{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        if previous_mode is not None and os.name != "nt":
            temporary.chmod(previous_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_owned_file(path: Path, identity: tuple[int, int]) -> bool:
    """清理前复核文件身份；调用方须避免其他写入者同时覆盖该路径。"""
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != identity:
        return False
    path.unlink()
    return True


def move_file_without_overwrite(source: Path, target: Path) -> tuple[int, int]:
    """独占创建目标再移除源；跨卷复制失败时清理本次创建的目标。

    同卷优先使用硬链接保留文件内容及元数据；不支持硬链接时采用排他复制。
    返回本次发布文件的 (st_dev, st_ino)，供成对发布失败时核对清理归属。
    """
    source_info = source.stat()
    identity = source_info.st_dev, source_info.st_ino
    try:
        os.link(source, target)
    except FileExistsError:
        raise
    except OSError:
        with source.open("rb") as source_file:
            target_file = target.open("xb")
            target_info = os.fstat(target_file.fileno())
            identity = target_info.st_dev, target_info.st_ino
            try:
                with target_file:
                    shutil.copyfileobj(source_file, target_file)
                shutil.copystat(source, target)
            except BaseException:
                remove_owned_file(target, identity)
                raise
    try:
        source.unlink()
    except OSError:
        if source.exists():
            remove_owned_file(target, identity)
        raise
    return identity
