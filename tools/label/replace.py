"""
tools/label/replace.py
LabelMe 标签替换/追加工具。

读取 LabelMe 标注目录，对 `face` 等指定标签执行"替换"或"追加"操作：
- 替换：把源 label 改成目标 label；
- 追加：在保留源 label 的同时，复制一份 shape 并赋予新 label。

所有未被规则命中的标注均原样保留，不会删除任何已有标签。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.core import (
    find_image_for_json,
    list_labelme_files,
    load_labelme,
    save_labelme,
)


# =============================================================================
# 1. 标签规则解析
# =============================================================================


def parse_label_rules(arg: str | None) -> dict[str, str]:
    """解析形如 `face=person,face=mask` 的标签映射字符串。

    多个规则用逗号分隔；源标签与目标标签之间用 `=` 连接。
    空字符串返回空字典。
    """
    if not arg:
        return {}
    rules: dict[str, str] = {}
    for part in arg.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        source, target = part.split("=", 1)
        rules[source.strip()] = target.strip()
    return rules


# =============================================================================
# 2. Shape 处理
# =============================================================================


def replace_labels(shapes: list[dict], rules: dict[str, str]) -> list[dict]:
    """按 `rules` 替换 shape 的 `label`，未命中规则的原样保留。"""
    new_shapes: list[dict] = []
    for shape in shapes:
        label = shape.get("label")
        if label in rules:
            shape = dict(shape)
            shape["label"] = rules[label]
        new_shapes.append(shape)
    return new_shapes


def append_labels(shapes: list[dict], rules: dict[str, str]) -> list[dict]:
    """按 `rules` 为指定 label 的 shape 追加一份副本，并使用新 label。

    原始 shape 保持不变，因此该操作不会删除任何标签。
    """
    new_shapes: list[dict] = list(shapes)
    for shape in shapes:
        label = shape.get("label")
        if label in rules:
            appended = dict(shape)
            appended["label"] = rules[label]
            new_shapes.append(appended)
    return new_shapes


# =============================================================================
# 3. 单文件处理
# =============================================================================


def process_labelme_file(
    json_path: Path,
    output_dir: Path,
    replace_rules: dict[str, str],
    append_rules: dict[str, str],
) -> None:
    """读取单个 LabelMe JSON，执行替换/追加后写出，并复制同名图片。"""
    data = load_labelme(json_path)
    shapes = data.get("shapes", [])

    # 先追加（基于原始标签），再替换；这样替换后追加的副本仍然保留。
    shapes = append_labels(shapes, append_rules)
    shapes = replace_labels(shapes, replace_rules)

    data["shapes"] = shapes

    out_json = output_dir / json_path.name
    save_labelme(data, out_json)

    image_path = find_image_for_json(json_path)
    if image_path is not None:
        shutil.copy2(str(image_path), str(output_dir / image_path.name))


# =============================================================================
# 4. 命令行参数
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """构造脚本的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="LabelMe 人脸标签替换工具：只替换和追加，不删除。",
    )
    parser.add_argument(
        "--input", required=True, help="输入 LabelMe 目录"
    )
    parser.add_argument(
        "--output", required=True, help="输出目录"
    )
    parser.add_argument(
        "--replace",
        default="",
        help="替换规则，如 face=person；多个规则用逗号分隔",
    )
    parser.add_argument(
        "--append",
        default="",
        help="追加规则，如 face=mask；会保留原标签并额外添加新标签",
    )
    return parser


# =============================================================================
# 5. 主入口与执行
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """脚本主入口：解析参数并执行标签替换/追加链路。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"[错误] 输入目录不存在：{input_dir}", file=sys.stderr)
        return 1

    replace_rules = parse_label_rules(args.replace)
    append_rules = parse_label_rules(args.append)

    if not replace_rules and not append_rules:
        print("[错误] 请至少指定 --replace 或 --append", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    labelme_files = list_labelme_files(input_dir)
    if not labelme_files:
        print(f"[错误] 未找到 LabelMe JSON 文件：{input_dir}", file=sys.stderr)
        return 1

    for json_path in labelme_files:
        process_labelme_file(json_path, output_dir, replace_rules, append_rules)

    print(
        f"[完成] 共处理 {len(labelme_files)} 个文件，输出到：{output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
