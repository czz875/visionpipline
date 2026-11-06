# tools/core — 工具脚本共享组件包

本目录存放 `tools/` 下各脚本共享的**低代码组件**，目标是把通用能力（文件扫描、LabelMe I/O、边界框几何）抽离出来，让新增工具脚本只需“按需导入、填空业务逻辑”。

---

## 设计原则

- **单一职责**：每个子模块只处理一个领域，如 `images.py` 只负责图片扫描，`labelme.py` 只负责 LabelMe JSON。
- **包级导出**：常用符号在 `tools.core.__init__` 中统一导出，脚本推荐 `from tools.core import ...` 的方式使用。
- **零外部重依赖**：`constants`、`images`、`labelme`、`geometry` 仅依赖标准库，避免工具脚本因导入公共包而引入 `torch/transformers` 等重型依赖。
- **命令行友好**：业务脚本负责 argparse，公共包只提供可复用的函数积木。

---

## 模块速查

| 模块 | 说明 | 主要导出 |
|---|---|---|
| `constants.py` | 共享常量 | `DEFAULT_DATASET_PATH`、`IMAGE_EXTENSIONS`、`LABELME_EXT`、`BATCH_DIR_RE`、`YOLO_CLASS_NAMES`、`CLASS_PRIORITY` |
| `images.py` | 图片文件扫描 | `list_images(folder, extensions=..., recursive=...)` |
| `image_io.py` | 中文路径安全的图片读写 | `read_image`、`write_image`、`image_size` |
| `file_io.py` | 原子写盘与不覆盖目标的移动 | `atomic_write_bytes`、`move_file_without_overwrite`、`remove_owned_file` |
| `labelme.py` | LabelMe JSON 扫描与读写 | `list_labelme_files`、`load_labelme`、`save_labelme`、`find_image_for_json`、`find_json_for_image` |
| `geometry.py` | 边界框几何工具 | `rect_to_xyxy`、`xyxy_to_points`、`get_boxes_dist`、`merge_near_boxes` |
| `output_dir.py` | 输出目录协议 | `build_timestamped_output_dir`、`build_batch_stage_dir`、`resolve_latest_batch_stage_dir` |
| `copy_dataset.py` | 数据集复制 / 硬链接 | `copy_dataset` |

---

## 包级 API

```python
from tools.core import (
    DEFAULT_DATASET_PATH,
    IMAGE_EXTENSIONS,
    LABELME_EXT,
    list_images,
    list_labelme_files,
    load_labelme,
    save_labelme,
    find_image_for_json,
    find_json_for_image,
    rect_to_xyxy,
    xyxy_to_points,
    get_boxes_dist,
    merge_near_boxes,
)
```

---

## 使用示例

### 1. 扫描目录下的图片与 JSON

```python
from pathlib import Path
from tools.core import list_images, list_labelme_files

folder = Path("datasets/behavior")
images = list_images(folder)
jsons = list_labelme_files(folder)
```

### 2. 读取并修改 LabelMe JSON

```python
from tools.core import load_labelme, save_labelme

json_path = Path("datasets/behavior/0001/000001.json")
data = load_labelme(json_path)

# 修改 shapes ...

save_labelme(data, json_path)
```

`save_labelme` 和 `write_image` 会先写入目标同目录的临时文件，再原子替换指定路径。
发布失败时保留原文件并清理本次临时文件；目标与其他文件互为硬链接时，其他链接的内容保持不变。
符号链接仍指向原目标，写入会在实际目标处原子替换；POSIX 已有文件权限保留，新文件遵循正常 umask。
父目录由调用方准备。`move_file_without_overwrite` 用于新文件发布，目标已存在时直接失败；
不支持硬链接或跨卷时采用排他复制，并在复制失败时清理本次创建的目标。
它返回发布文件的设备号和 inode，`remove_owned_file` 在清理前按该标识复核，保留已识别的替换文件。
身份复核与删除不是跨进程事务，调用方应避免其他程序同时覆盖同一路径。

### 3. 图片与 JSON 互找

```python
from tools.core import find_image_for_json, find_json_for_image

json_path = Path("datasets/behavior/0001/000001.json")
img_path = find_image_for_json(json_path)  # 未找到返回 None

image_path = Path("datasets/behavior/0001/000001.jpg")
json_path = find_json_for_image(image_path)
```

### 4. 合并邻近边界框

```python
from tools.core.geometry import merge_near_boxes

boxes = [(10, 10, 50, 50), (55, 10, 100, 50)]
merged = merge_near_boxes(
    boxes,
    distance_x=20,
    distance_y=20,
    max_width=200,
    max_height=200,
    max_count=5,
)
```

---

## 新增脚本的最佳实践

1. 先判断需要哪些公共能力（图片扫描 / JSON 读写 / 框合并 / CLI）。
2. 从 `tools.core` 或对应子模块导入，不要在脚本里重新实现。
3. 业务脚本负责：
   - `argparse` 参数解析；
   - 进度条 / 多线程等业务流程；
   - 特定领域的判断逻辑（如过滤条件、标签映射）。
4. 如果新能力具有通用性，继续补充到 `tools/core/` 的对应模块中，并在 `__init__.py` 导出。

---

## 直接运行脚本时的路径兼容

`tools/` 下的脚本支持两种方式执行：

- 从仓库根目录：`python tools/xxx.py`
- 直接双击 / 在 `tools/` 目录下运行：`python xxx.py`

推荐在每个脚本顶部保留这段路径保护，确保能正确导入 `tools.core`：

```python
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```
