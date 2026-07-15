"""
tools/core

工具脚本共享的低代码组件包。

常用符号已在本包顶层重新导出，方便脚本以
`from tools.core import list_images, load_labelme` 的方式即插即用。
"""

from __future__ import annotations

from tools.core.constants import (
    DEFAULT_DATASET_PATH,
    IMAGE_EXTENSIONS,
    LABELME_EXT,
)
from tools.core.geometry import (
    get_boxes_dist,
    merge_near_boxes,
    rect_to_xyxy,
    xyxy_to_points,
)
from tools.core.copy_dataset import copy_dataset
from tools.core.images import list_images
from tools.core.labelme import (
    detections_to_labelme_dict,
    find_image_for_json,
    find_json_for_image,
    labelme_dict_to_detections,
    list_labelme_files,
    load_labelme,
    save_labelme,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "IMAGE_EXTENSIONS",
    "LABELME_EXT",
    "detections_to_labelme_dict",
    "find_image_for_json",
    "find_json_for_image",
    "get_boxes_dist",
    "labelme_dict_to_detections",
    "copy_dataset",
    "list_images",
    "list_labelme_files",
    "load_labelme",
    "merge_near_boxes",
    "rect_to_xyxy",
    "save_labelme",
    "xyxy_to_points",
]
