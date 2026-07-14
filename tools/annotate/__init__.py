"""数据标注工具（annotate）。

结构：

- ``backends/`` —— 检测器后端（按模型类型分离：onnx / sam / yolo / detr）
- ``ops``       —— 打标/覆盖共用底层（框几何 / 打码 / LabelMe IO）
- ``auto``      —— 打标编排（supervision 数据集式：YOLO / SAM3 / DETR）
- ``auto_onnx_sam`` —— 打标编排（ONNX + SAM 两路 + 打码）
- ``reannotate_onnx`` —— 标签覆盖（ONNX 覆盖指定类别 + 保留其它类别）
- ``merge``     —— 框合并
"""
