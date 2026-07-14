"""数据标注工具（annotate）。

结构：

- ``backends/`` —— 检测器后端（按模型类型分离：onnx / sam / yolo / detr）
- ``ops``       —— 打标/覆盖共用底层（框几何 / 打码 / LabelMe IO）
- ``auto``      —— 统一标注入口（supervision 后端：YOLO / SAM3 / DETR；
  ONNX 后端：两路打标 / 覆盖，打码抽到 ops.apply_blackout）
- ``merge``     —— 框合并
"""
