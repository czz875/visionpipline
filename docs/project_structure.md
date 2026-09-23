# 项目目录结构

> 目录速查与职责说明。协作规则见 [AGENTS.md](../AGENTS.md)。

## 目录结构

```text
visionpipline/
├── .conda/                        # 项目自带的便携式 Python 3.11
├── datasets/                      # 数据集（raw_jpg / raw / yolo / autolabel / behavior ...）
├── runs/                          # yolo detect train 输出
├── archive/                       # tools/train/archive.py 生成的每日归档
├── weight/                        # 训练好的 YOLO 权重
├── docs/                          # 文档与参考资料（如 superpowers/）
├── tools/                         # 业务脚本（按阶段分组，仿 ultralytics 包结构）
│   ├── __init__.py                #   顶层 API 导出
│   ├── core/                      # 公共模块
│   │   ├── constants.py           #   常量（DEFAULT_DATASET_PATH 等）
│   │   ├── images.py              #   list_images 等
│   │   ├── labelme.py             #   LabelMe JSON 扫描/读写
│   │   ├── geometry.py            #   矩形/合并/距离工具
│   │   ├── README.md              #   公共模块使用说明
│   │   └── __init__.py            #   统一对外导出
│   ├── cfg/                       # 工作流配置（仿 ultralytics/cfg）
│   │   ├── __init__.py            #   load_config / resolve_config / substitute_variables
│   │   ├── default.yaml           #   系统默认（paths / parameters / log_file）
│   │   ├── workflow.yaml          #   系统主工作流 stage 定义
│   │   ├── inherit_yolo.yaml      #   任务专项：接续 + 重命名 + 转 YOLO（走 stages_only）
│   │   ├── hand_face_mosaic.yaml  #   任务专项：备份 datasets/1|2|3 + 脸(ONNX)/手(SAM) 标注打码（走 stages_only）
│   │   ├── all_modules.yaml.example # 全部模块「用法总表」：每个 tools 脚本一个 stage，带 enabled/order
│   │   ├── detectors.yaml.example #   多检测器组合标注模板（--detectors-config 引用，见 §4.10）
│   │   └── workflow_config.yaml.example   #   项目级覆盖示例（复制为 src/workflow_config.yaml）
│   ├── engine/                    # 各 stage 聚合入口（仿 ultralytics/engine）
│   │   └── __init__.py            #   把 10 个同级 stage 包（annotate/augment/.../train）re-export 到 tools.engine 命名空间（engine 自身无子包）
│   ├── annotate/                  # 标注阶段
│   │   ├── backends/              #   检测器后端（按模型类型分离）
│   │   │   ├── base.py            #     AutoLabeler 抽象接口 + 类型别名
│   │   │   ├── onnx.py            #     ONNX 功能（OnnxDetector + YOLO 风格解码）
│   │   │   ├── sam.py             #     ultralytics SAM（SAMTextDetector / SAM3Labeler）
│   │   │   ├── yolo.py            #     ultralytics YOLO（YOLOLabeler）
│   │   │   └── detr.py            #     ultralytics DETR（DETRLabeler，RT-DETR 实现）
│   │   ├── ops.py                 #   打标/覆盖共用底层：框几何 + 打码 + LabelMe IO
│   │   ├── auto.py                #   统一标注入口：supervision 后端（YOLO/SAM3/DETR）+ ONNX 后端（两路打标 / 覆盖）+ 多检测器组合（--detectors-config 任意 N 路混搭，打码抽到 ops.apply_blackout）
│   │   └── merge.py               #   框合并
│   ├── clean/                     # 清洗阶段
│   │   ├── blurry.py              #   CleanVision 模糊检测
│   │   ├── orphan_json.py         #   孤儿 JSON
│   │   └── orphan_images.py       #   缺失 JSON 的图片
│   ├── label/                     # 标签处理
│   │   ├── align_labelme.py       #   PNG/JSON 错位一键对齐（见 §6 约束）
│   │   ├── replace.py             #   批量标签替换
│   │   └── fix_labelme.py         #   修复 LabelMe JSON 常见损坏
│   ├── split/                     # 拆分阶段
│   │   └── dataset.py             #   数据集拆分
│   ├── augment/                   # 数据增强
│   │   └── ir_enhance.py          #   IR 全局/局部光照增强 + 传感器效果
│   ├── rename/                    # 文件批量改名
│   │   └── timestamp_rename.py    #   按时间戳改名为 YYYYMMDD_HHMMSS_NNNNNN[__ms]，可选同步 LabelMe imagePath
│   ├── backup/                    # 数据备份
│   │   └── snapshot.py            #   打 .tar.gz 备份（默认 dry-run）
│   ├── merge/                     # 数据接续/合并
│   │   └── inherit_dataset.py     #   autolabel 按 1000/批接续到 behavior
│   ├── convert/                   # 格式转换
│   │   ├── jpg_to_png.py          #   JPG 批量转 PNG + 修复 LabelMe JSON
│   │   ├── labelme_to_yolo.py
│   │   └── yolo_to_labelme.py
│   ├── train/                     # 训练相关
│   │   ├── archive.py             #   按时间戳归档
│   │   └── create_tar_gz.py       #   压缩归档为 .tar.gz
│   └── workflow.py                # 工作流编排器入口（薄编排，配置解析走 tools.cfg）
├── src/                           # 本地入口 + 临时工作流（被 .gitignore 整体忽略，不入 git）
│   ├── run.py                     #   临时工作流统一入口（按需切换 cfg；默认 dry_run=True）
│   ├── main.py                    #   详细参数版入口（Python 函数式）
│   ├── run_*.py                   #   任务专项入口变体（如 run_hand_face_714.py / run_clean_behavior_to_yolo.py）
│   ├── _run_hand.py               #   内部入口脚本
│   ├── workflow_config.yaml       #   当前项目级覆盖入口（--config 传入；见 §4.9）
│   └── *.yaml                     #   临时工作流 cfg（命名：<功能>.yaml，如 clean_behavior_to_yolo.yaml / recover_yolo0708.yaml）
├── workflow.md                    # 工作流说明文档
├── requirements.txt               # 依赖清单
└── AGENTS.md                      # 本文件
```

---
