# YOLOv8_dual_Stream 项目梳理

## 1. 项目是干什么的

这个仓库本质上是一个基于 `Ultralytics YOLOv8` 改出来的双流视觉模型项目，核心目标不是普通单目检测，而是：

- 同时输入两路图像做视觉理解，例如 `RGB + Depth` 或 `RGB + IR`
- 在 YOLO 框架里做双流特征提取与融合
- 重点面向实例分割、检测，以及论文里提到的箱体抓取场景
- 同时支持论文实验场景中的多组对比模型，例如 `Add`、`REF`、`SLBAF`、`CMMHA`、`Mamba` 等融合结构

从 README 的描述看，这个项目对应的是论文里的 `DS-YOLO` 思路：

- 在杂乱堆叠的箱体场景上做 `RGBD` 实例分割
- 再结合后续特征点匹配/抓取参数计算，完成抓取与摆放
- 也把双流结构扩展到了 `RGB + IR` 数据集，如 `LLVIP`、`M3FD`

所以它不是“纯 YOLOv8 官方仓库”，而是：

- 以 `Ultralytics 8.2.0` 为底座
- 在 `数据读取`、`模型前向`、`融合模块`、`模型 YAML` 上做了双流化改造
- 外层再配了简单的训练脚本和推理脚本

## 2. 项目整体结构

### 2.1 总体分层

这个仓库可以分成 4 层：

1. `最外层项目文件`
   - 说明项目用途、依赖、训练/推理入口。
2. `model/`
   - 放各种双流模型结构配置 YAML，是“实验配置层”。
3. `ultralytics/`
   - 放真正执行训练、推理、验证、模型搭建的核心代码，是“框架层”。
4. `doc/`
   - 文档目录，我这次新增的说明文档放在这里。

### 2.2 运行链路

项目的主链路可以理解为：

1. `train.py` 或 `predict.py` 作为入口。
2. 入口脚本用 `from ultralytics import YOLO` 加载某个 YAML 或权重。
3. `ultralytics/nn/tasks.py` 负责解析 YAML，并把模型改造成双输入前向：
   - 第一流是 `x`
   - 第二流是 `x2`
4. `ultralytics/data/build.py`、`ultralytics/data/utils.py`、验证器等代码把数据组织成：
   - `img_rgb`
   - `img_depth`
5. `ultralytics/nn/modules/rgbd_fusion.py` 中的融合模块在多尺度特征层完成融合。
6. `ultralytics/models/yolo/segment`、`detect` 等任务头输出结果。

## 3. 当前仓库状态判断

从仓库内容看，它更像“论文核心代码快照”，不是完全自洽的可直接运行工程，原因如下：

- `predict.py` 依赖 `catch.dataset_utils`，但仓库里没有 `catch/` 目录。
- `train.py` 使用 `dataset/M3FD.yaml`，但仓库里没有 `dataset/` 目录。
- `ultralytics/data/__init__.py` 和 `ultralytics/data/build.py` 引用了 `ultralytics.data.dataset`，但当前仓库里缺少 `ultralytics/data/dataset.py`。
- `predict.py` 和部分注释中存在大量硬编码本地路径，说明作者本地环境依赖比较重。

因此更准确的定位是：

- 这是一个“二次开发过的 Ultralytics 双流研究代码仓库”
- 主要有参考、复现实验、继续改模型的价值
- 如果要直接跑，需要你补齐数据集描述、缺失模块和本地路径配置

## 4. 顶层目录和文件说明

| 路径 | 用途 |
| --- | --- |
| `model/` | 自定义模型结构 YAML 目录，定义双流 backbone、融合方式和输出头。 |
| `ultralytics/` | 项目核心代码，基于官方 Ultralytics 修改而来。 |
| `.gitignore` | Git 忽略规则。 |
| `.pre-commit-config.yaml` | 代码提交前检查配置。 |
| `CITATION.cff` | 论文/项目引用信息。 |
| `CONTRIBUTING.md` | 贡献说明，基本来自上游 Ultralytics。 |
| `LICENSE` | 开源许可证。 |
| `mkdocs.yml` | 文档站点配置，基本属于 Ultralytics 文档体系。 |
| `predict.py` | 简单推理脚本，演示如何加载权重并输入双流图片做预测。依赖仓库外的 `catch` 包。 |
| `pyproject.toml` | Python 包构建与依赖配置，声明该包名仍是 `ultralytics`。 |
| `README.md` | 项目英文说明，介绍 DS-YOLO、数据集、实验结果、视频链接。 |
| `README.zh-CN.md` | 中文 README，内容更长，但文件编码看起来有些异常。 |
| `system.png` | 论文/系统框图图片。 |
| `train.py` | 简单训练入口，调用 `YOLO(...).train(...)`。当前硬编码使用 `dataset/M3FD.yaml`。 |
| `doc/` | 文档目录，本次新增项目梳理文档放在这里。 |

## 5. `model/` 目录说明

这个目录是项目里最容易直接改动、也最体现实验思路的地方。每个 YAML 都是在定义一套网络拓扑。

| 文件 | 用途 |
| --- | --- |
| `model/yolov8m.yaml` | 常规 YOLOv8m 结构配置，可视为基线。 |
| `model/yolov8m-seg-default.yaml` | 常规 YOLOv8m 分割配置，作为双流改造前的分割基线。 |
| `model/yolov8-seg.yaml` | 双流分割模型配置，使用 `Concat3` 等方式把两路输入并入 YOLOv8 分割结构。`train.py` 当前就是拿它训练。 |
| `model/yolov8m-seg-2stream-add-decect.yaml` | 双流检测/分割变体，使用简单 `Add` 做融合。文件名里 `decect` 应该是 `detect` 的拼写笔误。 |
| `model/yolov8m-seg-2stream-add-postfusion.yaml` | 双流后融合版本，核心融合策略是直接逐层相加。 |
| `model/yolov8m-seg-2stream-cmm-postfusion.yaml` | 使用跨模态/注意力式融合的后融合版本。 |
| `model/yolov8m-seg-2stream-cmmha-allfusion.yaml` | 使用多头跨模态注意力的全阶段融合版本。 |
| `model/yolov8m-seg-2stream-ref-allfusion.yaml` | 使用 `ResidualExciteFusion` 的全阶段融合版本。 |
| `model/yolov8m-seg-2stream-ref-postfusion.yaml` | 使用 `ResidualExciteFusion` 的后融合版本。 |
| `model/yolov8m-seg-2stream-sef-postfusion.yaml` | 使用 `SEF/Squeeze-and-Excite` 类融合模块的后融合版本。 |
| `model/yolov8m-seg-2stream-SLBAFNET.yaml` | 双流 + `SLBAFNet` 风格融合结构，属于论文对比或实验模型之一。 |
| `model/yolov5l_fusion_add_llvip.yaml` | 面向 `LLVIP` 的 YOLOv5 风格双流融合结构。 |
| `model/yolov8L_fusion_transformer_FLIR.yaml` | 面向 `FLIR` 或类似 RGB-IR 任务的 YOLOv8L 双流 Transformer 融合结构。 |
| `model/Mamba-YOLO-L.yaml` | 结合 Mamba 风格主干/模块的实验配置。 |

### 5.1 `model/` 目录怎么理解

如果你以后要继续改论文模型，优先看这个目录：

- 想换融合方式：改这里
- 想增减融合层：改这里
- 想切换 RGBD / RGB-IR 实验版本：改这里
- 想快速复现实验对比：多数情况下也是从这里选 YAML

## 6. `ultralytics/` 目录说明

这个目录是整个项目的“引擎室”。绝大多数内容来自官方 Ultralytics，但有一部分明显做了双流定制。

### 6.1 `ultralytics/__init__.py`

| 文件 | 用途 |
| --- | --- |
| `ultralytics/__init__.py` | 包入口，暴露 `YOLO`、`RTDETR`、`SAM`、`FastSAM`、`Explorer` 等对象；版本号是 `8.2.0`。 |

### 6.2 `ultralytics/cfg/`

这是配置中心，负责默认训练参数、跟踪器配置以及各版本模型模板。

| 路径 | 用途 |
| --- | --- |
| `ultralytics/cfg/default.yaml` | 训练、验证、预测的默认参数。 |
| `ultralytics/cfg/trackers/bytetrack.yaml` | ByteTrack 配置。 |
| `ultralytics/cfg/trackers/botsort.yaml` | BoT-SORT 配置。 |
| `ultralytics/cfg/models/v3/` | YOLOv3 模板。 |
| `ultralytics/cfg/models/v5/` | YOLOv5 模板。 |
| `ultralytics/cfg/models/v6/` | YOLOv6 模板。 |
| `ultralytics/cfg/models/v8/` | YOLOv8 官方模板。 |
| `ultralytics/cfg/models/v9/` | YOLOv9 模板。 |
| `ultralytics/cfg/models/rt-detr/` | RT-DETR 模型模板。 |

这些文件大多是上游模板，本仓库主要还是用 `model/` 目录下那些自定义 YAML。

### 6.3 `ultralytics/data/`

这是双流项目里非常关键的一层，因为数据不再是单图像，而是两路图像共同喂入模型。

| 文件/目录 | 用途 |
| --- | --- |
| `ultralytics/data/__init__.py` | 数据模块入口，导出数据集类和构建函数。 |
| `ultralytics/data/base.py` | 数据集基类，定义通用数据读取框架。 |
| `ultralytics/data/build.py` | 构建 dataloader 和数据集；这里已经改成支持 `img_path_rgb` + `img_path_depth`。 |
| `ultralytics/data/loaders.py` | 推理时加载图片、视频、流、张量等输入。 |
| `ultralytics/data/augment.py` | 训练增强逻辑。 |
| `ultralytics/data/annotator.py` | 标注辅助工具。 |
| `ultralytics/data/converter.py` | 数据集格式转换工具。 |
| `ultralytics/data/split_dota.py` | DOTA 数据切分工具，基本是上游能力。 |
| `ultralytics/data/utils.py` | 数据集检查、路径解析、缓存、mask 处理。这里已改成识别 `train_rgb`、`train_depth`、`val_rgb`、`val_depth`。 |
| `ultralytics/data/scripts/` | 官方数据下载脚本。 |
| `ultralytics/data/explorer/` | 数据探索工具，依赖额外数据库/前端组件。 |

#### 双流相关关键点

- `ultralytics/data/build.py`
  - `build_yolo_dataset()` 已经改成接收 `img_path_rgb` 和 `img_path_depth`
  - 支持 `YOLOMultiModalDataset`
- `ultralytics/data/utils.py`
  - 数据 YAML 检查键位从单流的 `train/val` 改成了双流的 `train_rgb/train_depth/val_rgb/val_depth`
- `ultralytics/engine/validator.py`
  - 验证时直接调用 `model([batch['img_rgb'], batch['img_depth']])`

#### 当前缺口

当前仓库缺少 `ultralytics/data/dataset.py`，但很多地方都在引用它。这说明：

- 项目原始运行环境里应该有这个文件
- 但当前仓库并没有把它一起提交进来
- 如果你打算真正训练，这个文件必须补齐

### 6.4 `ultralytics/engine/`

这一层是训练、推理、验证、导出的通用流程控制。

| 文件 | 用途 |
| --- | --- |
| `ultralytics/engine/model.py` | `YOLO` 等高层模型接口。 |
| `ultralytics/engine/trainer.py` | 训练主流程。 |
| `ultralytics/engine/predictor.py` | 推理流程控制。 |
| `ultralytics/engine/validator.py` | 验证流程控制，本仓库已改成双流输入。 |
| `ultralytics/engine/results.py` | 结果对象封装。 |
| `ultralytics/engine/exporter.py` | 模型导出。 |
| `ultralytics/engine/tuner.py` | 超参数搜索。 |
| `ultralytics/engine/__init__.py` | engine 包初始化。 |

#### 和本项目最相关的文件

- `ultralytics/engine/validator.py`
  - 已明确改为双流验证
- `ultralytics/engine/predictor.py`
  - 仍保持 Ultralytics 的推理骨架
  - 真正的双流适配主要体现在传给模型的是双图列表，而不是单图

### 6.5 `ultralytics/models/`

这是任务层封装，决定“做检测、分割、姿态还是别的任务”。

| 路径 | 用途 |
| --- | --- |
| `ultralytics/models/__init__.py` | 模型集合入口。 |
| `ultralytics/models/yolo/` | YOLO 系列任务实现。 |
| `ultralytics/models/rtdetr/` | RT-DETR 任务实现。 |
| `ultralytics/models/nas/` | NAS 相关模型。 |
| `ultralytics/models/sam/` | Segment Anything 相关实现。 |
| `ultralytics/models/fastsam/` | FastSAM 相关实现。 |
| `ultralytics/models/utils/` | 模型公共工具。 |

#### `ultralytics/models/yolo/` 子目录

| 路径 | 用途 |
| --- | --- |
| `ultralytics/models/yolo/model.py` | YOLO 任务统一入口。 |
| `ultralytics/models/yolo/detect/` | 检测任务的 `train.py`、`val.py`、`predict.py`。 |
| `ultralytics/models/yolo/segment/` | 分割任务的 `train.py`、`val.py`、`predict.py`。本项目最相关。 |
| `ultralytics/models/yolo/classify/` | 分类任务实现。 |
| `ultralytics/models/yolo/pose/` | 姿态估计任务实现。 |
| `ultralytics/models/yolo/obb/` | 旋转框任务实现。 |
| `ultralytics/models/yolo/world/` | YOLO-World 相关实现。 |
| `ultralytics/models/yolo/__init__.py` | yolo 子包入口。 |

整体上，这一层大多仍是上游代码，真正把单流改成双流的关键不在这里，而在：

- `data`
- `nn`
- 自定义模型 YAML

### 6.6 `ultralytics/nn/`

这是本项目最值得重点看的目录，因为模型结构和双流改造主要发生在这里。

| 文件/目录 | 用途 |
| --- | --- |
| `ultralytics/nn/tasks.py` | 模型解析和前向逻辑核心文件，本仓库已改成双输入前向。 |
| `ultralytics/nn/modules/__init__.py` | 注册各类基础模块和自定义融合模块。 |
| `ultralytics/nn/modules/conv.py` | 卷积类基础模块。 |
| `ultralytics/nn/modules/block.py` | YOLO 常见 block，如 C2f、C3 等。 |
| `ultralytics/nn/modules/head.py` | Detect、Segment、Pose 等任务头。 |
| `ultralytics/nn/modules/transformer.py` | Transformer 类基础模块。 |
| `ultralytics/nn/modules/rgbd_fusion.py` | 本项目最核心的双流融合模块实现文件。 |
| `ultralytics/nn/modules/mamba_yolo.py` | Mamba-YOLO 相关模块。 |
| `ultralytics/nn/modules/mambaout.py` | MambaOut 相关模块。 |
| `ultralytics/nn/autobackend.py` | 多后端推理封装。 |
| `ultralytics/nn/v10.py` | YOLOv10 相关模块。 |
| `ultralytics/nn/yolo12.py` | YOLO12 相关实验模块。 |
| `ultralytics/nn/deyolo.py` | 另一组双流/融合实验模块。 |
| `ultralytics/nn/Asyolo.py` | 自定义实验模块。 |
| `ultralytics/nn/BiFocus.py` | 自定义特征处理模块。 |
| `ultralytics/nn/CAFMAttention.py` | 自定义注意力模块。 |
| `ultralytics/nn/C2f_GhostModule_DynamicConv.py` | Ghost/DynamicConv 相关实验模块。 |
| `ultralytics/nn/__init__.py` | nn 包入口。 |

#### `ultralytics/nn/tasks.py` 的作用

这个文件是双流改造的主枢纽：

- 把默认的 `forward(x)` 改成了能处理两路输入
- `predict(x, x2)`、`_predict_once(x, x2)` 已支持双输入
- 利用 `m.f == -4` 这种方式从第二流取特征
- loss/验证流程里也改成了 `batch["img_rgb"]` 和 `batch["img_depth"]`

可以说，`tasks.py` 让“YAML 里写双流网络”这件事真正可执行。

#### `ultralytics/nn/modules/rgbd_fusion.py` 的作用

这是本项目的研究价值核心，里面实现了大量双流融合模块，包括但不限于：

- `DAF`
- `iAFF`
- `AFF`
- `MS_CAM`
- `Add`
- `Add2`
- `GPT`
- `SelfAttention`
- `ResidualExciteFusion`
- `ResidualAttentionFusion`
- `MHAttentionFusionSecond`
- `MHAttentionFusionThird`
- `CrossModalMultiHeadAttention`
- `Concat3`

这些模块就是论文实验里各种“融合策略”的代码来源。

### 6.7 `ultralytics/trackers/`

多目标跟踪模块，基本来自上游能力，不是这个仓库当前论文主线。

| 文件/目录 | 用途 |
| --- | --- |
| `ultralytics/trackers/track.py` | 跟踪主入口。 |
| `ultralytics/trackers/basetrack.py` | 跟踪器基类。 |
| `ultralytics/trackers/byte_tracker.py` | ByteTrack 实现。 |
| `ultralytics/trackers/bot_sort.py` | BoT-SORT 实现。 |
| `ultralytics/trackers/utils/` | 匹配、卡尔曼滤波、GMC 等工具。 |
| `ultralytics/trackers/__init__.py` | trackers 包入口。 |

### 6.8 `ultralytics/utils/`

这一层是工具箱，负责日志、下载、损失、度量、绘图、回调、设备管理等。

| 路径 | 用途 |
| --- | --- |
| `ultralytics/utils/__init__.py` | 全局常量、日志和公共函数。 |
| `ultralytics/utils/checks.py` | 环境检查。 |
| `ultralytics/utils/downloads.py` | 下载相关工具。 |
| `ultralytics/utils/files.py` | 文件路径工具。 |
| `ultralytics/utils/loss.py` | 检测/分割等损失函数。 |
| `ultralytics/utils/metrics.py` | mAP、混淆矩阵等指标。 |
| `ultralytics/utils/ops.py` | 框、mask、后处理算子。 |
| `ultralytics/utils/plotting.py` | 可视化绘图。 |
| `ultralytics/utils/torch_utils.py` | 设备、模型融合、权重初始化等。 |
| `ultralytics/utils/benchmarks.py` | 基准测试。 |
| `ultralytics/utils/autobatch.py` | 自动 batch 估计。 |
| `ultralytics/utils/tal.py` | Task-Aligned Learning 相关逻辑。 |
| `ultralytics/utils/triton.py` | Triton 推理支持。 |
| `ultralytics/utils/tuner.py` | 调参工具。 |
| `ultralytics/utils/errors.py` | 异常定义。 |
| `ultralytics/utils/instance.py` | 实例对象表示。 |
| `ultralytics/utils/patches.py` | 兼容性补丁。 |
| `ultralytics/utils/dist.py` | 分布式训练辅助。 |
| `ultralytics/utils/callbacks/` | 和 TensorBoard、MLflow、W&B、Comet、ClearML 等平台联动。 |

这些大多是框架基础设施，改动价值低，除非你要调训练流程或日志系统。

### 6.9 `ultralytics/hub/`

与 Ultralytics HUB 平台交互的代码，和本项目双流研究关系不大。

| 文件 | 用途 |
| --- | --- |
| `ultralytics/hub/__init__.py` | HUB 入口。 |
| `ultralytics/hub/auth.py` | HUB 登录认证。 |
| `ultralytics/hub/session.py` | HUB 会话管理。 |
| `ultralytics/hub/utils.py` | HUB 工具函数。 |

### 6.10 `ultralytics/solutions/`

官方演示级解决方案集合，基本不是本项目主线。

| 文件 | 用途 |
| --- | --- |
| `ultralytics/solutions/ai_gym.py` | 运动/姿态类示例。 |
| `ultralytics/solutions/analytics.py` | 分析示例。 |
| `ultralytics/solutions/distance_calculation.py` | 距离计算示例。 |
| `ultralytics/solutions/object_counter.py` | 计数示例。 |
| `ultralytics/solutions/parking_management.py` | 停车管理示例。 |
| `ultralytics/solutions/queue_management.py` | 排队管理示例。 |
| `ultralytics/solutions/speed_estimation.py` | 速度估计示例。 |
| `ultralytics/solutions/__init__.py` | solutions 包入口。 |
| `ultralytics/solutions.zip` | 可能是附带的解决方案打包文件，不属于双流核心。 |

## 7. 最关键的几个文件

如果只想最快读懂这个仓库，优先看下面这些文件：

| 文件 | 为什么优先看 |
| --- | --- |
| `README.md` | 明确项目论文背景、任务目标、实验结果。 |
| `train.py` | 明确作者当前默认怎么训练。 |
| `predict.py` | 明确作者当前默认怎么推理双流输入。 |
| `model/yolov8-seg.yaml` | 看作者如何把两路输入并到 YOLOv8 分割结构里。 |
| `model/yolov8m-seg-2stream-*.yaml` | 看不同融合实验是怎么切换的。 |
| `ultralytics/nn/tasks.py` | 看双流前向是怎么真正跑起来的。 |
| `ultralytics/nn/modules/rgbd_fusion.py` | 看各种融合模块的实现细节。 |
| `ultralytics/data/build.py` | 看双流 dataset/dataloader 怎么接。 |
| `ultralytics/data/utils.py` | 看数据 YAML 要满足什么键名格式。 |
| `ultralytics/engine/validator.py` | 看验证阶段如何传双流输入。 |

## 8. 这个仓库更适合怎么用

### 8.1 如果你的目标是“快速理解项目”

建议阅读顺序：

1. `README.md`
2. `train.py`
3. `model/yolov8-seg.yaml`
4. `ultralytics/nn/tasks.py`
5. `ultralytics/nn/modules/rgbd_fusion.py`
6. `ultralytics/data/build.py`
7. `ultralytics/data/utils.py`

### 8.2 如果你的目标是“继续做实验”

建议重点改这三类文件：

- `model/*.yaml`
  - 改网络拓扑和融合位置
- `ultralytics/nn/modules/rgbd_fusion.py`
  - 改融合算子本体
- `ultralytics/nn/tasks.py`
  - 改模型解析与双输入逻辑

### 8.3 如果你的目标是“真正跑通训练”

你还需要补齐下面这些外部条件：

- 数据集 YAML，例如 `M3FD.yaml`、`LLVIP.yaml`
- 实际数据目录
- 缺失的 `ultralytics/data/dataset.py`
- `predict.py` 依赖的 `catch/` 工具包
- 本地硬编码路径改造成相对路径或配置项

## 9. 一句话总结

这个仓库的本质不是一个普通的 YOLOv8 demo，而是一个围绕 `RGBD / RGB-IR 双流融合` 做研究实验的 Ultralytics 二次开发版本：

- `model/` 是实验配置层
- `ultralytics/nn/` 是模型改造层
- `ultralytics/data/` 是双流数据接入层
- `train.py` / `predict.py` 是作者本地使用的轻量入口

如果后续你要继续维护它，最值得长期关注的是：

- 双流数据集定义是否补全
- 融合模块是否统一整理
- 训练脚本是否去掉硬编码
- 不同 YAML 实验配置是否补充注释和命名规范
