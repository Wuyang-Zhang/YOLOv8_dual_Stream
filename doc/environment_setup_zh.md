# 环境安装说明

## 1. 推荐环境

这个仓库建议优先使用下面这套环境：

- Windows 10/11
- Python 3.10
- PyTorch 2.1.x 或 2.2.x
- CUDA 11.8 或 CUDA 12.1

不建议一开始就用 Python 3.12，因为部分第三方包兼容性更容易出问题。

## 2. 先说明两种安装目标

### 2.1 只跑双流预测链路

如果你现在只是想：

- 跑通 `predict.py`
- 验证双流 `RGB + Depth/IR` 输入链路
- 不训练模型

那只需要装最小依赖。

### 2.2 后续要训练

如果你后面还要：

- 跑 `train.py`
- 跑验证
- 调整模型 YAML

那建议直接按“训练环境”来装。

## 3. Conda 创建环境

推荐用 Conda。

```bash
conda create -n ds-yolo python=3.10 -y
conda activate ds-yolo
python -V
```

## 4. 安装 PyTorch

先装 PyTorch，再装项目依赖。

### 4.1 如果你有 NVIDIA GPU，推荐 CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4.2 如果你只想先用 CPU 跑通链路

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

安装完成后检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 5. 安装项目基础依赖

在仓库根目录执行：

```bash
pip install -U pip setuptools wheel
pip install -e .
```

这一步会安装 `pyproject.toml` 里声明的大部分基础依赖，例如：

- `opencv-python`
- `pillow`
- `pyyaml`
- `requests`
- `scipy`
- `matplotlib`
- `pandas`
- `seaborn`
- `thop`

## 6. 安装这个仓库额外实际需要的依赖

这个仓库代码里还额外用到了几个 `pyproject.toml` 里没有完整声明、但实际会导入的包。

### 6.1 最小双流预测环境

如果你现在只跑默认双流预测脚本，建议再补：

```bash
pip install mmengine timm einops
```

原因：

- `mmengine`
  - [ultralytics/nn/modules/rgbd_fusion.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/modules/rgbd_fusion.py) 里会用到
- `timm`
  - 多个自定义模块会导入
- `einops`
  - 自定义注意力和 Mamba 模块里会导入

### 6.2 如果你还要用一些附加功能

```bash
pip install shapely tensorboard albumentations
```

用途：

- `shapely`
  - 某些 solutions 和 DOTA 切分工具会用
- `tensorboard`
  - 训练日志可视化
- `albumentations`
  - 部分训练增强场景可能会用到

## 7. 一套建议直接执行的命令

### 7.1 只跑预测链路

```bash
conda create -n ds-yolo python=3.10 -y
conda activate ds-yolo
pip install -U pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install mmengine timm einops
```

### 7.2 预测 + 后续训练都要

```bash
conda create -n ds-yolo python=3.10 -y
conda activate ds-yolo
pip install -U pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install mmengine timm einops shapely tensorboard albumentations
```

如果你没有 GPU，把 `cu121` 改成 `cpu` 即可。

## 8. 安装完成后的检查

### 8.1 检查基础导入

```bash
python -c "import cv2, torch, yaml, mmengine, timm, einops; print('ok')"
```

### 8.2 检查项目能否导入

```bash
python -c "from ultralytics import YOLO; print('ultralytics ok')"
```

### 8.3 检查预测脚本语法

```bash
python -m py_compile predict.py
```

## 9. 预测链路测试命令

环境装完以后，你可以先跑最简单的双流预测链路测试：

```bash
python predict.py --rgb path/to/rgb.jpg --aux path/to/ir_or_depth.jpg
```

如果你要指定模型：

```bash
python predict.py --model model/yolov8m-seg-2stream-ref-postfusion.yaml --rgb path/to/rgb.jpg --aux path/to/ir_or_depth.jpg
```

说明：

- 当前默认模型已经是双流 YAML
- 如果你现在没有训练权重，只是拿 YAML 跑，主要作用是验证链路能不能通
- 输出结果不代表模型有效，因为随机初始化权重没有实际检测能力

## 10. 训练前还需要知道的事情

即使环境装好了，这个仓库当前也不是立刻就能完整训练的，因为还有几个代码/数据缺口：

- 当前仓库缺少 `ultralytics/data/dataset.py`
- 数据集配置文件和数据目录需要你自己补齐
- 训练时双流数据必须成对组织好

也就是说：

- 环境问题解决后，你可以先跑预测链路
- 真正开始训练前，还要先把数据集加载部分补完整

## 11. 推荐阅读

如果你怕后面忘：

- 项目总梳理：
  - [doc/project_overview_zh.md](/g:/Code_Repository/YOLOv8_dual_Stream/doc/project_overview_zh.md)
- 双流预测链路：
  - [doc/dual_stream_inference_chain_zh.md](/g:/Code_Repository/YOLOv8_dual_Stream/doc/dual_stream_inference_chain_zh.md)
- 数据集模板：
  - [dataset/M3FD.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/dataset/M3FD.yaml)

## 12. `pip install -e .` 详细说明

### 12.1 这条命令拆开看是什么意思

```bash
pip install -e .
```

可以拆成三部分：

- `pip`
  - Python 的包管理工具
- `install`
  - 安装一个包
- `-e .`
  - 以可编辑模式安装当前目录这个项目

这里：

- `-e` 是 `editable`
- `.` 表示“当前目录”

因为你是在仓库根目录执行，所以这个命令的完整含义就是：

- 把当前这个仓库，以可编辑模式安装到当前 Python 环境

### 12.2 什么叫“可编辑模式”

可编辑模式的核心意思是：

- 不是把代码复制一份到环境里再用
- 而是让环境直接指向你当前这份源码目录

所以你修改仓库里的源码后，通常：

- 不需要重新安装
- 再运行脚本时就会直接使用你改过的版本

这就是为什么它特别适合：

- 正在开发的项目
- 研究代码
- 需要频繁改源码和 YAML 的仓库

### 12.3 为什么这个项目建议用它

这个仓库不是单纯拿来调用的第三方库，而是你要本地修改、调试、训练的项目。

你现在已经在动这些内容：

- `predict.py`
- `model/*.yaml`
- `ultralytics/*`

如果不用 `pip install -e .`，那很容易出现：

- 你改了源码
- 但 Python 运行时没有使用你当前改的这份代码

所以这里推荐它的根本原因是：

- 让当前环境始终使用你工作区里的这份源码

### 12.4 它和 `pip install .` 的区别

`pip install .`

- 普通安装
- 更像把当前项目“打包后装进去”
- 改完源码后，不一定立即反映到环境里

`pip install -e .`

- 开发模式安装
- 更像给环境和当前源码目录建立连接
- 你改源码后，通常立即生效

所以对当前仓库：

- 开发、调试、训练阶段用 `pip install -e .`
- 只有在你做正式打包发布时，才更可能考虑 `pip install .`

### 12.5 这个命令具体会读取哪里

它会读取当前仓库里的：

- [pyproject.toml](/g:/Code_Repository/YOLOv8_dual_Stream/pyproject.toml)

然后根据其中定义：

- 安装当前项目
- 安装声明过的依赖
- 注册命令入口，比如 `yolo`

所以它不是只做一件事，而是同时完成：

- 项目注册
- 依赖安装
- 环境关联

### 12.6 对这个仓库来说它最重要的作用

这个仓库最关键的一点是：

- 你本地可能已经装过官方 `ultralytics`
- 但这个仓库里的 `ultralytics` 已经被改成双流版本了

如果不做：

```bash
pip install -e .
```

就可能出现：

- 你以为在运行当前仓库的双流代码
- 实际导入的是系统里原来装的官方单流 `ultralytics`

那最终结果就是：

- 双流链路不生效
- 你看到的行为和仓库源码对不上

所以这个命令对当前项目不是可有可无，而是非常关键。

### 12.7 安装完成后应该怎么验证

执行完后，建议立刻跑：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
```

第一条最重要。

如果输出类似：

```text
G:\Code_Repository\YOLOv8_dual_Stream\ultralytics\__init__.py
```

才说明当前环境真正导入的是你这个仓库里的源码。

如果输出指向别的 Python 安装目录下的 `site-packages`，那说明你没有成功用上当前仓库这份代码。

### 12.8 什么时候要重新执行一次

一般改普通 `.py` 文件时，不需要重复执行。

但如果你改了下面这些内容，建议重新跑一次：

- `pyproject.toml`
- 依赖列表
- 包结构
- 命令行入口

例如你新增了某个依赖，就可以再执行一次：

```bash
pip install -e .
```

### 12.9 一句话记忆

你可以直接记成：

- `pip install .`：把项目装进去
- `pip install -e .`：让环境直接使用我当前这份源码

对这个双流项目，应该优先用第二种。
