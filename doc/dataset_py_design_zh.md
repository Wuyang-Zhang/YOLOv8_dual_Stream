# `ultralytics/data/dataset.py` 说明

## 1. 这个文件解决了什么问题

这个仓库原先缺少 [ultralytics/data/dataset.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/data/dataset.py)，导致两类链路都会直接断掉：

- 预测阶段很多 import 会因为缺少 `YOLODataset` 而失败
- 训练阶段真正需要的数据集构建入口 `build_yolo_dataset()` 没有落点

我补的这个文件，不是为了完全还原官方 Ultralytics 全部数据集能力，而是先把当前双流仓库最关键的训练接口补齐，让 `detect/segment` 训练链能拿到正确 batch。

---

## 2. 设计目标

这版 `dataset.py` 的设计目标是：

- 对齐当前仓库已经改过的双流接口
- 兼容 `train.py -> build_yolo_dataset() -> YOLODataset`
- 输出当前 `trainer/validator` 真正消费的字段
- 尽量少碰无关功能，比如 grounding、semantic

它优先服务的是这条链：

1. [train.py](/g:/Code_Repository/YOLOv8_dual_Stream/train.py)
2. [ultralytics/engine/trainer.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/engine/trainer.py)
3. [ultralytics/models/yolo/detect/train.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/models/yolo/detect/train.py)
4. [ultralytics/data/build.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/data/build.py)
5. [ultralytics/data/dataset.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/data/dataset.py)

---

## 3. 这个实现的核心思路

### 3.1 双流输入

当前仓库不是单图数据集，而是双流：

- 一路 `RGB`
- 一路 `Depth/IR`

所以 `YOLODataset` 构造时要求两套路径：

- `img_path_rgb`
- `img_path_depth`

如果调用方只给了 `img_path`，实现里会把它同时映射给两路，主要是为了兼容一些还带着上游单流调用习惯的地方。

### 3.2 配对规则

RGB 和第二路图像通过文件名 stem 配对，也就是：

- `rgb/000123.jpg`
- `ir/000123.jpg`

会被视为同一个样本。

对应逻辑在 `YOLODataset._pair_modal_files()`：

- 先检查 RGB 和第二路数量是否一致
- 再把第二路文件按 stem 建索引
- 最后按照 RGB 的顺序重排 depth/IR 列表

这样可以保证后续同一个 `index` 下：

- `self.im_files_rgb[index]`
- `self.im_files_depth[index]`

一定是配对的。

### 3.3 标签来源

当前实现默认标签按 RGB 路径去找，也就是沿用 YOLO 常规逻辑：

- `images/.../xxx.jpg`
- `labels/.../xxx.txt`

这里调用的是 [ultralytics/data/utils.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/data/utils.py) 里的：

- `img2label_paths()`
- `verify_image_label()`

也就是说，这版 `dataset.py` 没有自造标签格式，而是直接沿用仓库里现有的 YOLO 标签解析接口。

---

## 4. `YOLODataset` 做了哪些事情

### 4.1 构造阶段

构造函数主要做四件事：

1. 解析兼容参数
2. 记录任务类型
3. 调用 [BaseDataset](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/data/base.py)
4. 建立单流兼容别名

其中任务类型会影响后面输出什么：

- `task == "detect"`：只需要框
- `task == "segment"`：需要分割 mask 对应的 polygon
- `task == "pose"`：需要 keypoints
- `task == "obb"`：需要面向旋转框的 segment 表达

### 4.2 `get_labels()`

这个函数负责把磁盘上的标签变成内部标签列表。

每个样本最终会整理成一个字典，核心字段包括：

- `im_file`
- `shape`
- `cls`
- `bboxes`
- `segments`
- `keypoints`
- `normalized`
- `bbox_format`

同时这里还做了一件很关键的事：

- 如果某些图片/标签被 `verify_image_label()` 判定为坏样本，就把 `im_files_rgb` 和 `im_files_depth` 也同步过滤

这是为了避免后面出现“标签列表长度变短了，但图像列表没变”的索引错位问题。

### 4.3 `update_labels_info()`

这个函数负责把原始标签转成 [Instances](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/utils/instance.py)。

原因是当前增强链和格式化链基本都围绕 `Instances` 工作，比如：

- 随机仿射
- 翻转
- mosaic
- mask 生成

如果是 `segment/obb` 任务，它还会把 polygon 通过 `resample_segments()` 统一采样到固定长度，形成：

- `N x 1000 x 2`

这是为了兼容后续增强和 mask 处理代码。

### 4.4 `build_transforms()`

这个函数决定 train 和 val 分别走什么增强。

训练阶段：

- 走 `v8_transforms(...)`
- 包括 mosaic、mixup、随机仿射、翻转等

验证阶段：

- 只做 `LetterBox`
- 再接 `Format`

最后都会拼上 `Format(...)`，把样本整理成训练器真正需要的张量结构。

### 4.5 `collate_fn()`

这是当前双流实现里一个很关键的地方。

它没有在 dataloader collate 阶段直接把图像 stack 成一个大 tensor，而是：

- `img_rgb` 保持为列表
- `img_depth` 保持为列表

原因是当前 [DetectionTrainer](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/models/yolo/detect/train.py) 和 [DetectionValidator](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/models/yolo/detect/val.py) 会在 `preprocess()` 里自己做：

- `torch.stack(batch["img_rgb"], dim=0)`
- `torch.stack(batch["img_depth"], dim=0)`

这样可以保留后续多尺度 resize 的处理空间。

而下面这些字段会在 collate 阶段直接拼接：

- `cls`
- `bboxes`
- `masks`
- `keypoints`
- `batch_idx`

其中 `batch_idx` 会按样本编号做偏移，确保 loss 计算时还能知道每个目标来自哪个样本。

---

## 5. 这个文件里还补了哪些兼容类

除了 `YOLODataset`，还补了几个兼容类：

- `YOLOMultiModalDataset`
  - 只是 `YOLODataset` 的别名，用来兼容 builder

- `YOLOConcatDataset`
  - 继承 `ConcatDataset`
  - 复用了 `YOLODataset.collate_fn`

- `ClassificationDataset`
  - 做了一个最小可用实现
  - 主要是防止分类路径 import 时报错
  - 它不是真正双流分类，只是把 `depth` 占位成 RGB

- `GroundingDataset`
- `SemanticDataset`
  - 目前显式抛 `NotImplementedError`
  - 因为这两个不是你当前训练双流检测/分割的关键路径

---

## 6. 我顺手修掉的配套问题

单独补 `dataset.py` 还不够，所以我同时修了两个配套问题。

### 6.1 [ultralytics/data/base.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/data/base.py)

这个文件之前已经被改成双流了，但缓存逻辑里还残留很多单流字段：

- `self.npy_files`
- `self.ims`
- `self.im_files`

这些在开启 `cache` 时会直接出错。

我把这些地方改成了双流版本：

- `npy_files_rgb / npy_files_depth`
- `ims_rgb / ims_depth`
- `im_files_rgb / im_files_depth`

### 6.2 [ultralytics/models/yolo/detect/val.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/models/yolo/detect/val.py)

这个文件原来写死了一个作者本地路径：

- `D:\project\datasets\YOLO-10000-18\valid\images`

这会导致你即使数据集 YAML 配好了，验证阶段还是跑不到自己的数据。

现在已经改成从 `data.yaml` 里读：

- `val_rgb`
- `val_depth`

如果 `split` 不是 `val`，也会优先读对应的：

- `{split}_rgb`
- `{split}_depth`

---

## 7. 当前这版的边界

这版 `dataset.py` 现在适合你当前目标：

- 训练双流 detection
- 训练双流 segmentation

但它不是全量恢复版，边界也很明确：

- `grounding` 没实现
- `semantic` 没实现
- `classification` 只是最小兼容
- Explorer 这类上游单流工具链没有做完整适配

也就是说，它是“先把你当前训练主链跑通”的工程化补丁，而不是“把整个 Ultralytics 所有数据路径完整重建”。

---

## 8. 你接下来该怎么用

你现在要重点检查三件事：

1. `dataset/M3FD.yaml` 里的路径是否真实存在
2. RGB 和 IR/Depth 文件名是否一一对应
3. 标签格式是否和模型任务一致

如果你训练检测模型：

- 每个标签文件应该是 5 列 YOLO 框标注

如果你训练分割模型：

- 每个标签文件应该是 polygon 分割标注，而不只是 5 列框

---

## 9. 建议的最小验证命令

先验证数据集类能不能 import：

```bash
python -c "from ultralytics.data.dataset import YOLODataset; print('dataset ok')"
```

再跑训练入口：

```bash
python train.py
```

如果还报错，下一层最常见的问题通常是：

- 环境缺依赖
- 数据路径写错
- RGB/IR 文件名配对不上
- `seg` 模型用了检测框标签