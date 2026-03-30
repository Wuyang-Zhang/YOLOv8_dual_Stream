# 双流预测链路说明

## 1. 先说结论

这个仓库的双流预测是否真正生效，要同时满足两件事：

1. 推理代码层已经支持把两张图送进模型
2. 你加载的模型 YAML 本身真的定义了第二分支和融合层

两者缺一不可。

## 2. 双流入口在哪里

### 2.1 入口脚本

预测入口是 [predict.py](/g:/Code_Repository/YOLOv8_dual_Stream/predict.py)。

它做的事情很简单：

1. 读取命令行参数
2. 加载模型
3. 把两张图作为一个列表传给 `YOLO(...)`
4. 取 `results[0]` 画图并保存

当前脚本默认模型已经改成：

- [model/yolov8m-seg-2stream-ref-postfusion.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-ref-postfusion.yaml)

这是一份真正带双流 backbone 和融合层的模型结构文件。

## 3. 调用链路

实际的调用顺序是：

1. [predict.py](/g:/Code_Repository/YOLOv8_dual_Stream/predict.py)
2. [ultralytics/models/yolo/model.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/models/yolo/model.py)
3. [ultralytics/engine/model.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/engine/model.py)
4. [ultralytics/engine/predictor.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/engine/predictor.py)
5. [ultralytics/nn/autobackend.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/autobackend.py)
6. [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py)
7. 对应任务 predictor 的后处理
8. `results[0].plot()` 输出可视化结果

## 4. 双流输入是怎么拆开的

双流拆分发生在 [ultralytics/nn/autobackend.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/autobackend.py) 的 `AutoBackend.forward()` 里。

关键位置：

- [ultralytics/nn/autobackend.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/autobackend.py#L442)
- [ultralytics/nn/autobackend.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/autobackend.py#L443)
- [ultralytics/nn/autobackend.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/autobackend.py#L444)
- [ultralytics/nn/autobackend.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/autobackend.py#L460)

逻辑是：

- 第一张图取出来作为 `im`
- 第二张图取出来作为 `im_depth`
- 然后调用底层模型：
  - `self.model([im, im_depth], ...)`

也就是说，推理输入层面，这个仓库已经明确按“两张图一组”的方式处理了。

## 5. 第二路输入是怎么进入网络的

真正让第二路图像进入网络的是 [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py)。

关键位置：

- [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py#L104)
- [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py#L116)
- [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py#L136)
- [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py#L156)
- [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py#L157)

逻辑是：

1. `forward(x)` 不再把输入当一张图，而是把：
   - `x[0]` 当第一路
   - `x[1]` 当第二路
2. 然后进入：
   - `predict(x, x2)`
   - `_predict_once(x, x2)`
3. 在 `_predict_once()` 里：
   - 普通层继续走第一路 `x`
   - 当层的 `from` 标记为 `-4` 时，改为从第二路 `x2` 取特征

这就是这个仓库“第二分支入口”的核心机制。

## 6. `-4` 代表什么

在这个仓库里，`-4` 不是普通 Ultralytics 单流模型里常见的层索引，它被作者拿来表示：

- 这一层的输入来自第二路图像

对应逻辑就在：

- [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py#L156)
- [ultralytics/nn/tasks.py](/g:/Code_Repository/YOLOv8_dual_Stream/ultralytics/nn/tasks.py#L157)

所以你以后读 YAML 时，看到：

```yaml
- [-4, 1, Focus, [64, 3]]
```

就应该理解成：

- 这不是继续接上一层的 RGB 特征
- 而是从第二路输入重新起一个分支

## 7. 为什么不是所有 YAML 都是双流

这是最容易混淆的地方。

代码虽然已经支持双流，但模型结构文件不一定是双流。

### 7.1 单流示例

[model/yolov8-seg.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8-seg.yaml) 是单流结构。

你在这个文件里看不到：

- `-4`
- `Concat3`
- `ResidualExciteFusion`
- `Add2`
- `GPT`

所以它虽然能被当前代码加载，但本身不会真正使用第二路图像。

### 7.2 双流示例

[model/yolov8m-seg-2stream-ref-postfusion.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-ref-postfusion.yaml) 是双流结构。

典型位置：

- [model/yolov8m-seg-2stream-ref-postfusion.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-ref-postfusion.yaml#L29)
  - 第二路 backbone 从这里开始
- [model/yolov8m-seg-2stream-ref-postfusion.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-ref-postfusion.yaml#L41)
- [model/yolov8m-seg-2stream-ref-postfusion.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-ref-postfusion.yaml#L42)
- [model/yolov8m-seg-2stream-ref-postfusion.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-ref-postfusion.yaml#L43)
  - 这里是两路特征的融合位置

另一个双流例子是：

- [model/yolov8m-seg-2stream-SLBAFNET.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-SLBAFNET.yaml#L19)
- [model/yolov8m-seg-2stream-SLBAFNET.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-SLBAFNET.yaml#L21)

其中：

- `-4` 表示第二路入口
- `Concat3` 表示两路特征融合

## 8. 预测阶段的数据组织方式

当前推理逻辑默认是：

- 一组输入由两张图组成
- 第一张是 RGB
- 第二张是 depth 或 IR

也就是说，调用方式应类似：

```python
results = model([rgb_path, aux_path])
```

或者命令行：

```bash
python predict.py --rgb path/to/rgb.jpg --aux path/to/ir_or_depth.jpg
```

## 9. 当前默认预测脚本用的是什么模型

现在 [predict.py](/g:/Code_Repository/YOLOv8_dual_Stream/predict.py) 的默认模型是：

- [model/yolov8m-seg-2stream-ref-postfusion.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8m-seg-2stream-ref-postfusion.yaml)

这样做的目的就是避免再误用单流的 [model/yolov8-seg.yaml](/g:/Code_Repository/YOLOv8_dual_Stream/model/yolov8-seg.yaml)。

## 10. 现在这条双流预测链能做什么

在没有训练权重的情况下，它可以用于：

- 验证双流输入是否能成功进入模型
- 验证 predictor 到 model 的调用链是否打通
- 验证后处理和画图逻辑是否正常

但它不能用于：

- 判断模型效果
- 判断类别准确率
- 判断双流融合质量

因为没有训练权重时，网络参数基本是随机的。

## 11. 一句话记忆版

记住下面这句就够了：

- `autobackend.py` 负责“把两张图拆成两路”
- `tasks.py` 负责“让第二路真的进网络”
- `model/*.yaml` 决定“第二路到底有没有被真正用上”
