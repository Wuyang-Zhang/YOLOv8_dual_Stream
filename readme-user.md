预测：

```
python predict.py --model  model/yolov8m-seg-2stream-ref-postfusion.yaml --rgb dataset/test/images/000018_png.rf.2193e37a18154e646b0e64b85e5640ef.jpg --aux dataset/test/depth_jet/000018_png.rf.2193e37a18154e646b0e64b85e5640ef.jpg --output runs/predict/result.png

```


```
0: 480x640 (no detections), 239.6msSpeed: 7.0ms preprocess, 239.6ms inference, 6.0ms postprocess per image at shape (1, 3, 480, 640)
Saved prediction result to: G:\Code_Repository\YOLOv8_dual_Stream\runs\predict\result.png
```
