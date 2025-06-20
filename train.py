from ultralytics import YOLO
import cv2 as cv

if __name__ == '__main__':
    model = YOLO('model/yolov8-seg.yaml')  # build a new model from YAML
    """
    更多详细配置在   ultralytics/yolo/cfg/default.yaml
    """
    model.train(data="dataset/M3FD.yaml", epochs=10, imgsz=640, device=0, batch=4, amp=False,
                project='runs/segment/',
                name='M3FD-V8MADD',
                # 是否覆盖原来的目录
                exist_ok=True,
                # 非确定性
                deterministic=False)

    # 推理
    # model = YOLO(r'D:\project\Python\ultralytics_two_stream\weights\add_jet_0.901.pt')  # load a custom model
    #
    # # Predict with the model
    # results = model([r'D:\project\Python\ultralytics_two_stream\catch_result\论文\color.png',r'D:\project\Python\ultralytics_two_stream\catch_result\论文\depth.png'])  # predict on an image*
    # # 保存结果图
    # res_plotted = results[0].plot(labels=False)
    # # 保存至photo文件夹下
    # cv.imwrite("result.png", res_plotted)