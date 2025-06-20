from catch.dataset_utils import depth_rendered_by_realsense
from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # depth_rendered_by_realsense(r'D:\project\Python\ultralytics_two_stream\catch_result\论文\depth.png', r'D:\project\Python\ultralytics_two_stream\catch_result\论文\jet.png')
    # 加载模型
    model = YOLO(r"D:\project\Python\ultralytics_two_stream\weights\SLBAF.pt")
    results = model([r'C:\Users\ASUS\Desktop\论文\LLVIP/190205.jpg',
                     r'C:\Users\ASUS\Desktop\论文\LLVIP/190205-ir.jpg'])
    # 保存结果图，结果图参数：https://docs.ultralytics.com/modes/predict/?h=plot%28%29#plot-method-parameters
    res_plotted = results[0].plot(labels=False)
    # 保存至photo文件夹下
    cv2.imwrite(r"D:\project\Python\ultralytics_two_stream\catch_result\photo\result.png", res_plotted)
