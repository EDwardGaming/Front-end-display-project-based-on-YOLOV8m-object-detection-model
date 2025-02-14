import cv2
from ultralytics import YOLO
import numpy as np
# 加载模型
model = YOLO(r'D:\Program Files\road_snow\yolov8n\runs\detect\train310\weights\best.pt')
 
# 进行预测
results = model.predict(r"D:\dataset\PNG\12.02\IMG_0723.PNG")
 
# 提取检测结果
for result in results:
    boxes = result.boxes.xyxy  # 边界框坐标
    scores = result.boxes.conf  # 置信度分数
    classes = result.boxes.cls  # 类别索引
    
    # 如果有类别名称，可以通过类别索引获取
    class_names = [model.names[int(cls)] for cls in classes]
    
    # 打印检测结果
    for box, score, class_name in zip(boxes, scores, class_names):
        print(f"Class: {class_name}, Score: {score:.2f}, Box: {box.cpu().tolist()}")
        
    # 可视化检测结果
    annotated_img = result.plot()
    
    # 显示图像
    cv2.imshow('Detected Image', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()