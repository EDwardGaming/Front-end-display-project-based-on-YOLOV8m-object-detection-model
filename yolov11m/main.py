from DangerDetectApi import DangerDetector

model_path = r"E:\education\road_snow\yolov11m\runs\detect\train38\weights\best.pt" # 模型路径
conf_threshold = 0.5 # 置信度阈值

# 初始化检测器
detector = DangerDetector(model_path,conf_threshold)

# 从文件路径预测
result = detector.predict_from_image(r"E:\education\dataset\text_dataset\测试集_自己拍的图片\ce5d5374c7789e445506ca01511f09d3.mp4")
print(result['danger_level'], result['message'])