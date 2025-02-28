from DangerDetectApii import DangerDetector

model_path = r"D:\Program Files\road_snow\yolov8n\runs\detect\train38\weights\best.pt" # 模型路径
conf_threshold = 0.5 # 置信度阈值

# 初始化检测器
detector = DangerDetector(model_path,conf_threshold)

# 从文件路径预测
result = detector.predict_from_image(r"D:\dataset\json_dataset\159\IMG_2119.png")
print(result['danger_level'], result['message'])