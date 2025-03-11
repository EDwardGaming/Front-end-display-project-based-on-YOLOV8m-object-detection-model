from DangerDetectApi import DangerDetector

model_path = r"E:\education\road_snow\yolov8n\runs\detect\train3\weights\best.pt" # 模型路径
conf_threshold = 0.5 # 置信度阈值

# 初始化检测器
detector = DangerDetector(model_path,conf_threshold)

# 从文件路径预测
result = detector.predict_from_image(r"E:\education\dataset\json_dataset\160\IMG_2179.png")
print(result['danger_level'], result['message'])