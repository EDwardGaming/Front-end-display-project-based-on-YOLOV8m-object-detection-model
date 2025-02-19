from DangerDetectApi import DangerDetector

# 初始化检测器
detector = DangerDetector(
    model_path=r"D:\Program Files\road_snow\yolov8n\runs\detect\train310\weights\best.pt",
    conf_threshold=0.5
)

# 从文件路径预测
result = detector.predict_from_image(r"D:\dataset\pure_dataset\冰\IMG_5893.JPG")
print(result['danger_level'], result['message'])
