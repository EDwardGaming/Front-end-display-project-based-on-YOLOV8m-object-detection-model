from DangerDetectApi import DangerDetector

model_path = r"E:\education\road_snow\yolov11m\runs\detect\train38\weights\best.pt" # 模型路径
conf_threshold = 0.5 # 置信度阈值

# 初始化检测器
detector = DangerDetector(model_path,conf_threshold)

# 从文件路径预测
while True:
    # 输入图片路径
    image_path = input("请输入图片路径（输入exit或按ctrl + c退出）：")
    if image_path.lower() == "exit":
        break
    
    image_path = image_path.replace("\\", "/").replace("\"","")  # 替换反斜杠为正斜杠
    # 进行预测
    result = detector.predict_from_image(image_path)
    print(result['danger_level'], result['message'])