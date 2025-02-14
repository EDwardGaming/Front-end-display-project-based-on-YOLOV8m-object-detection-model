from shapely.geometry import box
import cv2
from ultralytics import YOLO

# ======================== 核心算法 ========================
def calculate_danger_level(results, image_width, image_height):
    """
    :param results: YOLOv8的预测结果对象（单张图片）
    :param image_width: 图片宽度
    :param image_height: 图片高度
    :return: danger_level (str), message (str)
    """
    # 类别映射（根据训练时的数据集顺序）
    class_map = {0: "water", 1: "snow", 2: "ice"}
    
    # 从YOLO结果中提取检测框数据（修正后的数据提取方式）
    detections = []
    if results.boxes:
        for result in results:  # 遍历所有检测框
            xyxys = result.boxes.xyxy  # 边界框坐标
            scores = result.boxes.conf  # 置信度分数
            classes_ids = result.boxes.cls  # 类别索引
    
            for xyxy, score, class_id in zip(xyxys, scores, classes_ids):
                x1, y1, x2, y2 = xyxy
                detections.append((class_map[int(class_id)], x1, y1, x2, y2))

    # 优先级映射（ice > snow > water）
    priority = {"ice": 3, "snow": 2, "water": 1}
    total_pixels = image_width * image_height

    # 按优先级排序检测框
    sorted_detections = sorted(detections, key=lambda x: -priority.get(x[0], 0))

    # 初始化覆盖区域（使用Shapely处理几何图形）
    covered_areas = []
    ice_area = snow_area = water_area = 0

    for detection in sorted_detections:
        cls, x1, y1, x2, y2 = detection
        current_box = box(x1, y1, x2, y2)

        # 检查与已覆盖区域的重叠
        overlap = False
        for existing in covered_areas:
            if current_box.intersects(existing["geometry"]):
                overlap = True
                break

        if not overlap:
            # 计算面积并记录
            area = current_box.area
            if cls == "ice":
                ice_area += area
            elif cls == "snow":
                snow_area += area
            elif cls == "water":
                water_area += area
            covered_areas.append({"class": cls, "geometry": current_box})

    # 计算危险值（带权重）
    danger_value = (ice_area + snow_area*0.8 + water_area*0.6) / total_pixels

    # 危险等级划分（修正阈值逻辑）
    if danger_value >= 0.8:
        danger_level = "高度危险"
    elif danger_value >= 0.5:
        danger_level = "中度危险"
    elif danger_value >= 0.3:
        danger_level = "轻度危险"
    else:
        danger_level = "安全"
    
    # 生成提示信息
    max_class = max(["ice", "snow", "water"], key=lambda x: ice_area if x=="ice" else snow_area if x=="snow" else water_area)
    
    messages = {
        "ice": "⚠️ 路面存在结冰区域，请保持车距并使用防滑链！",
        "snow": "❄️ 路面存在积雪，请保持车距！",
        "water": "💧 路面湿滑，建议减速慢行！"
    }
    message = messages[max_class] if danger_level != "安全" else "✅ 路面状况安全，可正常行驶"

    return danger_level, message

# ======================== 主程序 ========================
if __name__ == "__main__":
    
        model = YOLO(r'D:\Program Files\road_snow\yolov8n\runs\detect\train310\weights\best.pt')
        img_path = input("输入图片路径: ")
        results = model.predict(
            source=img_path,
            save=True,
            conf=0.5  # 置信度阈值
        )
        
        # 获取原始图片尺寸
        img = results[0].orig_img
        h, w = img.shape[:2]
        
        # 调用危险评估算法
        danger_level, message = calculate_danger_level(results[0], w, h)
        print(f"危险等级: {danger_level}\n提示信息: {message}")
        
