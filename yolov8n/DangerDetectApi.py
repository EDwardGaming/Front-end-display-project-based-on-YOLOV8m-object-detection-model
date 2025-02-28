from shapely.geometry import box
import cv2
from ultralytics import YOLO
from typing import Tuple
import numpy as np

class DangerDetector:
    """路面危险评估API核心类"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        初始化检测器
        :param model_path: YOLOv8模型文件路径
        :param conf_threshold: 置信度阈值 (默认0.5)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_map = {0: "water", 1: "snow", 2: "ice"}
        self.priority = {"ice": 3, "snow": 2, "water": 1}
        self.messages = {
            "ice": "⚠️ 路面存在结冰区域，请保持车距并使用防滑链！",
            "snow": "❄️ 路面存在积雪，请保持车距！",
            "water": "💧 路面湿滑，建议减速慢行！",
            "safe": "✅ 路面状况安全，可正常行驶"
        }

    def _calculate_danger_level(self, results, image_width: int, image_height: int) -> Tuple[str, str]:
        """内部危险评估算法"""
        detections = []

        if results.boxes:
            boxes = results.boxes.cpu().numpy()
            for i in range(len(boxes.xyxy)):
                x1, y1, x2, y2 = boxes.xyxy[i]
                cls_id = int(boxes.cls[i])
                detections.append((self.class_map[cls_id], x1, y1, x2, y2))

        # 按优先级排序
        sorted_detections = sorted(detections, key=lambda x: -self.priority.get(x[0], 0))

        # 计算覆盖面积
        covered_areas = []
        ice_area = snow_area = water_area = 0

        for detection in sorted_detections:
            cls, x1, y1, x2, y2 = detection
            current_box = box(x1, y1, x2, y2)

            # 检查重叠
            if not any(current_box.intersects(existing["geometry"]) for existing in covered_areas):
                area = current_box.area
                if cls == "ice":
                    ice_area += area
                elif cls == "snow":
                    snow_area += area
                elif cls == "water":
                    water_area += area
                covered_areas.append({"class": cls, "geometry": current_box})
        # 计算危险值
        total_pixels = image_width * image_height
        danger_value = (ice_area + snow_area * 0.8 + water_area * 0.6) / total_pixels

        # 判定危险等级
        if danger_value >= 0.8:
            danger_level = "高度危险"
        elif danger_value >= 0.5:
            danger_level = "中度危险"
        elif danger_value >= 0.3:
            danger_level = "轻度危险"
        else:
            danger_level = "安全"

        # 生成提示信息
        max_class = max(["ice", "snow", "water"],
                        key=lambda x: ice_area if x == "ice" else snow_area if x == "snow" else water_area)
        message = self.messages[max_class] if danger_level != "安全" else self.messages["safe"]

        return danger_level, message

    def predict_from_image(self, image_path: str) -> dict:
        """
        从图片文件路径进行预测
        :param image_path: 图片文件路径
        :return: 包含危险等级和提示信息的字典
        """
        results = self.model.predict(
            source=image_path,
            save=True,
            conf=self.conf_threshold
        )

        # 获取图片尺寸
        img = results[0].orig_img
        h, w = img.shape[:2]

        # 计算危险等级
        danger_level, message = self._calculate_danger_level(results[0], w, h)

        return {
            "danger_level": danger_level,
            "message": message,
            "image_size": (w, h),
            "results": results
        }

    def predict_from_array(self, image_array : np.ndarray) -> dict:
        """
        从numpy数组进行预测
        :param image_array: 输入图像数组 (HWC格式)
        :return: 包含危险等级和提示信息的字典
        """
        results = self.model.predict(
            source=image_array,
            save=False,
            conf=self.conf_threshold
        )

        # 获取图片尺寸
        h, w = image_array.shape[:2]

        # 计算危险等级
        danger_level, message = self._calculate_danger_level(results[0], w, h)

        return {
            "danger_level": danger_level,
            "message": message,
            "image_size": (w, h),
            "results": results
        }


# 示例用法
if __name__ == "__main__":
    # 初始化检测器
    detector = DangerDetector(
        model_path=r"D:\Program Files\road_snow\yolov8n\runs\detect\train310\weights\best.pt",
        conf_threshold=0.5
    )

    # 示例预测
    test_image = input("输入图片路径: ")
    result = detector.predict_from_image(test_image)
    print(f"危险等级: {result['danger_level']}")
    print(f"提示信息: {result['message']}")