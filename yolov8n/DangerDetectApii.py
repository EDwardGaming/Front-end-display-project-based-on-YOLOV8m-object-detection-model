from shapely.geometry import box
import cv2
from ultralytics import YOLO
from typing import Tuple
import numpy as np

class DangerDetector:
    """路面危险评估API核心类(精确面积计算版)"""




    def __init__(self, model_path: str, conf_threshold: float = 0.5):
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
        """优化后的危险评估算法"""
        detections = []
        if results.boxes:
            boxes = results.boxes.cpu().numpy()
            for i in range(len(boxes.xyxy)):
                x1, y1, x2, y2 = boxes.xyxy[i]
                conf = boxes.conf[i]
                cls_id = int(boxes.cls[i])
                area = (x2 - x1) * (y2 - y1)
                detections.append({
                    "class": self.class_map[cls_id],
                    "coords": (x1, y1, x2, y2),
                    "confidence": conf,
                    "area": area
                })

        # 优化排序逻辑
        sorted_detections = sorted(
            detections,
            key=lambda x: (
                -self.priority.get(x["class"], 0),
                -x["area"],
                -x["confidence"]
            )
        )

        # 精确面积计算逻辑
        covered_areas = []
        ice_area = snow_area = water_area = 0

        for detection in sorted_detections:
            cls = detection["class"]
            x1, y1, x2, y2 = detection["coords"]
            current_box = box(x1, y1, x2, y2)
            current_polygon = current_box

            # 计算有效未覆盖区域
            valid_area = current_polygon
            for existing in covered_areas:
                if valid_area.intersects(existing["geometry"]):
                    valid_area = valid_area.difference(existing["geometry"])
            
            effective_area = valid_area.area

            if effective_area > 0:
                # 扣除被覆盖区域的面积
                for existing in covered_areas:
                    if current_polygon.intersects(existing["geometry"]):
                        overlap = existing["geometry"].intersection(current_polygon)
                        if overlap.area > 0:
                            if existing["class"] == "ice":
                                ice_area -= overlap.area
                            elif existing["class"] == "snow":
                                snow_area -= overlap.area
                            elif existing["class"] == "water":
                                water_area -= overlap.area

                # 添加新区域面积
                if cls == "ice":
                    ice_area += effective_area
                elif cls == "snow":
                    snow_area += effective_area
                elif cls == "water":
                    water_area += effective_area

                # 合并覆盖区域
                new_geometry = current_polygon
                for existing in covered_areas:
                    if new_geometry.intersects(existing["geometry"]):
                        new_geometry = new_geometry.union(existing["geometry"])
                covered_areas.append({
                    "class": cls,
                    "geometry": new_geometry,
                    "confidence": detection["confidence"]
                })

        # 后续逻辑保持不变
        total_pixels = image_width * image_height
        weighted_ice = ice_area 
        weighted_snow = snow_area 
        weighted_water = water_area 
        danger_value = (weighted_ice + weighted_snow + weighted_water) / total_pixels
        
        thresholds = {
            "high": 0.7 ,
            "medium": 0.4 ,
            "low": 0.2 
        }

        if danger_value >= thresholds["high"]:
            danger_level = "高度危险"
        elif danger_value >= thresholds["medium"]:
            danger_level = "中度危险"
        elif danger_value >= thresholds["low"]:
            danger_level = "轻度危险"
        else:
            danger_level = "安全"

        dominant_class = max(
            ["ice", "snow", "water"],
            key=lambda x: (ice_area, snow_area, water_area)[["ice", "snow", "water"].index(x)]
        )
        message = self.messages[dominant_class] if danger_level != "安全" else self.messages["safe"]
        
        return danger_level, message

    # 保持原有predict方法不变
    
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
