---
title: 基于yolov8m物体检测模型的前端展示项目
date: 2025-02-04 22:31:07
tags:
  - 计算机视觉
  - 机器学习
  - Yolo
  - python
categories:
  - 人工智能
comments: true
---
# 基于yolov8m物体检测模型的前端展示项目

## 大纲

- 基于yolov8m和机器视觉的冰雪检测模型
- 选择自动划分数据集为训练集和验证集。

## 环境配置

```bash
pip pip install -r .\requirements.txt
```

##  项目目录配置

```bash
PS D:\Program Files\road_snow> tree
卷 原神 的文件夹 PATH 列表
卷序列号为 0EDC-5FA0
D:.
├─.idea
│  └─inspectionProfiles
├─.vscode
├─training_set
│  ├─images
│  └─labels
└─yolov8n
    ├─runs
    │  └─detect
    │      └─train310
    │          └─weights
    ├─static
    │  ├─css
    │  ├─results
    │  └─uploads
    └─templates
```

## 识别对象分类

训练决定舍弃数据集中背景和干燥这两大类，只选择moist、snow和ice这三类。目的是只为识别冰雪，增强模型识别的针对性。

| 类型 | 危险指数 |
| ---- | -------- |
| snow | 中度危险 |
| ice  | 高度危险 |

后续想增加`water `类型,指代更多的水，就是更危险的路面。

## 数据集介绍

在之前的YoloV5n模型的基础上，引入纯态数据集进行训练。纯态数据集里面包含的都是单一类型图片，就是整张图片区域都是moist、snow或ice中的一种。

### 数据集目录结构

```bash
PS E:\education\dataset> tree
卷 龙版 的文件夹 PATH 列表
卷序列号为 64E0-4EF7
E:.
├─12.01
├─json_dataset #图片和标注图片的json文件
│  ├─12.02
│  ├─157
│  ├─158
│  ├─159
│  ├─160
│  ├─161
│  ├─162
│  ├─163
│  ├─164
│  ├─165
│  ├─166
│  ├─167
│  ├─168
│  ├─169
│  ├─170
│  ├─171
│  ├─172
│  ├─173
│  ├─174
│  ├─175
│  ├─176
│  ├─177
│  ├─178
│  ├─179
│  ├─180
│  ├─181
│  ├─182
│  ├─183
│  ├─184
│  ├─185
│  ├─186
│  ├─187
│  ├─188
│  ├─189
│  ├─190
│  ├─191
│  ├─192
│  ├─193
│  ├─194
│  ├─195
│  ├─196
│  ├─197
│  ├─198
│  ├─199
│  ├─200
│  ├─201
│  ├─202
│  ├─203
│  ├─204
│  ├─205
│  ├─206
│  ├─208
│  └─209
├─pure_dataset 
│  ├─labels
│  ├─冰
│  ├─干燥
│  ├─潮湿
│  ├─积水
│  └─雪
└─text_dataset 
    ├─测试集
    ├─测试集_自己拍的图片 #自己找的一些用于测试的图片
    └─视频测试集 #视频
```

### `yolov8m.yaml`配置

```yaml
path: E:\education\road_snow\training_set

# 显式指定数据集路径
train: E:\education\road_snow\training_set\images  # **预先划分好的训练集**
val: E:\education\road_snow\training_set\images     # **预先划分好的验证集**

nc: 2
names: ['snow', 'ice']

# 优化增强参数
augmentations:
  mosaic: 1.0        # **全程开启mosaic**
  mixup: 0.5         # **增强混合比例**
  copy_paste: 0.5    # **提升小目标复制概率**
  scale: [0.5, 1.5]  # **新增随机缩放**
  shear: 0.3         # **新增剪切变换**
  perspective: 0.001 # **新增透视变换**

# 优化检测参数
conf_thres: 0.25     # **平衡召回与精度**
iou_thres: 0.45      # **适应雪地目标重叠**
```



## 数据集标注

### 标注纯态数据集

#### 需求

- 需要为Yolo机器学习标注数据集，有以下数据集：moist 、snow和ice三种。需要实现数据集标注。返回图片对应的yolo的.txt标注文件。
- 现在有一些纯态图片，就是整张图片区域都是moist、snow或ice中的一种。

- 标注方法的话，纯态代表整张图片，所以对应的`label.txt`就是 `<class_id> 0.5 0.5 1.0 1.0`

- 再把标注过的图片移动到训练集下

  ```bash
  ├─training_set
     ├─images
     └─labels
  ```

#### 代码实现

```python
# 该脚本实现划分未标注的纯态数据集为训练集目录的功能

import os
import shutil
import random

# 定义类别和 class_id
categories = {
    "冰": 2,
    "积水": 0,
    "潮湿": 0,
    "雪": 1
}

# 原始数据目录
base_dirs = {
    "冰": r"D:\Program Files\数据集\纯态\冰",
    "积水": r"D:\Program Files\数据集\纯态\积水",
    "潮湿": r"D:\Program Files\数据集\纯态\潮湿",
    "雪": r"D:\Program Files\数据集\纯态\雪"
}

# 目标存放路径
dataset_dir = r"D:\Program Files\road_snow\training_set" #数据集绝对路径
img_dir = os.path.join(dataset_dir, "images")
label_dir = os.path.join(dataset_dir, "labels")

# 创建目录
for path in [img_dir, label_dir]:
    os.makedirs(path, exist_ok=True)

# 遍历所有类别的图片
for category, dir_path in base_dirs.items():
    class_id = categories[category]
    all_images = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]

    # 处理训练集
    for filename in all_images:
        src_img_path = os.path.join(dir_path, filename)
        dst_img_path = os.path.join(img_dir, filename)

        # 复制图片
        shutil.copy(src_img_path, dst_img_path)

        # 生成标签
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


print("纯态数据集划分完成！")

```

### 标注普通数据集

#### 需求

- 是图片文件和它对应的json标注文件，需要把这些.json标注转为.txt。之后在分别复制到训练集下。

  ```bash
  PS D:\Program Files\数据集\PNG\12.02> ls
  
  
      目录: D:\Program Files\数据集\PNG\12.02
  
  
  Mode                 LastWriteTime         Length Name
  ----                 -------------         ------ ----
  -a----         2025/1/20     17:42        8872155 IMG_0721.json
  -a----         2025/1/20     17:44        6337944 IMG_0721.PNG
  -a----         2025/1/20     17:49        8872155 IMG_0722.json
  -a----         2025/1/20     17:48        6392482 IMG_0722.PNG
  -a----         2025/1/20     17:51        8872155 IMG_0723.json
  -a----         2025/1/20     17:46        6293090 IMG_0723.PNG
  -a----         2025/1/20     17:56        8872155 IMG_0724.json
  ......(省略)
  ```

#### 代码实现

```python
# 该脚本实现划分 将以.json格式标注过的数据集转成.txt格式 并将这些文件构建成训练目录
import os
import json
import random
import shutil

# 设置路径
image_dir = r"D:\数据集\PNG\12.02"  # 原始图片目录
label_output_dir = r"D:\Program Files\road_snow\training_set\labels"
image_output_dir = r"D:\Program Files\road_snow\training_set\images"
os.makedirs(label_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)
suffix_file = [".PNG",".JPG",".JPEG",".jpg",".png",".jpeg"]

# 定义类别
classes = ['water', 'snow', 'ice']

# 解析 JSON 并转换为 YOLO 格式
def convert_to_yolo(json_file, img_width, img_height):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    yolo_labels = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            label = shape['label']
            if label not in classes:
                continue
            class_id = classes.index(label)
            
            # 计算多边形的外接矩形
            points = shape['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 转换为 YOLO 格式
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_labels

# 保存图片和标签
def save_data(img_file, labels):
    base_name = os.path.basename(img_file).replace(".PNG", "").replace(".JPG", "").replace(".jpeg", "").replace(".JPEG", "").replace(".png", "").replace(".jpg", "")
    for i in suffix_file:
        if os.path.basename(img_file).endswith(i):
            img_output_path = os.path.join(image_output_dir, f"{base_name}{i}")
            break
    label_output_path = os.path.join(label_output_dir, f"{base_name}.txt")

    with open(label_output_path, "w") as f:
        f.write("\n".join(labels))

    # 复制图片
    shutil.copy(img_file, img_output_path)

# 获取所有图片文件
image_paths = []
for img_file in os.listdir(image_dir):
    for i in suffix_file:
        if img_file.endswith(i):
            json_file = os.path.join(image_dir, img_file.replace(".PNG", ".json").replace(".JPG", ".json").replace(".jpeg", ".json").replace(".JPEG", ".json").replace(".png", ".json").replace(".jpg", ".json"))
            if os.path.exists(json_file):#如果存在对应的json则操作
                # 假设图片分辨率为 2400x1600，可以根据实际情况调整
                img_width, img_height = 2400, 1600
                labels = convert_to_yolo(json_file, img_width, img_height)
                save_data(os.path.join(image_dir, img_file), labels)
                image_paths.append(os.path.join(image_output_dir, img_file))
        break

'''
# 手动划分训练集和验证集
random.shuffle(image_paths)
split_idx = int(len(image_paths) * 0.8)
train_imgs, val_imgs = image_paths[:split_idx], image_paths[split_idx:]

# 创建训练集和验证集目录
train_dir = os.path.join(image_output_dir, "train")
val_dir = os.path.join(image_output_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 移动文件
for img_file in train_imgs:
    shutil.move(img_file, os.path.join(train_dir, os.path.basename(img_file)))

for img_file in val_imgs:
    shutil.move(img_file, os.path.join(val_dir, os.path.basename(img_file)))
'''
print("数据集划分完成！")

```



## 模型训练

创建训练目录`yolov8m`，在该目录下创建`train.py`文件

```python
import torch
from ultralytics import YOLO
import torch.multiprocessing as mp

# 加载预训练的 YOLOv8 模型
model = YOLO(r"E:\education\road_snow\yolov11m\yolo11m.pt")  # 这里可以根据你的需求替换为其他版本的 YOLO 模型

# 配置训练参数
train_args = {
    'data': 'yolov11m.yaml',  # 数据集配置文件路径
    'epochs': 100,  # 训练的轮数
    'batch': 24,  # 每个批次的大小，适当调整（16到32之间）
    'imgsz': 640,  # 输入图像的大小
    'device': '0',  # 使用 GPU 训练，如果没有 GPU 可以改为 'cpu'
    'project': 'runs/detect',  # 存储训练结果的目录
    'name': 'train3',  # 训练结果保存的子目录
    'save': True,  # 是否保存模型
    'save_period': -1,  # 每隔多少轮保存一次模型
    'verbose': True,  # 是否打印详细日志
    'workers': 0,  # 数据加载器的工作线程数
    'optimizer': 'AdamW',  # 这里选择自动，还可以选择 AdamW 优化器
    
    'lr0': 0.001,  # 初始学习率，适当降低初始学习率，增加学习率衰减率。
    'lrf': 0.01,  # 学习率衰减率，更激进，加快收敛。
    
    'warmup_epochs': 3,  # 预热的轮数，适当降低预热时间
    'box': 5.0,  # 训练的框回归损失权重，越低越平衡归框任务。
    'cls': 1.0,  # 类别损失权重，越高越强化分类任务
    'dfl': 2.0,  # 关键点损失权重
    'pose': 12.0,  # 姿态估计损失权重
    'nbs': 64,  # 批次大小（此参数不会影响实际训练，但用于计算资源）
    'weight_decay' : 0.01,  # 权重衰减
    
    'split': "0.8 0.2",  # 训练集和验证集的比例
    
    'warmup_epochs': 3,
    'label_smoothing': 0.05,  # 标签平滑
       
    'close_mosaic': 15, # 最后15epoch关闭mosaic（稳定收敛）
    'augment': True,
    
    'patience': 10,  # 早停策略的耐心值，连续10个epoch验证集性能无提升则停止训练

    'persist': True,  # **保持预处理状态加速训练**
    'amp': True,      # **启用自动混合精度**
}

# 打印训练参数
print(train_args)

# 训练模型
train_results = model.train(**train_args)

# 输出训练结果
print(train_results)
```





## 模型评估与优化

### 效果图展示

![IMG_6163](assets/IMG_6163.jpg)

![IMG_5882](assets/IMG_5882.jpg)

![IMG_4644](assets/IMG_4644.jpg)

![IMG_0814](assets/IMG_0814.jpg)

![IMG_0730](assets/IMG_0730.jpg)

![IMG_0724](assets/IMG_0724.jpg)

![IMG_0666](assets/IMG_0666.jpg)

![IMG_0607](assets/IMG_0607.jpg)

![IMG_0569](assets/IMG_0569.jpg)

## 危险评估算法设计

面向对象设计的，可作为后端文件import导入使用

`DangerDetectApi.py`

```python
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
        self.class_map = {0: "snow", 1: "ice"}
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

```

使用示例：

`main.py`

```python
from DangerDetectApi import DangerDetector

model_path = r"E:\education\road_snow\yolov11m\runs\detect\train38\weights\best.pt" # 模型路径
conf_threshold = 0.5 # 置信度阈值

# 初始化检测器
detector = DangerDetector(model_path,conf_threshold)

# 从文件路径预测
result = detector.predict_from_image(r"E:\education\dataset\text_dataset\测试集_自己拍的图片\ce5d5374c7789e445506ca01511f09d3.mp4")
print(result['danger_level'], result['message'])
```

### 效果图

![image-20250220011613284](assets/image-20250220011613284.png)

## 前端展示代码

### 源代码

`static/css/styles.css`

```css
body {
    font-family: Arial, sans-serif;
    text-align: center;
    padding: 20px;
    background-color: #f4f4f9;
}

.container {
    max-width: 600px;
    margin: auto;
    background: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

h1 {
    margin-bottom: 20px;
}

button {
    margin-top: 10px;
    padding: 10px 20px;
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

button:hover {
    background-color: #0056b3;
}

#result img, #result video {
    margin-top: 20px;
}
```

`templates/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/static/css/styles.css">
    <title>YOLOv11m Object Detection</title>
</head>
<body>
    <div class="container">
        <h1>YOLOv11m Object Detection</h1>
        <form id="upload-form">
            <input type="file" id="file" name="file" accept="image/*,video/*" required>
            <button type="submit">Upload and Predict</button>
        </form>
        <div id="result">
            <h2>Prediction Result:</h2>
            <img id="result-image" src="" alt="Result will be displayed here" style="max-width: 100%; display: none;">
            <video id="result-video" controls style="max-width: 100%; display: none;"></video>
        </div>
    </div>

    <script>
        const form = document.getElementById('upload-form');
        const resultImage = document.getElementById('result-image');
        const resultVideo = document.getElementById('result-video');
    
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('file');
            const file = fileInput.files[0];
    
            if (!file) return alert('Please select a file.');
    
            const formData = new FormData();
            formData.append('file', file);
    
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
    
                const result = await response.json();
    
                if (result.error) {
                    alert(result.error);
                    return;
                }
    
                // 添加时间戳避免缓存问题
                const timestamp = new Date().getTime();
                const resultPath = `${result.result}?t=${timestamp}`;
    
                if (file.type.startsWith('image/')) {
                    resultImage.src = resultPath;
                    resultImage.style.display = 'block';
                    resultVideo.style.display = 'none';
                    resultVideo.src = ''; // 清除视频源
                } else if (file.type.startsWith('video/')) {
                    resultVideo.src = resultPath;
                    resultVideo.style.display = 'block';
                    resultImage.style.display = 'none';
                    resultImage.src = ''; // 清除图片源
                    resultVideo.load(); // 重新加载视频
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Prediction failed: ' + error.message);
            }
        });
    </script>
</body>
</html>

```

`yolov8n/app.py`

```python
from flask import Flask, request, render_template, send_from_directory, jsonify
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# 确保文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


def predict_yolov11x(input_path, file):
    try:
        # 使用 yolov11x 模型路径
        model = YOLO(r'E:\education\road_snow\yolov11m\runs\detect\train38\weights\best.pt')
        results = model.predict(
            source=input_path,
            project=app.config['RESULTS_FOLDER'],
            name=os.path.splitext(file.filename)[0],
            save=True,
            exist_ok=True
        )

        result_dir = str(results[0].save_dir)
        result_files = [f for f in os.listdir(result_dir) if f.endswith(('.avi', '.jpg', '.png'))]

        if not result_files:
            raise RuntimeError("No output file generated.")

        result_filename = result_files[0]

        # 确保 ffmpeg 在系统 PATH 中
        if result_filename.endswith('.avi'):
            result_filename_mp4 = os.path.splitext(result_filename)[0] + '.mp4'
            avi_path = os.path.join(result_dir, result_filename)
            mp4_path = os.path.join(result_dir, result_filename_mp4)

            # 调用 ffmpeg 进行转换
            subprocess.run(['ffmpeg', '-i', avi_path, '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', mp4_path], check=True)

            os.remove(avi_path)  # 删除原 .avi 文件
            result_filename = result_filename_mp4

        print(f"Result saved to: {result_dir}\{result_filename}")
        return f'/static/results/{os.path.basename(result_dir)}/{result_filename}'

    except Exception as e:
        print(f"Error in predict_yolov11x: {str(e)}")
        return None


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 安全处理文件名
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(input_path)
        result_url = predict_yolov11x(input_path, file)

        if not result_url:
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify({'result': result_url})

    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

```

直接运行

```
python app.py
```

即可在`http://127.0.0.1:5000/`与前端交互
