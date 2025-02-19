# 该脚本实现划分 将以.json格式标注过的dataset转成.txt格式 并将这些文件构建成训练目录
import os
import json
import random
import shutil

# 设置路径
image_dir = r"D:\dataset\json_dataset"  # 原始图片目录
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
dir_name_lis = os.listdir(image_dir)
for dir_name in dir_name_lis:
    for img_file in os.listdir(os.path.join(image_dir, dir_name)):
        for i in suffix_file:
            if img_file.endswith(i):
                json_file = os.path.join(image_dir, dir_name, img_file.replace(".PNG", ".json").replace(".JPG", ".json").replace(".jpeg", ".json").replace(".JPEG", ".json").replace(".png", ".json").replace(".jpg", ".json"))
                if os.path.exists(json_file):#如果存在对应的json则操作
                    # 假设图片分辨率为 2400x1600，可以根据实际情况调整
                    img_width, img_height = 2400, 1600
                    labels = convert_to_yolo(json_file, img_width, img_height)
                    save_data(os.path.join(image_dir, dir_name, img_file), labels)
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

print("dataset划分完成！")
