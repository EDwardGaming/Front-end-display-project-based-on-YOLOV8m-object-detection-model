# 该脚本实现划分未标注dataset为训练集目录的功能

import os
import shutil
import random



# 定义类别和 class_id
categories = {
    "冰": 1,
    "雪": 0
}

# 原始数据目录
base_dirs = {
    "冰": r"E:\education\dataset\pure_dataset\冰",
    "雪": r"E:\education\dataset\pure_dataset\雪"
}

# 目标存放路径
dataset_dir = r"E:\education\road_snow\training_set" #dataset绝对路径
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

print("dataset划分完成！")
