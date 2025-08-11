#在要生成训练集的目录下执行即可
import os
import json
import shutil
from ultralytics.data.split import autosplit

class dataprocessor:
    def __init__(self, image_dir, label_output_dir, image_output_dir):
        self.image_dir = image_dir
        self.label_output_dir = label_output_dir
        self.image_output_dir = image_output_dir
        os.makedirs(self.label_output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)
        self.suffix_file =  [".PNG",".JPG",".JPEG",".jpg",".png",".jpeg"]  # 支持的图片后缀
        self.classes = {"ice": 1,"snow": 0}  # 定义类别列表，顺序与 class_id 一致

        
    # 解析 JSON 并转换为 YOLO 格式
    def convert_to_yolo(self,json_file, img_width, img_height):
        with open(json_file, "r") as f:
            data = json.load(f)

        yolo_labels = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                label = shape['label']
                if label not in self.classes:
                    continue
                class_id = self.classes[label]

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

        return yolo_labels # 返回一个字符串

    # 保存图片和标签到相应的输出目录
    def save_data(self,img_file, labels):
        
        base_name = os.path.splitext(os.path.basename(img_file))[0] # 获取文件名不带扩展名
        for i in self.suffix_file:
            if os.path.basename(img_file).endswith(i):
                img_output_path = os.path.join(self.image_output_dir, f"{base_name}{i}")
                break
        label_output_path = os.path.join(self.label_output_dir, f"{base_name}.txt")

        with open(label_output_path, "w") as f:
            f.write("\n".join(labels))

        # 复制图片
        shutil.copy(img_file, img_output_path)

    # 遍历目录下的所有图片文件进行json转换txt并保存
    def traveldir_solve_json(self,dir_path):
        dir_name_lis = os.listdir(dir_path)
        for dir_name in dir_name_lis:
            for img_file in os.listdir(os.path.join(dir_path, dir_name)):
                for i in self.suffix_file:
                    if img_file.endswith(i):
                        json_file = os.path.join(dir_path, dir_name, os.path.splitext(img_file)[0] + ".json")
                        if os.path.exists(json_file):#如果存在对应的json则操作
                            # 假设图片分辨率为 2400x1600，可以根据实际情况调整
                            img_width, img_height = 2400, 1600
                            labels = self.convert_to_yolo(json_file, img_width, img_height)
                            self.save_data(os.path.join(dir_path, dir_name, img_file), labels)
                            break
        print("json解析转换保存流程已完成！")
    
    def traveldir_solve_pure(self,dir_path):
        # 遍历所有类别的图片
        for category, dir_path in dir_path.items():
            class_id = self.classes[category]
            all_images = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]

            # 处理训练集
            for filename in all_images:
                src_img_path = os.path.join(dir_path, filename)
                dst_img_path = os.path.join(self.image_output_dir, filename)

                # 复制图片
                shutil.copy(src_img_path, dst_img_path)

                # 生成标签
                label_filename = os.path.splitext(filename)[0] + ".txt"
                label_path = os.path.join(self.label_output_dir, label_filename)
                with open(label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        print("dataset划分完成！")

    def split_train_val(self):
        # 划分训练集和验证集
        autosplit(self.image_output_dir,weights=(0.8, 0.2, 0.0),annotated_only=True)
        print("训练集和验证集划分完成！")

# 设置路径全局变量
json_image_dir = "E:/education/dataset/json_dataset"  # json标注好的原始图片数据集的目录
json_label_output_dir = "training_set/labels"
json_image_output_dir = "training_set/images"

# 处理json标注好的图片数据集
dataprocessor_json = dataprocessor(image_dir=json_image_dir,label_output_dir=json_label_output_dir,image_output_dir=json_image_output_dir)
dataprocessor_json.traveldir_solve_json(json_image_dir)  # 执行json解析转换保存流程

# 处理纯图片数据集
pure_base_dirs = {
    "ice": "E:/education/dataset/pure_dataset/ice",
    "snow": "E:/education/dataset/pure_dataset/snow"
}
dataprocessor_pure = dataprocessor(image_dir=pure_base_dirs,label_output_dir=json_label_output_dir,image_output_dir=json_image_output_dir)
dataprocessor_pure.traveldir_solve_pure(pure_base_dirs)  # 执行dataset划分流程

# 划分训练集和验证集
dataprocessor_pure.split_train_val()  # 执行训练集和验证