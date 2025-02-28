import torch
from ultralytics import YOLO
import torch.multiprocessing as mp



# 加载预训练的 YOLOv8 模型
model = YOLO(r"D:\Program Files\road_snow\yolov8n\yolov8x.pt")  # 这里可以根据你的需求替换为其他版本的 YOLO 模型

# 配置训练参数
train_args = {
    'data': 'yolov8x.yaml',  # 数据集配置文件路径
    'epochs': 50,  # 训练的轮数
    'batch': 24,  # 每个批次的大小，适当调整（16到32之间）
    'imgsz': 640,  # 输入图像的大小
    'device': 'cuda',  # 使用 GPU 训练，如果没有 GPU 可以改为 'cpu'
    'project': 'runs/detect',  # 存储训练结果的目录
    'name': 'train3',  # 训练结果保存的子目录
    'save': True,  # 是否保存模型
    'save_period': -1,  # 每隔多少轮保存一次模型
    'verbose': True,  # 是否打印详细日志
    'workers': 0,  # 数据加载器的工作线程数
    'optimizer': 'AdamW',  # 使用 AdamW 优化器
    'lr0': 0.001,  # 初始学习率，适当降低初始学习率
    'lrf': 0.2,  # 学习率衰减率
    'warmup_epochs': 3,  # 预热的轮数，适当降低预热时间
    'box': 7.5,  # 训练的框回归损失权重
    'cls': 0.5,  # 类别损失权重
    'dfl': 1.5,  # 关键点损失权重
    'pose': 12.0,  # 姿态估计损失权重
    'nbs': 64,  # 批次大小（此参数不会影响实际训练，但用于计算资源）
    'freeze': 20,  # 冻结的层数（可尝试更多层的冻结）
    'split': "0.8 0.2",  # 训练集和验证集的比例
}

# 打印训练参数
print(train_args)

# 训练模型
train_results = model.train(**train_args)

# 输出训练结果
print(train_results)
