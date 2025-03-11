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