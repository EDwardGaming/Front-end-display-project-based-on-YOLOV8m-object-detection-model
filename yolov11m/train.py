import torch
from ultralytics import YOLO

def main():
    # 初始化模型（加载预训练权重）
    model = YOLO(r"E:\education\road_snow\yolov11m\yolo11m.pt")

    # 训练参数配置（大目标召回率优化版）
    train_args = {
        # ===== 数据配置 =====
        'data': 'yolov11m.yaml',  # 数据集配置文件路径
        
        # ===== 核心训练参数 =====
        'epochs': 150,       # [100-300] 总训练轮次。调大：更充分学习大目标特征；调小：可能欠拟合
        'batch': 24,         # [8-32] 批次大小。调大：稳定训练但需更大显存；调小：适合高分辨率训练
        'imgsz': 640,       # [640-1280] 输入分辨率。调大：提升大目标细节捕捉能力；调小：加快训练速度
        'device': '0',       # 使用GPU训练
        
        # ===== 优化器配置 =====
        'optimizer': 'AdamW',# 优化器选择。AdamW适合小批量数据，SGD更适合大批量
        'lr0': 0.001,        # [0.0005-0.002] 初始学习率。调大：加速收敛但可能震荡；调小：稳定但收敛慢
        'lrf': 0.01,         # [0.005-0.05] 最终学习率=lr0*lrf。调大：更快衰减；调小：保持后期学习能力
        'weight_decay': 0.005,# [0.001-0.01] 权重衰减。调大：防止过拟合；调小：模型容量更大
        
        # ===== 损失函数权重 =====
        'box': 10.0,         # [7.0-15.0] 定位损失权重。调大：增强大目标框体回归；调小：侧重分类任务
        'cls': 0.5,          # [0.3-1.0] 分类损失权重。调大：提高分类精度；调小：降低误检惩罚
        'dfl': 1.8,          # [1.5-2.5] 分布焦点损失。调大：提升边界预测；调小：降低分布约束
        
        # ===== 训练策略 =====
        'warmup_epochs': 15, # [10-20] 学习率预热。调大：稳定大目标初期学习；调小：快速进入正常训练
        'label_smoothing': 0.1,# [0.0-0.2] 标签平滑。调大：防止过拟合；调小：保持原始标签置信度
        'close_mosaic': 15,  # [10-20] 最后关闭mosaic的轮次。调大：更晚关闭增强；调小：提前稳定训练
        'patience': 30,      # [20-50] 早停等待。调大：给大目标充分收敛时间；调小：快速停止
        
        # ===== 工程优化 =====
        'workers': 0,        # Windows必须设为0
        'amp': True,         # 混合精度训练。节省30%显存
        'project': 'runs/detect',
        'name': 'snow_v5_largefocus',
        
        # ===== 高级配置 =====
        'overlap_mask': True, # 增强掩模重叠处理（针对大面积目标）
        #'image_weights': True # 启用困难样本加权采样（平衡大小目标）
    }

    # 打印配置
    print("============ 训练配置 ============")
    for k, v in train_args.items():
        print(f"{k:20s} : {v}")

    # 启动训练
    results = model.train(**train_args)

    # 输出关键指标
    print("\n============ 训练结果 ============")
    print(f"大目标召回率: {results.results_dict['metrics/recall(B)']:.3f}")
    print(f"小目标召回率: {results.results_dict['metrics/recall(S)']:.3f}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()