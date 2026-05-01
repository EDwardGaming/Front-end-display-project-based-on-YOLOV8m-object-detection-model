"""
快速测试脚本
验证环境配置和模型加载
"""
import sys
import torch
from ultralytics import RTDETR, YOLO


def check_dependencies():
    """检查依赖包"""
    print("="*60)
    print("检查依赖包...")
    print("="*60)
    
    packages = {
        'torch': torch.__version__,
    }
    
    try:
        import pandas as pd
        packages['pandas'] = pd.__version__
    except ImportError:
        packages['pandas'] = '❌ 未安装'
    
    try:
        from PIL import Image
        packages['Pillow'] = '✓ 已安装'
    except ImportError:
        packages['Pillow'] = '❌ 未安装'
    
    try:
        import ultralytics
        packages['ultralytics'] = ultralytics.__version__
    except ImportError:
        packages['ultralytics'] = '❌ 未安装'
    
    try:
        import transformers
        packages['transformers'] = transformers.__version__
    except ImportError:
        packages['transformers'] = '❌ 未安装 (可选)'
    
    for name, version in packages.items():
        print(f"  {name:<20} {version}")
    
    print("="*60)


def check_cuda():
    """检查CUDA"""
    print("\nCUDA信息:")
    print("="*60)
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    显存: {mem_total:.2f} GB")
    
    print("="*60)


def test_yolo_loading():
    """测试YOLO模型加载"""
    print("\n测试YOLO模型加载:")
    print("="*60)
    
    models_to_test = [
        'yolov8n.pt',
        'yolov11m.pt',
    ]
    
    for model_name in models_to_test:
        try:
            print(f"  尝试加载 {model_name}...", end=' ')
            model = YOLO(model_name)
            print("✓ 成功")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")
    
    print("="*60)


def test_rtdetr_loading():
    """测试RT-DETR模型加载"""
    print("\n测试RT-DETR模型加载:")
    print("="*60)
    
    models_to_test = [
        'rtdetr-l.pt',
        'rtdetr-x.pt',
        'deformable-detr-base',
        'deformable_detr_base.pt'
    ]
    
    for model_name in models_to_test:
        try:
            print(f"  尝试加载 {model_name}...", end=' ')
            model = RTDETR(model_name)
            print("✓ 成功")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")
    
    print("="*60)


def test_dataset():
    """测试数据集"""
    import os
    from pathlib import Path
    
    print("\n检查数据集:")
    print("="*60)
    
    # YOLO格式
    yolo_path = Path('./yolov11m/datasets')
    if yolo_path.exists():
        print("  ✓ YOLO格式数据集存在")
        
        images_dir = yolo_path / 'images'
        labels_dir = yolo_path / 'labels'
        
        if images_dir.exists():
            img_count = len(list(images_dir.rglob('*.jpg'))) + len(list(images_dir.rglob('*.png')))
            print(f"    图片数量: {img_count}")
        
        if labels_dir.exists():
            label_count = len(list(labels_dir.rglob('*.txt')))
            print(f"    标注数量: {label_count}")
        
        # 检查split文件
        train_txt = yolo_path / 'autosplit_train.txt'
        val_txt = yolo_path / 'autosplit_val.txt'
        
        if train_txt.exists():
            with open(train_txt, 'r') as f:
                train_count = len(f.readlines())
            print(f"    训练集: {train_count} 张")
        
        if val_txt.exists():
            with open(val_txt, 'r') as f:
                val_count = len(f.readlines())
            print(f"    验证集: {val_count} 张")
    else:
        print("  ✗ YOLO格式数据集不存在")
    
    # COCO格式
    coco_path = Path('./datasets_coco')
    if coco_path.exists():
        print("  ✓ COCO格式数据集存在")
        
        train_json = coco_path / 'annotations' / 'instances_train.json'
        val_json = coco_path / 'annotations' / 'instances_val.json'
        
        if train_json.exists():
            import json
            with open(train_json, 'r') as f:
                data = json.load(f)
            print(f"    训练集图片: {len(data['images'])}")
            print(f"    训练集标注: {len(data['annotations'])}")
        
        if val_json.exists():
            import json
            with open(val_json, 'r') as f:
                data = json.load(f)
            print(f"    验证集图片: {len(data['images'])}")
            print(f"    验证集标注: {len(data['annotations'])}")
    else:
        print("  ✗ COCO格式数据集不存在")
        print("    运行: python convert_yolo_to_coco.py 来转换")
    
    print("="*60)


def quick_inference_test():
    """快速推理测试"""
    print("\n快速推理测试:")
    print("="*60)
    
    try:
        from PIL import Image
        import numpy as np
        
        # 创建一个测试图片
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 测试YOLO
        print("  测试YOLO推理...", end=' ')
        model = YOLO('yolov8n.pt')
        results = model(test_img, verbose=False)
        print("✓ 成功")
        
        # 测试RT-DETR
        print("  测试RT-DETR推理...", end=' ')
        model = RTDETR('rtdetr-l.pt')
        results = model(test_img, verbose=False)
        print("✓ 成功")
        
        print("\n  推理功能正常!")
        
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
    
    print("="*60)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DETR系列模型环境测试")
    print("="*60 + "\n")
    
    # 1. 检查依赖
    check_dependencies()
    
    # 2. 检查CUDA
    check_cuda()
    
    # 3. 测试模型加载
    test_yolo_loading()
    test_rtdetr_loading()
    
    # 4. 检查数据集
    test_dataset()
    
    # 5. 快速推理测试
    quick_inference_test()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print("\n下一步:")
    print("  1. 如果COCO格式数据集不存在: python convert_yolo_to_coco.py")
    print("  2. 训练单个模型: python train_detr_single.py")
    print("  3. 对比DETR系列: python compare_detr_models.py")
    print("  4. 完整对比实验: python compare_yolo_vs_detr.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()