"""
YOLO格式转COCO格式数据集转换脚本
将YOLO的txt标注转换为COCO的json格式
"""
import os
import json
from PIL import Image
from pathlib import Path
from datetime import datetime
import shutil

def convert_yolo_to_coco(
    yolo_data_dir,
    output_dir,
    train_txt='autosplit_train.txt',
    val_txt='autosplit_val.txt',
    class_names=['snow', 'ice']
):
    """
    将YOLO格式数据集转换为COCO格式
    
    Args:
        yolo_data_dir: YOLO数据集根目录（包含datasets文件夹）
        output_dir: 输出COCO格式数据集的目录
        train_txt: 训练集图片列表文件
        val_txt: 验证集图片列表文件
        class_names: 类别名称列表
    """
    
    # 创建输出目录结构
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / 'images'
    annotations_dir = output_dir / 'annotations'
    
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换训练集和验证集
    for split, txt_file in [('train', train_txt), ('val', val_txt)]:
        print(f"\n{'='*60}")
        print(f"转换 {split} 数据集...")
        print(f"{'='*60}")
        
        # 读取图片列表
        txt_path = Path(yolo_data_dir) / 'datasets' / txt_file
        if not txt_path.exists():
            print(f"警告: {txt_path} 不存在，跳过 {split} 集")
            continue
            
        with open(txt_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"共找到 {len(image_paths)} 张图片")
        
        # 初始化COCO格式数据结构
        coco_data = {
            "info": {
                "description": "Snow Detection Dataset (Converted from YOLO)",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 添加类别信息
        for idx, class_name in enumerate(class_names):
            coco_data["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "weather"
            })
        
        annotation_id = 1
        successful_images = 0
        
        # 处理每张图片
        for image_id, img_path_str in enumerate(image_paths, start=1):
            # 处理路径
            img_path = Path(yolo_data_dir) / 'datasets' / img_path_str.replace('\\', '/')
            
            if not img_path.exists():
                print(f"警告: 图片不存在 {img_path}")
                continue
            
            try:
                # 读取图片获取尺寸
                img = Image.open(img_path)
                width, height = img.size
                
                # 添加图片信息
                img_filename = f"{split}_{image_id:06d}{img_path.suffix}"
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": img_filename,
                    "width": width,
                    "height": height
                })
                
                # 复制图片到新目录
                dst_img_path = images_dir / split / img_filename
                shutil.copy2(img_path, dst_img_path)
                
                # 读取对应的YOLO标注文件
                label_path = img_path.parent.parent / 'labels' / img_path.stem
                label_path = label_path.with_suffix('.txt')
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # YOLO格式: class_id x_center y_center width height (归一化)
                            parts = line.split()
                            if len(parts) != 5:
                                continue
                            
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * width
                            y_center = float(parts[2]) * height
                            bbox_width = float(parts[3]) * width
                            bbox_height = float(parts[4]) * height
                            
                            # 转换为COCO格式 (x_min, y_min, width, height)
                            x_min = x_center - bbox_width / 2
                            y_min = y_center - bbox_height / 2
                            
                            # 确保bbox在图片范围内
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            bbox_width = min(bbox_width, width - x_min)
                            bbox_height = min(bbox_height, height - y_min)
                            
                            if bbox_width > 0 and bbox_height > 0:
                                coco_data["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": class_id,
                                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                                    "area": bbox_width * bbox_height,
                                    "iscrowd": 0
                                })
                                annotation_id += 1
                
                successful_images += 1
                if successful_images % 100 == 0:
                    print(f"  已处理 {successful_images}/{len(image_paths)} 张图片")
                    
            except Exception as e:
                print(f"处理图片出错 {img_path}: {str(e)}")
                continue
        
        # 保存COCO格式的json文件
        json_path = annotations_dir / f'instances_{split}.json'
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\n✓ {split} 集转换完成!")
        print(f"  图片数量: {len(coco_data['images'])}")
        print(f"  标注数量: {len(coco_data['annotations'])}")
        print(f"  保存路径: {json_path}")
    
    # 创建数据集配置文件
    config = {
        "dataset_name": "Snow Detection",
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_annotations": str(annotations_dir / 'instances_train.json'),
        "val_annotations": str(annotations_dir / 'instances_val.json'),
        "train_images": str(images_dir / 'train'),
        "val_images": str(images_dir / 'val'),
        "conversion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = output_dir / 'dataset_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print("数据集转换完成!")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"配置文件: {config_path}")
    print(f"{'='*60}")
    
    return str(output_dir)


def main():
    """主函数"""
    # 配置参数
    yolo_data_dir = './yolov11m'  # YOLO数据集根目录
    output_dir = './datasets_coco'  # COCO格式输出目录
    
    print("="*60)
    print("YOLO格式 → COCO格式 数据集转换")
    print("="*60)
    print(f"输入目录: {yolo_data_dir}")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    # 执行转换
    result_dir = convert_yolo_to_coco(
        yolo_data_dir=yolo_data_dir,
        output_dir=output_dir,
        train_txt='autosplit_train.txt',
        val_txt='autosplit_val.txt',
        class_names=['snow', 'ice']
    )
    
    print("\n转换完成! 可以开始训练DETR系列模型了。")


if __name__ == "__main__":
    main()