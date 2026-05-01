"""
DETR系列模型对比实验脚本
支持批量训练和对比: RT-DETR系列
评估指标: F1 Score, mAP50, mAP50-95, Params, FPS
"""
import os
import json
import torch
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from ultralytics import RTDETR


def get_timestamp():
    """生成时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_model_metrics(model, model_name, train_result_path, data_yaml='yolov11m.yaml'):
    """
    提取模型的关键指标
    
    Args:
        model: 训练好的模型
        model_name: 模型名称
        train_result_path: 训练结果路径
        data_yaml: 数据集配置
        
    Returns:
        dict: 包含F1、mAP50、mAP50-95、Param、FPS的字典
    """
    metrics = {
        'Model': model_name,
        'F1': None,
        'mAP50': None,
        'mAP50-95': None,
        'Params(M)': None,
        'FPS': None
    }
    
    try:
        # 1. 从results.csv读取训练指标
        results_csv_path = os.path.join(train_result_path, 'results.csv')
        if os.path.exists(results_csv_path):
            df = pd.read_csv(results_csv_path)
            df.columns = df.columns.str.strip()
            
            # 获取mAP50最高的那一行
            if 'metrics/mAP50(B)' in df.columns:
                best_idx = df['metrics/mAP50(B)'].idxmax()
                best_row = df.loc[best_idx]
                
                # 计算F1 Score
                precision = best_row.get('metrics/precision(B)', 0)
                recall = best_row.get('metrics/recall(B)', 0)
                if precision + recall > 0:
                    metrics['F1'] = round(2 * precision * recall / (precision + recall), 4)
                
                # 提取mAP指标
                metrics['mAP50'] = round(best_row.get('metrics/mAP50(B)', 0), 4)
                metrics['mAP50-95'] = round(best_row.get('metrics/mAP50-95(B)', 0), 4)
        
        # 2. 获取模型参数量和FPS
        best_weight = os.path.join(train_result_path, 'weights', 'best.pt')
        if os.path.exists(best_weight):
            try:
                model_loaded = RTDETR(best_weight)
                
                # 计算参数量
                total_params = sum(p.numel() for p in model_loaded.model.parameters())
                metrics['Params(M)'] = round(total_params / 1e6, 2)
                
                # 测试FPS
                print(f"  正在测试 {model_name} 的推理速度...")
                val_results = model_loaded.val(
                    data=data_yaml,
                    batch=1,
                    imgsz=640,
                    verbose=False
                )
                
                if hasattr(val_results, 'speed'):
                    inference_time = val_results.speed.get('inference', 0)
                    if inference_time > 0:
                        metrics['FPS'] = round(1000 / inference_time, 2)
                        
            except Exception as e:
                print(f"  警告: 获取 {model_name} 参数量或FPS时出错: {str(e)}")
        
        print(f"  ✓ {model_name} 指标提取完成:")
        print(f"    F1={metrics['F1']}, mAP50={metrics['mAP50']}, mAP50-95={metrics['mAP50-95']}")
        print(f"    Params={metrics['Params(M)']}M, FPS={metrics['FPS']}")
        
    except Exception as e:
        print(f"  ✗ {model_name} 指标提取失败: {str(e)}")
    
    return metrics


def train_single_model(model_name, model_path, data_yaml, save_dir, train_params):
    """
    训练单个RT-DETR模型
    
    Args:
        model_name: 模型名称
        model_path: 预训练权重路径
        data_yaml: 数据集配置文件
        save_dir: 结果保存目录
        train_params: 训练参数字典
    """
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"{'='*60}")
    
    # 初始化模型
    model = RTDETR(model_path)
    
    # 配置训练参数
    train_args = {
        'data': data_yaml,
        'epochs': train_params.get('epochs', 300),
        'batch': train_params.get('batch', 16),
        'imgsz': train_params.get('imgsz', 640),
        'device': train_params.get('device', '0'),
        'optimizer': train_params.get('optimizer', 'AdamW'),
        'lr0': train_params.get('lr0', 0.0001),
        'lrf': train_params.get('lrf', 0.01),
        'weight_decay': train_params.get('weight_decay', 0.0001),
        'warmup_epochs': train_params.get('warmup_epochs', 10),
        'patience': train_params.get('patience', 50),
        'mosaic': train_params.get('mosaic', 1.0),
        'mixup': train_params.get('mixup', 0.1),
        'copy_paste': train_params.get('copy_paste', 0.1),
        'amp': train_params.get('amp', True),
        'cache': train_params.get('cache', True),
        'plots': True,
        'project': save_dir,
        'name': f'{model_name}_{get_timestamp()}',
    }
    
    # 打印配置
    print(f"\n{model_name} 训练配置:")
    print(f"  epochs: {train_args['epochs']}, batch: {train_args['batch']}, imgsz: {train_args['imgsz']}")
    print(f"  optimizer: {train_args['optimizer']}, lr0: {train_args['lr0']}, lrf: {train_args['lrf']}")
    print(f"  保存路径: {train_args['project']}/{train_args['name']}")
    
    # 启动训练
    try:
        results = model.train(**train_args)
        result_path = os.path.join(train_args['project'], train_args['name'])
        print(f"\n✓ {model_name} 训练完成! 结果保存至: {result_path}")
        return results, result_path, model
    except Exception as e:
        print(f"\n✗ {model_name} 训练失败: {str(e)}")
        return None, None, None


def compare_detr_models():
    """
    主函数：对比多个DETR系列模型
    """
    # 训练参数配置（针对DETR优化）
    train_params = {
        'data': 'yolov11m.yaml',
        'epochs': 300,
        'batch': 16,  # DETR通常用较小的batch
        'imgsz': 640,
        'device': '0',
        'optimizer': 'AdamW',
        'lr0': 0.0001,  # DETR学习率通常较小
        'lrf': 0.01,
        'weight_decay': 0.0001,
        'warmup_epochs': 10,
        'patience': 50,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'amp': True,
        'cache': True
    }
    
    # 定义要对比的DETR模型列表
    models_to_compare = [
        ('rtdetr-l', 'rtdetr-l.pt'),
        ('rtdetr-x', 'rtdetr-x.pt'),
        # 如果有其他DETR变体可以在这里添加
        # ('deformable-detr-base', 'deformable_detr_base.pt'),
    ]
    
    # 创建主结果目录
    timestamp = get_timestamp()
    main_save_dir = f'runs/compare/detr_comparison_{timestamp}'
    os.makedirs(main_save_dir, exist_ok=True)
    
    # 保存实验配置
    config_info = {
        'timestamp': timestamp,
        'train_params': train_params,
        'models': [m[0] for m in models_to_compare],
        'description': 'DETR系列算法对比实验'
    }
    
    with open(f'{main_save_dir}/experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=4, ensure_ascii=False)
    
    print("="*60)
    print("DETR系列算法对比实验")
    print("="*60)
    print(f"实验时间: {timestamp}")
    print(f"对比模型: {', '.join([m[0] for m in models_to_compare])}")
    print(f"训练参数: epochs={train_params['epochs']}, batch={train_params['batch']}, "
          f"lr0={train_params['lr0']}")
    print(f"结果保存至: {main_save_dir}")
    print("="*60)
    
    # 存储所有模型的结果
    all_results = []
    
    # 依次训练每个模型
    for model_name, model_weight in models_to_compare:
        result, result_path, model = train_single_model(
            model_name=model_name,
            model_path=model_weight,
            data_yaml=train_params['data'],
            save_dir=main_save_dir,
            train_params=train_params
        )
        
        if result is not None and result_path is not None:
            # 提取模型指标
            print(f"\n提取 {model_name} 的性能指标...")
            metrics = extract_model_metrics(
                model, 
                model_name, 
                result_path,
                data_yaml=train_params['data']
            )
            all_results.append(metrics)
        else:
            # 训练失败，记录空指标
            all_results.append({
                'Model': model_name,
                'F1': None,
                'mAP50': None,
                'mAP50-95': None,
                'Params(M)': None,
                'FPS': None
            })
    
    # 生成综合对比报告
    generate_final_comparison(all_results, main_save_dir, train_params)
    
    print("\n" + "="*60)
    print("所有模型训练完成!")
    print(f"详细结果请查看: {main_save_dir}")
    print("="*60)


def generate_final_comparison(results, save_dir, train_params):
    """
    生成最终对比报告
    
    Args:
        results: 所有模型的指标列表
        save_dir: 保存目录
        train_params: 训练参数
    """
    print("\n" + "="*60)
    print("生成综合对比报告...")
    print("="*60)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按mAP50降序排序
    df_sorted = df.sort_values('mAP50', ascending=False, na_position='last')
    
    # 保存CSV文件
    timestamp = get_timestamp()
    csv_path = f'{save_dir}/detr_comparison_{timestamp}.csv'
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 综合对比CSV已保存: {csv_path}")
    
    # 生成文本报告
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("DETR系列算法对比实验 - 综合报告")
    report_lines.append("="*80)
    report_lines.append(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n训练参数配置:")
    for key, value in train_params.items():
        report_lines.append(f"  {key}: {value}")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("性能对比结果（按mAP50排序）:")
    report_lines.append("="*80)
    report_lines.append(f"{'模型':<20} {'F1':<10} {'mAP50':<10} {'mAP50-95':<12} {'参数量(M)':<12} {'FPS':<10}")
    report_lines.append("-"*80)
    
    for _, row in df_sorted.iterrows():
        model = row['Model']
        f1 = f"{row['F1']:.4f}" if pd.notna(row['F1']) else "N/A"
        map50 = f"{row['mAP50']:.4f}" if pd.notna(row['mAP50']) else "N/A"
        map50_95 = f"{row['mAP50-95']:.4f}" if pd.notna(row['mAP50-95']) else "N/A"
        params = f"{row['Params(M)']:.2f}" if pd.notna(row['Params(M)']) else "N/A"
        fps = f"{row['FPS']:.2f}" if pd.notna(row['FPS']) else "N/A"
        
        report_lines.append(f"{model:<20} {f1:<10} {map50:<10} {map50_95:<12} {params:<12} {fps:<10}")
    
    report_lines.append("="*80)
    report_lines.append("\n指标说明:")
    report_lines.append("  - F1 Score: 精确率和召回率的调和平均数")
    report_lines.append("  - mAP50: IoU阈值为0.5时的平均精度")
    report_lines.append("  - mAP50-95: IoU阈值从0.5到0.95的平均精度")
    report_lines.append("  - Params(M): 模型参数量（百万）")
    report_lines.append("  - FPS: 每秒处理帧数（batch=1, imgsz=640）")
    report_lines.append("  - 各模型的详细训练日志和权重文件保存在对应子目录中")
    report_lines.append("="*80)
    
    # 保存文本报告
    report_path = f'{save_dir}/final_report_{timestamp}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 综合报告已保存: {report_path}")
    
    # 打印到控制台
    print("\n" + '\n'.join(report_lines))
    
    # 打印性能分析
    print("\n" + "="*80)
    print("性能分析:")
    print("="*80)
    
    valid_results = df_sorted.dropna(subset=['mAP50'])
    if not valid_results.empty:
        best_map50 = valid_results.iloc[0]
        print(f"✓ mAP50最高: {best_map50['Model']} ({best_map50['mAP50']:.4f})")
        
        if 'F1' in valid_results.columns and valid_results['F1'].notna().any():
            best_f1 = valid_results.loc[valid_results['F1'].idxmax()]
            print(f"✓ F1 Score最高: {best_f1['Model']} ({best_f1['F1']:.4f})")
        
        if 'FPS' in valid_results.columns and valid_results['FPS'].notna().any():
            best_fps = valid_results.loc[valid_results['FPS'].idxmax()]
            print(f"✓ FPS最高(速度最快): {best_fps['Model']} ({best_fps['FPS']:.2f})")
        
        if 'Params(M)' in valid_results.columns and valid_results['Params(M)'].notna().any():
            min_params = valid_results.loc[valid_results['Params(M)'].idxmin()]
            print(f"✓ 参数量最少(最轻量): {min_params['Model']} ({min_params['Params(M)']:.2f}M)")
    
    print("="*80)


def main():
    """主入口"""
    compare_detr_models()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
    print("\n============ DETR对比实验全部完成 ============")