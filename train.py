# hyperparameter_tuning_v2.py - YOLO超参数调优 (基于官方文档和实验结果优化)
import torch
from ultralytics import YOLO
import itertools
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

class YOLOHyperparameterTuner:
    def __init__(self):
        self.results_log = []
        self.best_result = {
            'recall': 0,
            'precision': 0, 
            'f1': 0,
            'params': {}
        }
        
        # 创建结果保存目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"hyperparameter_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def define_search_space(self):
        """定义超参数搜索空间 - 基于官方文档和实验结果优化"""
        
        # 方案1: 基于官方建议的精细化搜索 (重点推荐)
        search_space_official= {
            # 细化cls低值区间（0.2-0.7），验证0.5是否为最优
            "cls": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1],  
            # 细化box最优区间（3-7），验证5附近是否有更优值
            "box": [3, 4, 5, 6, 7, 8, 9,10]               
        }
        
        return search_space_official
    
    def use_official_tune_method(self, search_space):
        """使用官方tune方法进行超参数搜索"""
        print("\n🔬 使用Ultralytics官方tune方法...")
        
        try:
            # 初始化模型
            model = YOLO("yolo11m.pt")
            
            # 使用官方tune方法
            results = model.tune(
                data='yolov11m.yaml',
                epochs=5,              # 官方建议的epochs
                batch=32,             # 保持您的原始设置
                imgsz=640,              # 使用标准尺寸，加速搜索
                iterations=len(list(itertools.product(*search_space.values()))),  # 根据搜索空间计算
                optimizer="AdamW",
                space=search_space,      # 直接使用搜索空间字典
                plots=True,              # 开启绘图以便分析
                save=True,               # 保存最佳模型
                val=True,                # 开启验证
                device='0',
                project=f'{self.results_dir}/official_tune',
                name='tune_results',
                cache=True,
                patience=20,             # 适当减少patience
            )
            
            print("✅ 官方tune方法完成！")
            print(f"📁 结果保存在: {self.results_dir}/official_tune/tune_results/")
            
            return results
            
        except Exception as e:
            print(f"❌ 官方tune方法失败: {str(e)}")
            return None
    
    def train_single_configuration(self, params, config_id):
        """训练单个参数配置"""
        print(f"\n{'='*60}")
        print(f"开始训练配置 {config_id}: {params}")
        print(f"{'='*60}")
        
        try:
            # 初始化模型
            model = YOLO("yolo11m.pt")
            
            # 基础训练参数 (保持与您原始配置一致)
            base_train_args = {
                'data': 'yolov11m.yaml',
                'epochs': 150,            # 调优阶段使用较少epochs
                'batch': 32,             # 保持您的原始设置
                'imgsz': 640,            # 使用标准尺寸，加速搜索
                'device': '0',
                'optimizer': 'AdamW',
                'lr0': 0.00086,           # 固定学习率，专注搜索cls和box
                'lrf': 0.01,
                'workers': 12,
                'amp': True,
                'cache': True,
                'patience': 20,           # 减少patience用于快速搜索
                'project': f'{self.results_dir}/manual_runs',
                'name': f'config_{config_id:03d}',
                'save_period': -1,        # 不保存中间模型，节省空间
                'plots': False,           # 关闭绘图，加速训练
                'val': True,
                'verbose': False,         # 减少输出
                
                # 数据增强参数 (保持稳定)
                'mosaic': 1.0,
                'close_mosaic': 10,
                'label_smoothing': 0.1,
                'warmup_epochs': 3,
            }
            
            # 合并搜索参数
            train_args = {**base_train_args, **params}
            
            # 开始训练
            results = model.train(**train_args)
            
            # 提取关键指标
            metrics = {
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0)), 
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            }
            
            # 计算F1分数和自定义评分
            if metrics['precision'] + metrics['recall'] > 0:
                f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                # 自定义评分：更重视召回率 (权重 recall:precision = 2:1)
                custom_score = (2 * metrics['recall'] + metrics['precision']) / 3
            else:
                f1 = 0
                custom_score = 0
            
            metrics['f1'] = f1
            metrics['custom_score'] = custom_score
            
            # 记录结果
            result = {
                'config_id': config_id,
                'params': params.copy(),
                'metrics': metrics,
                'model_path': f'{self.results_dir}/manual_runs/config_{config_id:03d}/weights/best.pt'
            }
            
            self.results_log.append(result)
            
            # 更新最佳结果 (优先考虑召回率 > 自定义评分 > F1分数)
            recall_score = metrics['recall']
            custom_score = metrics['custom_score']
            f1_score = metrics['f1']
            
            if (recall_score > self.best_result['recall'] or 
                (recall_score >= self.best_result['recall'] and custom_score > self.best_result.get('custom_score', 0)) or
                (recall_score >= self.best_result['recall'] and custom_score >= self.best_result.get('custom_score', 0) and f1_score > self.best_result['f1'])):
                self.best_result = {
                    'recall': recall_score,
                    'precision': metrics['precision'],
                    'f1': f1_score,
                    'custom_score': custom_score,
                    'mAP50': metrics['mAP50'],
                    'params': params.copy(),
                    'config_id': config_id
                }
            
            print(f"配置 {config_id} 完成:")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  精确率: {metrics['precision']:.4f}") 
            print(f"  F1分数: {f1:.4f}")
            print(f"  自定义评分: {custom_score:.4f}")
            print(f"  mAP50: {metrics['mAP50']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"配置 {config_id} 训练失败: {str(e)}")
            return None
    
    def run_grid_search(self, search_space, max_configs=None):
        """执行网格搜索"""
        # 生成所有参数组合
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        
        if max_configs and len(all_combinations) > max_configs:
            print(f"总共有 {len(all_combinations)} 个组合，限制为 {max_configs} 个")
            # 随机选择部分组合
            import random
            random.seed(42)
            all_combinations = random.sample(all_combinations, max_configs)
        
        print(f"将测试 {len(all_combinations)} 个参数组合")
        
        # 训练每个配置
        successful_configs = 0
        for i, combination in enumerate(all_combinations, 1):
            params = dict(zip(param_names, combination))
            result = self.train_single_configuration(params, i)
            
            if result:
                successful_configs += 1
                
            # 保存中间结果
            self.save_results()
            
            print(f"\n进度: {i}/{len(all_combinations)} ({successful_configs} 个成功)")
            print(f"当前最佳召回率: {self.best_result['recall']:.4f} (自定义评分: {self.best_result.get('custom_score', 0):.4f})")
        
        return successful_configs
    
    def save_results(self):
        """保存结果到文件"""
        # 保存详细结果
        results_file = f"{self.results_dir}/detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results_log, f, indent=2, ensure_ascii=False)
        
        # 保存最佳结果
        best_file = f"{self.results_dir}/best_result.json"
        with open(best_file, 'w', encoding='utf-8') as f:
            json.dump(self.best_result, f, indent=2, ensure_ascii=False)
        
        # 创建结果汇总CSV
        if self.results_log:
            summary_data = []
            for result in self.results_log:
                row = {'config_id': result['config_id']}
                row.update(result['params'])
                row.update(result['metrics'])
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            # 按召回率降序排列
            df = df.sort_values('recall', ascending=False)
            df.to_csv(f"{self.results_dir}/results_summary.csv", index=False)
        
    def analyze_results(self):
        """分析结果并生成洞察"""
        if not self.results_log:
            return
        
        print("\n" + "="*80)
        print("📊 结果分析")
        print("="*80)
        
        # 转换为DataFrame进行分析
        df_data = []
        for result in self.results_log:
            row = {'config_id': result['config_id']}
            row.update(result['params'])
            row.update(result['metrics'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        if len(df) > 0:
            # 分析cls和box的影响
            print("🔍 参数影响分析:")
            
            # cls影响分析
            if 'cls' in df.columns:
                cls_analysis = df.groupby('cls').agg({
                    'recall': ['mean', 'max'],
                    'precision': ['mean'],
                    'f1': ['mean']
                }).round(4)
                print("\n📈 CLS参数影响 (值越低召回率越高的趋势):")
                print(cls_analysis)
            
            # box影响分析  
            if 'box' in df.columns:
                box_analysis = df.groupby('box').agg({
                    'recall': ['mean', 'max'],
                    'precision': ['mean'],
                    'f1': ['mean']
                }).round(4)
                print("\n📦 BOX参数影响:")
                print(box_analysis)
            
            # 找出召回率最高的前5个配置
            top_recall = df.nlargest(5, 'recall')[['config_id', 'cls', 'box', 'recall', 'precision', 'f1']]
            print("\n🏆 召回率最高的5个配置:")
            print(top_recall.to_string(index=False))
    
    def print_summary(self):
        """打印最终汇总结果"""
        print("\n" + "="*80)
        print("🎯 超参数调优完成！")
        print("="*80)
        
        if self.best_result['recall'] > 0:
            print(f"📈 最佳结果 (配置ID: {self.best_result.get('config_id', 'Unknown')}):")
            print(f"   召回率 (Recall):      {self.best_result['recall']:.4f}")
            print(f"   精确率 (Precision):    {self.best_result['precision']:.4f}")
            print(f"   F1分数:               {self.best_result['f1']:.4f}")
            print(f"   自定义评分:           {self.best_result.get('custom_score', 0):.4f}")
            print(f"   mAP50:               {self.best_result.get('mAP50', 0):.4f}")
            print(f"\n🔧 最佳参数组合:")
            for param, value in self.best_result['params'].items():
                print(f"   {param}: {value}")
            
            # 给出建议
            cls_val = self.best_result['params'].get('cls', 0)
            box_val = self.best_result['params'].get('box', 0)
            
            print(f"\n💡 优化建议:")
            if cls_val <= 0.3:
                print(f"   ✅ cls={cls_val} 确实有效提升召回率，继续使用低cls值")
            if box_val >= 0.1:
                print(f"   ✅ box={box_val} 在这个范围效果较好")
            
            print(f"\n📁 结果文件保存在: {self.results_dir}/")
            print("   - detailed_results.json: 详细训练结果")
            print("   - best_result.json: 最佳参数配置")
            print("   - results_summary.csv: 结果汇总表 (按召回率排序)")
        else:
            print("❌ 未找到有效的训练结果")
        
        print("="*80)

def main():
    """主函数"""
    print("🚀 YOLO超参数调优 v2 - 基于官方文档和实验结果优化")
    print("💡 策略: cls越低召回率越高 + 扩大box搜索范围")
    print("📖 参考: Ultralytics官方文档 box(0.02,0.2) cls(0.2,4.0)")
    print("🎯 目标: snow类别召回率 73% -> 80%+")
    
    # 创建调优器
    tuner = YOLOHyperparameterTuner()
    
    # 定义搜索空间
    search_space_official = tuner.define_search_space()

    # 选择搜索策略
    print("\n请选择搜索策略:")
    print("1. 官方tune方法 (推荐, 自动优化)")
    print("2. 基于官方建议的网格搜索 (cls:6个值 × box:8个值 = 48组合)")
    print("3. 超细化低cls搜索 (更细化的低值搜索)")
    print("4. 极限召回率搜索 (激进的极低cls策略)")
    print("5. 手动指定cls和box范围")
    
    #choice = input("请输入选择 (1-5): ").strip()
    choice = '2'
    
    if choice == "1":
        # 使用官方tune方法
        print("\n🔬 使用官方tune方法...")
        # 为官方tune方法准备搜索空间（使用元组格式）
        official_space = {
            "lr0": (0.00001, 0.001),   
            "lrf": (0.01, 1.0)
        }

        results = tuner.use_official_tune_method(official_space)
        return
        
    elif choice == "2":
        print("\n🔍 开始基于官方建议的网格搜索...")
        successful_configs = tuner.run_grid_search(search_space_official)
        
    elif choice == "3":
        print("\n🔍 开始超细化低cls搜索...")
        successful_configs = tuner.run_grid_search(search_space_ultra_fine)
        
    elif choice == "4":
        print("\n🔍 开始极限召回率搜索...")
        successful_configs = tuner.run_grid_search(search_space_extreme)
        
    elif choice == "5":
        # 手动指定范围
        print("\n⚙️  手动指定搜索范围:")
        cls_values = input("请输入cls值列表 (用逗号分隔，如: 0.1,0.2,0.3): ").split(',')
        box_values = input("请输入box值列表 (用逗号分隔，如: 0.05,0.1,0.15): ").split(',')
        
        try:
            custom_space = {
                "cls": [float(x.strip()) for x in cls_values],
                "box": [float(x.strip()) for x in box_values]
            }
            print(f"自定义搜索空间: {custom_space}")
            successful_configs = tuner.run_grid_search(custom_space)
        except ValueError:
            print("输入格式错误，使用默认搜索空间")
            successful_configs = tuner.run_grid_search(search_space_official)
    else:
        print("使用默认官方建议搜索")
        successful_configs = tuner.run_grid_search(search_space_official)
    
    # 分析和打印结果
    tuner.analyze_results()
    tuner.print_summary()
    
    print(f"\n✅ 调优完成！成功训练了 {successful_configs} 个配置")
    print("🎯 建议使用最佳参数重新进行完整训练 (epochs=300-400)")

if __name__ == "__main__":
    # 设置多进程支持
    torch.multiprocessing.freeze_support()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA设备!")
        exit(1)
    
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    main()