"""
训练脚本 - PyTorch + YOLO11
支持GPU/CPU、混合精度、模型保存等
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import json

# 使用 ultralytics YOLO11
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# 工业缺陷检测训练器
class YoloDetector:
    """工业缺陷检测训练器"""
    
    def __init__(self, config_path: str, output_dir: str = './results'):
        """初始化训练器
        
        参数:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.output_dir / f'train_log_{timestamp}.txt'
        
        # 首先加载YOLO全部配置
        all_configs_path = Path('configs/all_configs.yaml')
        if all_configs_path.exists():
            with open(all_configs_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.log(f"✅ 已加载默认配置: {all_configs_path}")
        else:
            self.config = {}
        
        # 然后加载用户配置，覆盖默认配置
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        
        # 合并配置：用户配置优先
        if user_config:
            self._merge_config(self.config, user_config)
        
        # 确定设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 用于存储 trainer 引用的变量（供 GUI 轮询使用）
        self.trainer_ref = None
        
        self.log(f"🚀 工业缺陷检测系统")
        self.log(f"📱 设备: {self.device}")
        self.log(f"⚙️  配置文件: {config_path}\n")
        
        # 初始化模型
        self._init_model()
    
    def log(self, message: str):
        """记录日志"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def _merge_config(self, base_config: dict, user_config: dict) -> None:
        """合并配置字典，用户配置优先
        
        参数:
            base_config: 基础配置
            user_config: 用户配置
        """
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                self._merge_config(base_config[key], value)
            else:
                # 用户配置覆盖基础配置
                base_config[key] = value
    
    def _init_model(self):
        """初始化YOLO模型"""
        model_type = self.config['model'].get('version', 'yolo11')  # 模型类型
        backbone = self.config['model'].get('backbone', 'small')  # nano/small/medium/large/xlarge
        
        # 模型名称映射表
        # {版本: {大小: 模型文件名}}
        # 注意：每个版本的命名规律略有不同
        model_name_map = {
            'yolo11': {
                'nano': 'yolo11n.pt',      # yolo11n
                'small': 'yolo11s.pt',     # yolo11s
                'medium': 'yolo11m.pt',    # yolo11m
                'large': 'yolo11l.pt',     # yolo11l
                'xlarge': 'yolo11x.pt',    # yolo11x
            },
            'yolo9': {
                'nano': 'yolov9t.pt',      # yolov9t (tiny)
                'small': 'yolov9s.pt',     # yolov9s
                'medium': 'yolov9m.pt',    # yolov9m
                'large': 'yolov9c.pt',     # yolov9c (compact)
                'xlarge': 'yolov9e.pt',    # yolov9e (extra)
            },
            'yolo8': {
                'nano': 'yolov8n.pt',      # yolov8n
                'small': 'yolov8s.pt',     # yolov8s
                'medium': 'yolov8m.pt',    # yolov8m
                'large': 'yolov8l.pt',     # yolov8l
                'xlarge': 'yolov8x.pt',    # yolov8x
            },
            'yolo12': {
                'nano': 'yolo12n.pt',      # yolo12n
                'small': 'yolo12s.pt',     # yolo12s
                'medium': 'yolo12m.pt',    # yolo12m
                'large': 'yolo12l.pt',     # yolo12l
                'xlarge': 'yolo12x.pt',    # yolo12x
            },
        }

        # 版本到目录映射
        version_dir_map = {
            'yolo11': '11',
            'yolo9': '9',
            'yolo8': '8',
            'yolo12': '12',
        }

        # 获取模型文件名和配置文件名
        if model_type not in model_name_map:
            raise ValueError(f"不支持的模型类型: {model_type}。支持的类型: {list(model_name_map.keys())}")
        
        if backbone not in model_name_map[model_type]:
            raise ValueError(f"{model_type} 不支持 {backbone} 大小。支持的大小: {list(model_name_map[model_type].keys())}")
        
        model_file_name = model_name_map[model_type][backbone]
        version_dir = version_dir_map.get(model_type, '11')
        
        # 从对应版本目录加载预训练权重
        yolopt_dir = Path('yolopt') / version_dir
        model_path = yolopt_dir / model_file_name
        if not model_path.exists():
            self.log(f"⚠️  预训练模型不存在: {model_path}")
            self.log(f"💡 请先运行: python download_models.py")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
       
        
        self.model = YOLO(str(model_path))
        self.log(f"✅ 已加载预训练模型: {model_path} ({model_type} - {backbone})")
        
        # 设置为目标检测任务
        self.model.task = 'detect'
    
    def log(self, message: str):
        """记录日志"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def _fix_data_yaml_path(self, data_yaml_path: Path, dataset_root: Path):
        """修复 data.yaml 中的路径，确保 YOLO11 能正确找到图片 
        参数:
            data_yaml_path: data.yaml 文件路径
            dataset_root: 数据集根目录
        """
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # 获取当前的路径配置
            current_path = data_config.get('path', '.')
            
            # 如果 path 不是绝对路径，更新为绝对路径
            path_obj = Path(current_path)
            if not path_obj.is_absolute():
                # 计算相对于 data.yaml 的绝对路径
                abs_path = (dataset_root / current_path).resolve()
                data_config['path'] = str(abs_path)
                
                # 保存更新
                with open(data_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
                
                self.log(f"✅ 已更新 data.yaml 路径为: {abs_path}")
        
        except Exception as e:
            self.log(f"⚠️  修复 data.yaml 路径出错: {e}")
    
    def train(self, dataset_root: str, resume: str = None):
        """训练模型
        
        参数:
            dataset_root: 数据集根目录
            resume: 恢复训练的权重路径
        """
        cfg = self.config
        
        # 训练参数
        epochs = cfg['training']['epochs']
        batch_size = cfg['training']['batch']
        lr = cfg['training']['lr0']
        device = str(self.device).replace('cuda:', '')  # YOLO11 需要 '0' 而不是 'cuda:0'
        
        # 使用已准备好的 data.yaml
        dataset_root = Path(dataset_root).resolve()  # 获取绝对路径
        data_yaml_path = dataset_root / 'data.yaml'
        
        # 验证数据集结构
        if not self._verify_data_yaml(dataset_root):
            return None
        
        # 修复 data.yaml 中的路径（确保 YOLO11 能正确找到图片）
        self._fix_data_yaml_path(data_yaml_path, dataset_root)
        
        self.log(f"\n📚 开始训练:")
        self.log(f"  - 数据集: {dataset_root}")
        self.log(f"  - 迭代数: {epochs}")
        self.log(f"  - 批次大小: {batch_size}")
        self.log(f"  - 学习率: {lr}")
            
        # 使用官方训练接口
        results = self.model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=cfg['training']['imgsz'],
            batch=batch_size,
            device=0 if self.device.type == 'cuda' else 'cpu',
            lr0=lr,
            lrf=0.01,  # 最终学习率
            momentum=0.937,
            weight_decay=cfg['training']['weight_decay'],
            warmup_epochs=cfg['training']['warmup_epochs'],
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=cfg['training']['box'],
            cls=cfg['training']['cls'],
            dfl=cfg['training']['dfl'],
            patience=cfg['training']['patience'],
            close_mosaic=cfg['training']['close_mosaic'],
            project=str(self.output_dir),
            name='yolo11_defect',
            exist_ok=True,
            resume=resume is not None,
            save=True,
            save_period=1,
            seed=42,
            deterministic=True,
            verbose=True,
            amp=cfg['training']['amp'],  # 混合精度
        )
        
        # 保存最优模型到结果目录和数据目录
        best_model_path = self.output_dir / 'yolo11_defect' / 'weights' / 'best.pt'
        if best_model_path.exists():
            import shutil
            
            # 1. 保存到结果目录
            final_path = self.output_dir / 'best.pt'
            shutil.copy(best_model_path, final_path)
            self.log(f"\n✅ 最优模型已保存到: {final_path}")
            
            # 2. 拷贝到数据集目录（方便用户查找）
            dataset_root = Path(dataset_root).resolve()
            data_model_dir = dataset_root / 'models'
            data_model_dir.mkdir(exist_ok=True)
            
            data_model_path = data_model_dir / 'best.pt'
            shutil.copy(best_model_path, data_model_path)
            self.log(f"✅ 最优模型已拷贝到: {data_model_path}")
            
            # 3. 同时拷贝最后一个epoch的模型
            last_model_path = self.output_dir / 'yolo11_defect' / 'weights' / 'last.pt'
            if last_model_path.exists():
                last_copy_path = data_model_dir / 'last.pt'
                shutil.copy(last_model_path, last_copy_path)
                self.log(f"✅ 最后模型已拷贝到: {last_copy_path}")
            
            self.log(f"\n📁 所有模型位置:")
            self.log(f"  - 结果目录: {self.output_dir}")
            self.log(f"  - 数据目录: {data_model_dir}")
        
        return results
    
    def _verify_data_yaml(self, dataset_root: Path) -> bool:
        """验证 data.yaml 和数据集结构
        
        返回:
            验证是否成功
        """
        data_yaml_path = dataset_root / 'data.yaml'
        
        if not data_yaml_path.exists():
            self.log(f"❌ data.yaml 不存在: {data_yaml_path}")
            return False
        
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 验证必要的字段
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data_config:
                self.log(f"❌ data.yaml 缺少字段: {field}")
                return False
        
        # 验证数据集目录
        train_dir = dataset_root / data_config['train']
        val_dir = dataset_root / data_config['val']
        
        if not train_dir.exists():
            self.log(f"❌ 训练数据集目录不存在: {train_dir}")
            return False
        
        if not val_dir.exists():
            self.log(f"❌ 验证数据集目录不存在: {val_dir}")
            return False
        
        self.log(f"✅ data.yaml 和数据集结构验证通过")
        return True

    
    def evaluate(self, dataset_root: str, weights: str = None):
        """在测试集上评估
        
        参数:
            dataset_root: 数据集根目录
            weights: 权重文件路径
        """
        if weights is None:
            weights = self.output_dir / 'best.pt'
        
        if not Path(weights).exists():
            self.log(f"❌ 权重文件不存在: {weights}")
            return
        
        # 加载最优模型
        model = YOLO(str(weights))
        
        data_yaml = Path(dataset_root) / 'data.yaml'
        
        self.log(f"\n📊 开始评估:")
        self.log(f"  - 权重: {weights}")
        self.log(f"  - 数据集: {data_yaml}")
        
        # 评估
        metrics = model.val(data=str(data_yaml), device=0 if self.device.type == 'cuda' else 'cpu')
        
        self.log(f"\n✅ 评估完成!")
        self.log(f"  - mAP@0.5: {metrics.box.map50:.4f}")
        self.log(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
        
        return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='工业缺陷检测训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, required=True, help='数据集根目录')
    parser.add_argument('--output', type=str, default='./results', help='输出目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的权重路径')
    parser.add_argument('--eval', action='store_true', help='仅评估')
    parser.add_argument('--weights', type=str, default=None, help='评估用的权重')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = YoloDetector(args.config, args.output)
    
    if args.eval:
        # 只评估
        trainer.evaluate(args.dataset, args.weights)
    else:
        # 训练
        trainer.train(args.dataset, args.resume)
        # 训练后自动评估
        trainer.evaluate(args.dataset)


if __name__ == '__main__':
    main()
