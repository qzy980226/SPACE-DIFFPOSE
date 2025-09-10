#!/usr/bin/env python3
"""
SPEED+数据集训练脚本（支持动态mask）
"""

import argparse
import yaml
import os
import sys
import logging
import torch

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from runners.diffpose_frame import Diffpose

def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 转换为namespace对象，便于访问
    class ConfigNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    def dict_to_namespace(d):
        if isinstance(d, dict):
            ns = ConfigNamespace()
            for k, v in d.items():
                setattr(ns, k, dict_to_namespace(v))
            return ns
        return d
    
    return dict_to_namespace(config)

def validate_config(config):
    """验证配置文件"""
    required_fields = [
        'data.dataset',
        'data.json_path', 
        'data.keypoints_path',
        'model.hid_dim',
        'training.epochs',
        'training.lr'
    ]
    
    for field in required_fields:
        parts = field.split('.')
        obj = config
        try:
            for part in parts:
                obj = getattr(obj, part)
        except AttributeError:
            raise ValueError(f"Missing required config field: {field}")
    
    # 检查数据文件是否存在
    if not os.path.exists(config.data.json_path):
        raise FileNotFoundError(f"JSON file not found: {config.data.json_path}")
    
    if not os.path.exists(config.data.keypoints_path):
        raise FileNotFoundError(f"Keypoints file not found: {config.data.keypoints_path}")
    
    print("✓ 配置验证通过")

def create_args_namespace(config):
    """创建args命名空间（兼容原有代码）"""
    args = argparse.Namespace()
    
    # 设置默认args
    args.config = config
    args.resume = None
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.doc = getattr(config, 'experiment', {}).get('name', 'speedplus_experiment')
    
    return args

def main():
    parser = argparse.ArgumentParser(description='Train SPEED+ with dynamic mask support')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logs')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载和验证配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # 设置日志
    log_dir = os.path.join(args.log_dir, config.data.dataset)
    setup_logging(log_dir)
    
    logging.info("=" * 50)
    logging.info("开始SPEED+训练（支持动态mask）")
    logging.info("=" * 50)
    logging.info(f"配置文件: {args.config}")
    logging.info(f"数据集: {config.data.dataset}")
    logging.info(f"JSON路径: {config.data.json_path}")
    logging.info(f"关键点模板: {config.data.keypoints_path}")
    logging.info(f"设备: {device}")
    
    # 打印关键配置
    logging.info(f"模型配置:")
    logging.info(f"  隐藏维度: {config.model.hid_dim}")
    logging.info(f"  关键点数: {config.model.n_pts}")
    logging.info(f"  mask策略: {getattr(config.model, 'mask_strategy', 'static')}")
    
    logging.info(f"训练配置:")
    logging.info(f"  训练轮数: {config.training.epochs}")
    logging.info(f"  学习率: {config.training.lr}")
    logging.info(f"  批次大小: {getattr(config.training, 'batch_size', config.data.batch_size)}")
    logging.info(f"  严格mask: {getattr(config.training, 'use_strict_masking', True)}")
    
    try:
        # 创建训练器
        args_ns = create_args_namespace(config)
        trainer = Diffpose(args_ns, config, device=device)
        
        # 开始训练
        logging.info("开始训练...")
        trainer.train()
        
        logging.info("训练完成！")
        
    except KeyboardInterrupt:
        logging.info("训练被用户中断")
    except Exception as e:
        logging.error(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()