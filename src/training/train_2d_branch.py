"""
训练2D语义分支（阶段二）
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../RbA'))
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch
from detectron2.data import build_detection_train_loader
from mask2former import add_maskformer2_config
from train_net import Trainer

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.data.nuscenes_dataset import NuScenesSemanticDataset


def main():
    parser = argparse.ArgumentParser(description='Train 2D Semantic Branch')
    parser.add_argument('--config-file', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--output-dir', type=str, default='outputs/semantic_2d',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # 使用Detectron2的训练框架
    # 这里需要配置Mask2Former的训练脚本
    # 实际使用时，可以直接使用RbA项目中的train_net.py
    
    print("=" * 50)
    print("Training 2D Semantic Branch (Mask2Former)")
    print("=" * 50)
    print(f"Config: {args.config_file}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 50)
    
    # 构建参数列表（Detectron2格式）
    detectron2_args = [
        '--config-file', args.config_file,
        '--num-gpus', str(args.num_gpus),
        'OUTPUT_DIR', args.output_dir,
    ]
    
    # 启动训练（使用Detectron2的launch）
    launch(
        Trainer.train,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url='tcp://127.0.0.1:23456',
        args=(detectron2_args,),
    )


if __name__ == '__main__':
    main()

