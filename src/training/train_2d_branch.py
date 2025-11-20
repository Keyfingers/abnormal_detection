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
from detectron2.engine import default_argument_parser, launch, default_setup
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
from train_net import Trainer, setup

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.data.nuscenes_dataset import NuScenesSemanticDataset

# 修复：注册nuScenes数据集（用于Detectron2训练）
def register_nuscenes_semantic_seg(data_root):
    """
    注册nuScenes数据集到Detectron2的DatasetCatalog
    
    修复说明：
    - Detectron2需要特定的数据集格式（dataset_dicts）
    - 需要提供图像路径和语义标签路径
    - 由于nuScenes没有语义标签，我们创建临时标签文件或使用占位符
    """
    from functools import partial
    import tempfile
    import os
    from PIL import Image
    import numpy as np
    
    # 创建临时目录存储占位符标签
    temp_label_dir = os.path.join(data_root, 'temp_semantic_labels')
    os.makedirs(temp_label_dir, exist_ok=True)
    
    def load_nuscenes_semantic(split='train'):
        """加载nuScenes语义分割数据集"""
        dataset = NuScenesSemanticDataset(
            data_root=data_root,
            version=None,  # 自动检测
            split=split,
            image_size=(1024, 2048),
            load_semantic_labels=True,
        )
        
        dataset_dicts = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_token = sample.get('sample_token', f'sample_{idx}')
            
            # 获取图像路径（从nuScenes API）
            # 注意：Detectron2需要文件路径，而不是内存中的图像
            # 这里我们需要保存图像到临时文件或使用nuScenes的原始路径
            try:
                from nuscenes.nuscenes import NuScenes
                nusc = NuScenes(version='v1.0-mini' if os.path.exists(os.path.join(data_root, 'v1.0-mini')) else 'v1.0-trainval', 
                               dataroot=data_root, verbose=False)
                sample_data = nusc.get('sample', sample_token)
                cam_front_data = nusc.get('sample_data', sample_data['data']['CAM_FRONT'])
                image_path = os.path.join(data_root, cam_front_data['filename'])
            except:
                # 如果无法获取路径，创建临时图像文件
                temp_image_dir = os.path.join(data_root, 'temp_images', split)
                os.makedirs(temp_image_dir, exist_ok=True)
                image_path = os.path.join(temp_image_dir, f'{sample_token}.jpg')
                # 保存图像
                image_tensor = sample['image']
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(image_np).save(image_path)
            
            # 创建占位符语义标签文件
            label_path = os.path.join(temp_label_dir, f'{sample_token}_label.png')
            if not os.path.exists(label_path):
                # 创建全0标签（背景类）
                label = np.zeros((1024, 2048), dtype=np.uint8)
                Image.fromarray(label).save(label_path)
            
            record = {
                "file_name": image_path,
                "image_id": idx,
                "sem_seg_file_name": label_path,
            }
            dataset_dicts.append(record)
        
        return dataset_dicts
    
    # 注册训练集和验证集
    DatasetCatalog.register(
        "nuscenes_sem_seg_train",
        partial(load_nuscenes_semantic, split='train')
    )
    MetadataCatalog.get("nuscenes_sem_seg_train").set(
        evaluator_type="sem_seg",
        ignore_label=255,
        stuff_classes=[
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle",
        ],
    )
    
    DatasetCatalog.register(
        "nuscenes_sem_seg_val",
        partial(load_nuscenes_semantic, split='val')
    )
    MetadataCatalog.get("nuscenes_sem_seg_val").set(
        evaluator_type="sem_seg",
        ignore_label=255,
        stuff_classes=[
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle",
        ],
    )
    print("Registered nuScenes dataset for semantic segmentation")


def train_main(args):
    """训练主函数，被launch调用"""
    # 注册nuScenes数据集
    print("Registering nuScenes dataset...")
    register_nuscenes_semantic_seg(args.data_root)
    
    # 设置配置
    cfg = setup(args)
    
    print("=" * 50)
    print("Training 2D Semantic Branch (Mask2Former)")
    print("=" * 50)
    print(f"Config: {args.config_file}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {cfg.OUTPUT_DIR}")
    print(f"Resume: {args.resume}")
    print("=" * 50)
    
    # 创建训练器并开始训练
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    print("Trainable Params:")
    for n, m in trainer._trainer.model.named_parameters():
        if m.requires_grad:
            print(n)
    return trainer.train()


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
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the last checkpoint')
    parser.add_argument('--pretrained-weights', type=str, default=None,
                        help='Path to pretrained model weights (overrides config MODEL.WEIGHTS)')
    
    # 解析参数
    args, unknown = parser.parse_known_args()
    
    # 创建Detectron2风格的参数对象
    detectron2_parser = default_argument_parser()
    detectron2_cmd_args = [
        '--config-file', args.config_file,
        '--num-gpus', str(args.num_gpus),
    ]
    if args.resume:
        detectron2_cmd_args.append('--resume')
    # OUTPUT_DIR作为配置覆盖项传递
    detectron2_cmd_args.extend(['OUTPUT_DIR', args.output_dir])
    
    # 修复：如果指定了预训练权重，覆盖配置文件中的MODEL.WEIGHTS
    if args.pretrained_weights:
        detectron2_cmd_args.extend(['MODEL.WEIGHTS', args.pretrained_weights])
    
    detectron2_args = detectron2_parser.parse_args(detectron2_cmd_args)
    
    # 添加data_root属性
    detectron2_args.data_root = args.data_root
    
    # 启动训练
    launch(
        train_main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url='tcp://127.0.0.1:23456',
        args=(detectron2_args,),
    )


if __name__ == '__main__':
    main()

