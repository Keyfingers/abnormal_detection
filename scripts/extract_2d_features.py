"""
提取2D语义特征脚本

根据项目规则，阶段二应该：
- 使用预训练的Mask2Former模型（在Cityscapes上训练）
- 在nuScenes上提取特征图，而不是训练
- 提取Pixel Decoder的2D特征图用于融合
- 计算RbA异常评分作为基线
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.semantic_2d import Semantic2DBranch
from src.data.nuscenes_dataset import NuScenesDataset


def extract_features(
    model: Semantic2DBranch,
    dataloader: DataLoader,
    output_dir: str,
    device: torch.device,
    save_features: bool = True,
    save_rba_scores: bool = True,
):
    """
    提取2D特征图和RbA评分
    
    Args:
        model: 预训练的Semantic2DBranch模型
        dataloader: 数据加载器
        output_dir: 输出目录
        device: 设备
        save_features: 是否保存特征图
        save_rba_scores: 是否保存RbA评分
    """
    model.eval()
    model.to(device)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    features_dir = os.path.join(output_dir, 'features_2d')
    rba_dir = os.path.join(output_dir, 'rba_scores')
    
    if save_features:
        os.makedirs(features_dir, exist_ok=True)
    if save_rba_scores:
        os.makedirs(rba_dir, exist_ok=True)
    
    # 统计信息
    all_features = []
    all_rba_scores = []
    sample_tokens = []
    
    # 检查已存在的文件（断点续传）
    existing_features = set()
    existing_rba = set()
    if save_features and os.path.exists(features_dir):
        existing_features = set([f.replace('.npy', '') for f in os.listdir(features_dir) if f.endswith('.npy')])
    if save_rba_scores and os.path.exists(rba_dir):
        existing_rba = set([f.replace('.npy', '') for f in os.listdir(rba_dir) if f.endswith('.npy')])
    
    skipped_count = 0
    total_batches = len(dataloader)
    
    print(f"开始提取特征，共 {total_batches} 个batch...")
    if existing_features or existing_rba:
        print(f"发现已存在的文件：特征图 {len(existing_features)} 个，RbA评分 {len(existing_rba)} 个")
        print("将跳过已存在的样本（断点续传）")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="提取特征")):
            images = batch['image'].to(device)  # (B, 3, H, W)
            batch_tokens = batch['sample_token']
            
            # 检查哪些样本需要处理
            need_process = []
            batch_indices = []
            
            for i, token in enumerate(batch_tokens):
                need_feat = save_features and token not in existing_features
                need_rba_score = save_rba_scores and token not in existing_rba
                
                if need_feat or need_rba_score:
                    need_process.append({
                        'index': i,
                        'token': token,
                        'need_feat': need_feat,
                        'need_rba': need_rba_score,
                    })
                    batch_indices.append(i)
                else:
                    skipped_count += 1
            
            # 如果所有样本都已存在，跳过这个batch
            if not batch_indices:
                sample_tokens.extend(batch_tokens)
                continue
            
            # 只处理需要的样本
            if len(batch_indices) < len(batch_tokens):
                # 需要过滤batch
                filtered_images = images[batch_indices]
            else:
                filtered_images = images
            
            # 前向传播
            try:
                results = model(
                    filtered_images,
                    return_features=save_features,
                    return_rba_score=save_rba_scores,
                )
            except Exception as e:
                print(f"\n警告: batch {batch_idx} 前向传播失败: {e}")
                print(f"跳过的token: {[item['token'] for item in need_process]}")
                continue
            
            # 保存特征图
            if save_features and 'features_2d' in results:
                features_2d = results['features_2d']  # (B, C_2D, H, W)
                for proc_idx, item in enumerate(need_process):
                    if item['need_feat']:
                        token = item['token']
                        feature_path = os.path.join(features_dir, f"{token}.npy")
                        try:
                            # 使用临时文件确保原子写入
                            # np.save会自动添加.npy扩展名，所以临时文件应该是xxx.tmp
                            temp_path = os.path.join(features_dir, f"{token}.tmp")
                            np.save(temp_path, features_2d[proc_idx].cpu().numpy())
                            # np.save创建的文件是xxx.tmp.npy，需要重命名为xxx.npy
                            temp_npy_path = temp_path + '.npy'
                            os.rename(temp_npy_path, feature_path)
                            all_features.append(feature_path)
                        except Exception as e:
                            print(f"\n警告: 保存特征图失败 {token}: {e}")
                            temp_npy_path = temp_path + '.npy'
                            if os.path.exists(temp_npy_path):
                                os.remove(temp_npy_path)
            
            # 保存RbA评分
            if save_rba_scores and 'rba_score' in results:
                rba_scores = results['rba_score']  # (B, H, W)
                for proc_idx, item in enumerate(need_process):
                    if item['need_rba']:
                        token = item['token']
                        rba_path = os.path.join(rba_dir, f"{token}.npy")
                        try:
                            # 使用临时文件确保原子写入
                            # np.save会自动添加.npy扩展名，所以临时文件应该是xxx.tmp
                            temp_path = os.path.join(rba_dir, f"{token}.tmp")
                            np.save(temp_path, rba_scores[proc_idx].cpu().numpy())
                            # np.save创建的文件是xxx.tmp.npy，需要重命名为xxx.npy
                            temp_npy_path = temp_path + '.npy'
                            os.rename(temp_npy_path, rba_path)
                            all_rba_scores.append(rba_path)
                        except Exception as e:
                            print(f"\n警告: 保存RbA评分失败 {token}: {e}")
                            temp_npy_path = temp_path + '.npy'
                            if os.path.exists(temp_npy_path):
                                os.remove(temp_npy_path)
            
            sample_tokens.extend(batch_tokens)
    
    # 保存样本token列表
    token_list_path = os.path.join(output_dir, 'sample_tokens.txt')
    with open(token_list_path, 'w') as f:
        for token in sample_tokens:
            f.write(f"{token}\n")
    
    print(f"\n提取完成！")
    print(f"输出目录: {output_dir}")
    print(f"本次提取特征图数量: {len(all_features) if save_features else 0}")
    print(f"本次提取RbA评分数量: {len(all_rba_scores) if save_rba_scores else 0}")
    print(f"跳过的样本数量: {skipped_count}")
    print(f"样本总数: {len(sample_tokens)}")
    
    # 统计最终文件数量
    if save_features and os.path.exists(features_dir):
        final_feature_count = len([f for f in os.listdir(features_dir) if f.endswith('.npy')])
        print(f"最终特征图总数: {final_feature_count}")
    if save_rba_scores and os.path.exists(rba_dir):
        final_rba_count = len([f for f in os.listdir(rba_dir) if f.endswith('.npy')])
        print(f"最终RbA评分总数: {final_rba_count}")
    
    # 打印特征图统计信息
    if save_features and len(all_features) > 0:
        sample_feature = np.load(all_features[0])
        print(f"\n特征图形状: {sample_feature.shape}")
        print(f"特征图数据类型: {sample_feature.dtype}")
        print(f"特征图值范围: [{sample_feature.min():.4f}, {sample_feature.max():.4f}]")
    
    # 打印RbA评分统计信息
    if save_rba_scores and len(all_rba_scores) > 0:
        sample_rba = np.load(all_rba_scores[0])
        print(f"\nRbA评分形状: {sample_rba.shape}")
        print(f"RbA评分值范围: [{sample_rba.min():.4f}, {sample_rba.max():.4f}]")
        print(f"RbA评分均值: {sample_rba.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description='提取2D语义特征')
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Mask2Former配置文件路径'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default='RbA/ckpts/swin_b_1dl/swin_b_1dl/model_final.pth',
        help='预训练模型权重路径'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='nuScenes数据根目录'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='数据集分割'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/2d_features',
        help='输出目录'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='批次大小'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='数据加载器工作进程数'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[1024, 2048],
        help='图像尺寸 (H, W)'
    )
    parser.add_argument(
        '--no-features',
        action='store_true',
        help='不保存特征图（只保存RbA评分）'
    )
    parser.add_argument(
        '--no-rba',
        action='store_true',
        help='不保存RbA评分（只保存特征图）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='设备'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("2D语义特征提取")
    print("=" * 60)
    print(f"配置文件: {args.config_file}")
    print(f"模型权重: {args.checkpoint_path}")
    print(f"数据根目录: {args.data_root}")
    print(f"数据集分割: {args.split}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像尺寸: {args.image_size}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"配置文件不存在: {args.config_file}")
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"模型权重不存在: {args.checkpoint_path}")
    
    # 设置设备
    device = torch.device(args.device)
    
    # 加载模型
    print("\n加载预训练模型...")
    # 如果配置文件不存在或有问题，直接使用模型目录下的配置文件
    if not os.path.exists(args.config_file):
        # 使用模型目录下的配置文件
        model_config_path = os.path.join(os.path.dirname(args.checkpoint_path), 'config.yaml')
        if os.path.exists(model_config_path):
            print(f"使用模型目录下的配置文件: {model_config_path}")
            config_path = model_config_path
        else:
            raise FileNotFoundError(f"配置文件不存在: {args.config_file}")
    else:
        config_path = args.config_file
    
    model = Semantic2DBranch(
        config_path=config_path,
        checkpoint_path=args.checkpoint_path,
        freeze_backbone=True,  # 冻结backbone，只用于特征提取
    )
    print("模型加载完成！")
    
    # 创建数据集
    print(f"\n创建数据集 ({args.split})...")
    dataset = NuScenesDataset(
        data_root=args.data_root,
        split=args.split,
        image_size=tuple(args.image_size),
        load_lidar=False,  # 2D特征提取不需要点云
        load_semantic_labels=False,  # 不需要标签
    )
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    def collate_fn(batch):
        """自定义collate函数"""
        images = torch.stack([item['image'] for item in batch])
        tokens = [item['sample_token'] for item in batch]
        return {
            'image': images,
            'sample_token': tokens,
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # 提取特征
    extract_features(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        device=device,
        save_features=not args.no_features,
        save_rba_scores=not args.no_rba,
    )
    
    print("\n完成！")


if __name__ == '__main__':
    main()

