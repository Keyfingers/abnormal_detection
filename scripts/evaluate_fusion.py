"""
评估融合模型
"""
import argparse
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.semantic_2d import Semantic2DBranch
from src.models.geometric_3d import Geometric3DBranch
from src.models.fusion import FusionHead, FusionModel
from src.data.anovox_dataset import AnoVoxDataset
from src.utils.metrics import compute_metrics


def collate_fn(batch):
    """自定义collate函数"""
    images = torch.stack([item['image'] for item in batch])
    point_clouds = [item['point_cloud'] for item in batch]
    anomaly_masks = torch.stack([item['anomaly_mask'] for item in batch])
    camera_intrinsics = torch.stack([item['camera_intrinsic'] for item in batch])
    camera_extrinsics = torch.stack([item['camera_extrinsic'] for item in batch])
    
    return {
        'images': images,
        'point_clouds': point_clouds,
        'anomaly_masks': anomaly_masks,
        'camera_intrinsics': camera_intrinsics,
        'camera_extrinsics': camera_extrinsics,
    }


def evaluate_fusion(
    model: FusionModel,
    dataloader: DataLoader,
    device: torch.device,
):
    """评估融合模型"""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating Fusion Model'):
            images = batch['images'].to(device)
            point_clouds = batch['point_clouds']
            anomaly_masks = batch['anomaly_masks'].numpy()
            camera_intrinsics = batch['camera_intrinsics'].to(device)
            camera_extrinsics = batch['camera_extrinsics'].to(device)
            
            # 过滤空点云
            valid_indices = [i for i, pc in enumerate(point_clouds) if len(pc) > 0]
            if len(valid_indices) == 0:
                continue
            
            images = images[valid_indices]
            point_clouds = [point_clouds[i] for i in valid_indices]
            anomaly_masks = anomaly_masks[valid_indices]
            camera_intrinsics = camera_intrinsics[valid_indices]
            camera_extrinsics = camera_extrinsics[valid_indices]
            
            # 前向传播
            results = model(
                images=images,
                point_clouds=point_clouds,
                camera_intrinsic=camera_intrinsics,
                camera_extrinsic=camera_extrinsics,
                return_individual_scores=False,
            )
            
            fusion_scores = results['fusion_score']  # (B, H, W)
            
            # 调整尺寸匹配
            if fusion_scores.shape[-2:] != anomaly_masks.shape[-2:]:
                fusion_scores = torch.nn.functional.interpolate(
                    fusion_scores.unsqueeze(1),
                    size=anomaly_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)
            
            # 归一化到[0, 1]
            fusion_scores_normalized = torch.sigmoid(fusion_scores)
            
            # 收集结果
            fusion_scores_np = fusion_scores_normalized.cpu().numpy()
            all_scores.append(fusion_scores_np.flatten())
            all_labels.append(anomaly_masks.flatten())
    
    # 合并所有结果
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    # 计算指标
    metrics = compute_metrics(all_scores, all_labels)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fusion Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fusion model checkpoint')
    parser.add_argument('--semantic-ckpt', type=str, required=True,
                        help='Path to semantic 2D branch checkpoint')
    parser.add_argument('--geometric-ckpt', type=str, required=True,
                        help='Path to geometric 3D branch checkpoint')
    parser.add_argument('--semantic-config', type=str, required=True,
                        help='Path to semantic 2D branch config')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to AnoVox dataset root')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--feature-2d-dim', type=int, default=256,
                        help='2D feature dimension')
    parser.add_argument('--feature-3d-dim', type=int, default=128,
                        help='3D feature dimension')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集
    print("Loading dataset...")
    test_dataset = AnoVoxDataset(
        data_root=args.data_root,
        split='test',
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # 构建模型
    print("Building model...")
    
    # 2D语义分支（冻结）
    semantic_2d = Semantic2DBranch(
        config_path=args.semantic_config,
        checkpoint_path=args.semantic_ckpt,
        freeze_backbone=True,
    )
    
    # 3D几何分支（冻结）
    geometric_3d = Geometric3DBranch(
        checkpoint_path=args.geometric_ckpt,
        freeze_backbone=True,
        feature_dim=args.feature_3d_dim,
    )
    
    # 融合头
    fusion_head = FusionHead(
        feature_2d_dim=args.feature_2d_dim,
        feature_3d_dim=args.feature_3d_dim,
    )
    
    # 加载融合头权重
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'fusion_head_state_dict' in checkpoint:
        fusion_head.load_state_dict(checkpoint['fusion_head_state_dict'])
    else:
        fusion_head.load_state_dict(checkpoint)
    
    # 完整融合模型
    model = FusionModel(
        semantic_2d_model=semantic_2d,
        geometric_3d_model=geometric_3d,
        fusion_head=fusion_head,
        freeze_2d=True,
        freeze_3d=True,
    )
    model = model.to(device)
    
    # 评估
    print("=" * 50)
    print("Evaluating Fusion Model")
    print("=" * 50)
    
    metrics = evaluate_fusion(model, test_loader, device)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AP: {metrics['ap']:.4f}")
    print(f"FPR@95: {metrics['fpr_at_95_tpr']:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()

