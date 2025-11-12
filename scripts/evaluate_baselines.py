"""
评估基线模型（基线1：2D-only，基线2：3D-only）
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
from src.data.anovox_dataset import AnoVoxDataset
from src.utils.metrics import compute_metrics


def collate_fn(batch):
    """自定义collate函数"""
    images = torch.stack([item['image'] for item in batch])
    point_clouds = [item['point_cloud'] for item in batch]
    anomaly_masks = torch.stack([item['anomaly_mask'] for item in batch])
    
    return {
        'images': images,
        'point_clouds': point_clouds,
        'anomaly_masks': anomaly_masks,
    }


def evaluate_2d_baseline(
    model: Semantic2DBranch,
    dataloader: DataLoader,
    device: torch.device,
):
    """评估2D基线（RbA评分）"""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating 2D Baseline'):
            images = batch['images'].to(device)
            anomaly_masks = batch['anomaly_masks'].numpy()
            
            # 获取RbA评分
            rba_scores = model.get_rba_score(images)  # (B, H, W)
            
            # 调整尺寸匹配
            if rba_scores.shape[-2:] != anomaly_masks.shape[-2:]:
                rba_scores = torch.nn.functional.interpolate(
                    rba_scores.unsqueeze(1),
                    size=anomaly_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)
            
            # 收集结果
            rba_scores_np = rba_scores.cpu().numpy()
            all_scores.append(rba_scores_np.flatten())
            all_labels.append(anomaly_masks.flatten())
    
    # 合并所有结果
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    # 归一化分数到[0, 1]
    all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-8)
    
    # 计算指标
    metrics = compute_metrics(all_scores, all_labels)
    
    return metrics


def evaluate_3d_baseline(
    model: Geometric3DBranch,
    dataloader: DataLoader,
    device: torch.device,
):
    """评估3D基线（重建误差）"""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating 3D Baseline'):
            point_clouds = batch['point_clouds']
            anomaly_masks = batch['anomaly_masks'].numpy()
            
            # 过滤空点云
            valid_indices = [i for i, pc in enumerate(point_clouds) if len(pc) > 0]
            if len(valid_indices) == 0:
                continue
            
            valid_pcs = [point_clouds[i] for i in valid_indices]
            valid_masks = anomaly_masks[valid_indices]
            
            # 获取重建误差
            reconstruction_errors = model.get_reconstruction_error(valid_pcs)
            
            # 重建误差是每个点的误差，需要投影回2D图像
            # 这里简化处理：使用平均误差作为异常分数
            # 实际应用中需要更精确的投影
            batch_scores = []
            for i, error in enumerate(reconstruction_errors):
                # 简化：使用平均误差
                avg_error = error.mean().item()
                # 创建一个与mask相同尺寸的分数图
                score_map = np.full(valid_masks[i].shape, avg_error)
                batch_scores.append(score_map)
            
            # 收集结果
            for score_map, mask in zip(batch_scores, valid_masks):
                all_scores.append(score_map.flatten())
                all_labels.append(mask.flatten())
    
    # 合并所有结果
    if len(all_scores) > 0:
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        # 归一化分数
        all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-8)
        
        # 计算指标
        metrics = compute_metrics(all_scores, all_labels)
    else:
        metrics = {'auroc': 0.0, 'ap': 0.0, 'fpr_at_95_tpr': 1.0}
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Baseline Models')
    parser.add_argument('--model-type', type=str, choices=['2d', '3d'], required=True,
                        help='Baseline type: 2d (RbA) or 3d (reconstruction error)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (required for 2D model)')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to AnoVox dataset root')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
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
    
    # 评估
    print("=" * 50)
    if args.model_type == '2d':
        print("Evaluating 2D Baseline (RbA)")
        print("=" * 50)
        
        if args.config is None:
            raise ValueError("--config is required for 2D model")
        
        model = Semantic2DBranch(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            freeze_backbone=True,
        )
        model = model.to(device)
        
        metrics = evaluate_2d_baseline(model, test_loader, device)
        
    else:  # 3d
        print("Evaluating 3D Baseline (Reconstruction Error)")
        print("=" * 50)
        
        model = Geometric3DBranch(
            checkpoint_path=args.checkpoint,
            freeze_backbone=True,
        )
        model = model.to(device)
        
        metrics = evaluate_3d_baseline(model, test_loader, device)
    
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

