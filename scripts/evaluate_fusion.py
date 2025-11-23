"""
评估融合模型及基线性能
计算 FPR95, AUPR, AUROC 指标
"""
import argparse
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.semantic_2d import Semantic2DBranch
from src.models.geometric_3d import Geometric3DBranch
from src.models.fusion import FusionHead, FusionModel
from src.data.anovox_normality_dataset import AnoVoxNormalityDataset
from src.data.anovox_dataset import AnoVoxDataset
from src.data.anovox_anomaly_dataset import AnoVoxAnomalyDataset

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

def compute_metrics(scores, labels):
    """
    计算 FPR95, AUPR, AUROC
    args:
        scores: (N,) numpy array, 异常分数
        labels: (N,) numpy array, 异常标签 (0/1)
    """
    # 确保是平坦的
    scores = scores.flatten()
    labels = labels.flatten()
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    
    # AUPR
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)
    
    # FPR95
    # FPR at 95% TPR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # 找到TPR >= 0.95的第一个索引
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        fpr95 = fpr[idx[0]]
    else:
        fpr95 = 1.0
        
    return {
        'AUROC': auroc,
        'AUPR': aupr,
        'FPR95': fpr95
    }

def evaluate(model, dataloader, device):
    model.eval()
    
    # 存储所有预测和标签
    # 为了节省内存，我们可以分批计算或者只存储采样点
    # 这里尝试存储所有点，如果OOM则需要优化
    
    fusion_scores_list = []
    rba_scores_list = []
    recon_errors_list = []
    labels_list = []
    
    print("开始评估...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['images'].to(device)
            point_clouds = batch['point_clouds']
            anomaly_masks = batch['anomaly_masks'].to(device) # (B, H, W)
            camera_intrinsics = batch['camera_intrinsics'].to(device)
            camera_extrinsics = batch['camera_extrinsics'].to(device)
            
            valid_indices = [i for i, pc in enumerate(point_clouds) if len(pc) > 0]
            if len(valid_indices) == 0:
                continue
            
            images = images[valid_indices]
            point_clouds = [point_clouds[i] for i in valid_indices]
            anomaly_masks = anomaly_masks[valid_indices]
            camera_intrinsics = camera_intrinsics[valid_indices]
            camera_extrinsics = camera_extrinsics[valid_indices]
            
            # 前向传播，请求所有分数
            results = model(
                images=images,
                point_clouds=point_clouds,
                camera_intrinsic=camera_intrinsics,
                camera_extrinsic=camera_extrinsics,
                return_individual_scores=True # 关键：获取单独分数
            )
            
            # 获取分数并调整尺寸
            fusion_score = results['fusion_score'] # (B, H, W)
            rba_score = results['rba_score'] # (B, H, W)
            
            # 重建误差通常是基于点的，但也可能在模型中被投影到了2D
            # Geometric3DBranch.forward 返回的 reconstruction_error 是基于点的 (B, N)
            # 但是 FusionModel.forward 并没有处理重建误差的投影
            # 我们需要检查 FusionModel 的实现
            # 如果 FusionModel 直接返回了 geometric_results['reconstruction_error']，那是点云误差
            
            # 在当前FusionModel实现中，results['reconstruction_error']是点云误差
            # 我们无法直接与2D mask比较，除非投影
            # 为了简化，我们只评估 2D 相关的 Fusion 和 RbA
            # 如果要评估 3D Reconstruction Error 在 2D 上的表现，需要投影
            
            # 投影重建误差到2D (类似于特征投影)
            # 这里我们暂时只评估 Fusion 和 RbA，或者手动投影误差
            # 为了完整性，我们跳过 3D error 的 2D 评估，或者假设 3D error 已经被投影
            
            # 统一尺寸到 anomaly_masks
            if fusion_score.shape[-2:] != anomaly_masks.shape[-2:]:
                fusion_score = torch.nn.functional.interpolate(
                    fusion_score.unsqueeze(1), size=anomaly_masks.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(1)
                
            if rba_score.shape[-2:] != anomaly_masks.shape[-2:]:
                rba_score = torch.nn.functional.interpolate(
                    rba_score.unsqueeze(1), size=anomaly_masks.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(1)
            
            # 保存结果 (转为cpu numpy)
            # 展平以节省维度管理
            # 内存优化：降采样并使用低精度
            stride = 4
            fusion_scores_list.append(fusion_score[:, ::stride, ::stride].cpu().numpy().flatten().astype(np.float16))
            rba_scores_list.append(rba_score[:, ::stride, ::stride].cpu().numpy().flatten().astype(np.float16))
            labels_list.append(anomaly_masks[:, ::stride, ::stride].cpu().numpy().flatten().astype(np.int8))
            
    # 拼接
    print("拼接数据...")
    all_fusion_scores = np.concatenate(fusion_scores_list).astype(np.float32)
    all_rba_scores = np.concatenate(rba_scores_list).astype(np.float32)
    all_labels = np.concatenate(labels_list)
    
    # 确保标签是二值的
    all_labels = (all_labels > 0.5).astype(np.int32)
    
    print(f"评估样本总像素数: {len(all_labels)}")
    print(f"异常像素比例: {all_labels.sum() / len(all_labels):.4f}")
    
    # 计算指标
    print("\n计算 Fusion Model 指标...")
    fusion_metrics = compute_metrics(all_fusion_scores, all_labels)
    print(f"Fusion: {fusion_metrics}")
    
    print("\n计算 2D RbA 指标...")
    rba_metrics = compute_metrics(all_rba_scores, all_labels)
    print(f"RbA (2D Only): {rba_metrics}")
    
    return fusion_metrics, rba_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate Fusion Model')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to AnoVox dataset root')
    parser.add_argument('--fusion-ckpt', type=str, required=True,
                        help='Path to fusion model checkpoint (model_best.pth)')
    parser.add_argument('--semantic-config', type=str, required=True,
                        help='Path to semantic 2D branch config')
    # 下面这些参数用于重建模型架构，必须与训练时一致
    parser.add_argument('--semantic-ckpt', type=str, required=True)
    parser.add_argument('--geometric-ckpt', type=str, required=True)
    parser.add_argument('--feature-2d-dim', type=int, default=256)
    parser.add_argument('--feature-3d-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集 (只加载测试集)
    normality_dir = os.path.join(args.data_root, 'AnoVox_Normality_Mono_Town03')
    
    if 'Static' in args.data_root or 'Dynamic' in args.data_root:
        print(f"Using AnoVoxAnomalyDataset with root: {args.data_root}")
        test_dataset = AnoVoxAnomalyDataset(
            data_root=args.data_root,
            split='test',
            image_size=(512, 1024)
        )
    elif os.path.exists(normality_dir) or 'Normality' in args.data_root:
        print(f"Using AnoVoxNormalityDataset with root: {args.data_root}")
        test_dataset = AnoVoxNormalityDataset(
            data_root=args.data_root,
            split='test',
        )
    else:
        print(f"Using AnoVoxDataset with root: {args.data_root}")
        test_dataset = AnoVoxDataset(
            data_root=args.data_root,
            split='test',
        )
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 构建模型
    print("Building models...")
    semantic_2d = Semantic2DBranch(
        config_path=args.semantic_config,
        checkpoint_path=args.semantic_ckpt,
        freeze_backbone=True,
    )
    
    geometric_3d = Geometric3DBranch(
        checkpoint_path=args.geometric_ckpt,
        freeze_backbone=True,
        feature_dim=args.feature_3d_dim,
    )
    
    fusion_head = FusionHead(
        feature_2d_dim=args.feature_2d_dim,
        feature_3d_dim=args.feature_3d_dim,
    )
    
    model = FusionModel(
        semantic_2d_model=semantic_2d,
        geometric_3d_model=geometric_3d,
        fusion_head=fusion_head,
        freeze_2d=True,
        freeze_3d=True,
    )
    
    # 加载融合权重
    print(f"Loading fusion checkpoint from {args.fusion_ckpt}...")
    checkpoint = torch.load(args.fusion_ckpt, map_location='cpu')
    # 兼容不同的保存格式
    if 'fusion_head_state_dict' in checkpoint:
        model.fusion_head.load_state_dict(checkpoint['fusion_head_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    
    # 评估
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
