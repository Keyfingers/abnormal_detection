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
    
    # 修复：添加相机参数（用于3D基线评估的投影）
    camera_intrinsics = torch.stack([item.get('camera_intrinsic', torch.eye(3)) for item in batch])
    camera_extrinsics = torch.stack([item.get('camera_extrinsic', torch.eye(4)) for item in batch])
    
    return {
        'images': images,
        'point_clouds': point_clouds,
        'anomaly_masks': anomaly_masks,
        'camera_intrinsics': camera_intrinsics,
        'camera_extrinsics': camera_extrinsics,
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
    """
    评估3D基线（重建误差）
    
    修复说明：
    - 规则要求：重建误差需要投影回2D图像坐标，而不是使用平均误差
    - 需要将每个点的重建误差投影到对应的2D像素位置
    - 这需要相机标定参数（内参和外参）
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating 3D Baseline'):
            point_clouds = batch['point_clouds']
            anomaly_masks = batch['anomaly_masks'].numpy()
            
            # 修复：需要获取相机标定参数用于投影
            # 如果数据加载器提供了相机参数，使用它们
            camera_intrinsics = batch.get('camera_intrinsics', None)
            camera_extrinsics = batch.get('camera_extrinsics', None)
            
            # 过滤空点云
            valid_indices = [i for i, pc in enumerate(point_clouds) if len(pc) > 0]
            if len(valid_indices) == 0:
                continue
            
            valid_pcs = [point_clouds[i] for i in valid_indices]
            valid_masks = anomaly_masks[valid_indices]
            
            # 获取重建误差和3D特征（用于投影）
            results = model(
                valid_pcs,
                return_features=True,
                return_reconstruction_error=True,
                apply_mask=False,  # 评估时不使用mask
            )
            
            reconstruction_errors = results['reconstruction_error']  # 每个点的误差
            features_3d = results['features_3d']  # 3D特征（稀疏张量）
            coords_3d = results['coords_list']  # 3D坐标列表
            
            # 修复：将重建误差投影回2D图像坐标
            # 需要将每个点的误差投影到对应的2D像素位置
            batch_scores = []
            for i, (pc, error, coords) in enumerate(zip(valid_pcs, reconstruction_errors, coords_3d)):
                mask_shape = valid_masks[i].shape
                H, W = mask_shape
                
                # 初始化2D误差图
                error_map_2d = np.zeros((H, W), dtype=np.float32)
                weight_map_2d = np.zeros((H, W), dtype=np.float32)
                
                # 获取相机参数（如果有）
                if camera_intrinsics is not None:
                    cam_intrinsic = camera_intrinsics[i].cpu().numpy() if isinstance(camera_intrinsics, torch.Tensor) else camera_intrinsics[i]
                else:
                    # 使用默认相机参数
                    cam_intrinsic = np.array([
                        [1000, 0, W / 2],
                        [0, 1000, H / 2],
                        [0, 0, 1],
                    ])
                
                if camera_extrinsics is not None:
                    cam_extrinsic = camera_extrinsics[i].cpu().numpy() if isinstance(camera_extrinsics, torch.Tensor) else camera_extrinsics[i]
                else:
                    cam_extrinsic = None
                
                # 将3D坐标投影到2D
                coords_3d_np = coords  # (N, 3) 原始坐标
                error_np = error.cpu().numpy() if isinstance(error, torch.Tensor) else error
                
                # 应用外参变换（如果提供）
                if cam_extrinsic is not None:
                    coords_homo = np.column_stack([coords_3d_np, np.ones(len(coords_3d_np))])
                    coords_camera = (cam_extrinsic @ coords_homo.T).T[:, :3]
                else:
                    coords_camera = coords_3d_np
                
                # 投影到2D图像坐标
                X, Y, Z = coords_camera[:, 0], coords_camera[:, 1], coords_camera[:, 2]
                valid_mask = Z > 0.1
                
                if valid_mask.sum() > 0:
                    X_valid = X[valid_mask]
                    Y_valid = Y[valid_mask]
                    Z_valid = Z[valid_mask]
                    error_valid = error_np[valid_mask]
                    
                    # 投影到像素坐标
                    u = (cam_intrinsic[0, 0] * X_valid / Z_valid + cam_intrinsic[0, 2]).astype(np.int32)
                    v = (cam_intrinsic[1, 1] * Y_valid / Z_valid + cam_intrinsic[1, 2]).astype(np.int32)
                    
                    # 过滤超出范围的点
                    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                    u_valid = u[in_bounds]
                    v_valid = v[in_bounds]
                    error_valid = error_valid[in_bounds]
                    
                    # 将误差填充到2D图（如果有多个点投影到同一像素，使用平均值）
                    for ui, vi, err in zip(u_valid, v_valid, error_valid):
                        error_map_2d[vi, ui] += err
                        weight_map_2d[vi, ui] += 1.0
                    
                    # 归一化（求平均）
                    non_zero_mask = weight_map_2d > 0
                    if non_zero_mask.sum() > 0:
                        error_map_2d[non_zero_mask] /= weight_map_2d[non_zero_mask]
                
                batch_scores.append(error_map_2d)
            
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

