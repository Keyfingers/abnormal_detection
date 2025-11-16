"""
训练3D几何分支（阶段三）
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import MinkowskiEngine as ME
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.models.geometric_3d import Geometric3DBranch
from src.data.nuscenes_dataset import NuScenesPointCloudDataset


def collate_fn(batch):
    """自定义collate函数"""
    images = torch.stack([item['image'] for item in batch])
    point_clouds = [item['point_cloud'] for item in batch]
    return {
        'images': images,
        'point_clouds': point_clouds,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    mask_ratio: float = 0.3,
):
    """
    训练一个epoch
    
    修复说明：
    - 添加mask_ratio参数，实现规则要求的随机mask训练（30%体素被mask）
    - 修复损失计算：应该计算原始点云和重建点云的L2损失，而不是重建误差的均值
    - 重建误差是异常分数，不是训练损失；训练损失应该是重建损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        point_clouds = batch['point_clouds']
        
        # 过滤空点云
        valid_pcs = [pc for pc in point_clouds if len(pc) > 0]
        if len(valid_pcs) == 0:
            continue
        
        optimizer.zero_grad()
        
        # 修复：前向传播时应用随机mask（规则要求：随机mask掉30%体素）
        # 这是自监督重建任务的核心：模型从部分可见的点云学习重建完整点云
        results = model(
            valid_pcs,
            return_features=False,
            return_reconstruction_error=False,  # 训练时不需要重建误差
            apply_mask=True,  # 修复：训练时应用mask
            mask_ratio=mask_ratio,  # 修复：使用30%的mask比例
        )
        
        # 修复：计算重建损失（L2损失）
        # 规则要求：训练MinkUNet来预测并重建被mask掉的部分
        # 损失应该是原始点云特征和重建点云特征的L2距离
        original_tensor = results['sparse_tensor']  # 原始完整点云
        reconstruction = results['reconstruction']  # 重建的点云（3维xyz）
        
        # 获取原始特征（xyz坐标）
        original_features = original_tensor.F  # (N, 3)
        
        # 将重建特征插值回原始坐标位置
        # 由于稀疏张量坐标可能不完全匹配，需要插值
        original_coords = original_tensor.C.float()[:, 1:]  # (N, 3) 去除batch索引
        reconstruction_coords = reconstruction.C.float()[:, 1:]  # (M, 3)
        reconstructed_features = reconstruction.F  # (M, 3)
        
        # 如果坐标匹配，直接计算L2损失
        if len(original_coords) == len(reconstruction_coords):
            coords_match = torch.allclose(original_coords, reconstruction_coords, atol=1e-3)
            if coords_match:
                # 直接计算L2损失
                loss = criterion(original_features, reconstructed_features)
            else:
                # 坐标不匹配，使用最近邻插值
                
                original_coords_np = original_coords.detach().cpu().numpy()
                reconstruction_coords_np = reconstruction_coords.detach().cpu().numpy()
                reconstructed_features_np = reconstructed_features.detach().cpu().numpy()
                
                distances = cdist(original_coords_np, reconstruction_coords_np)
                nearest_indices = np.argmin(distances, axis=1)
                
                nearest_reconstructed = torch.from_numpy(
                    reconstructed_features_np[nearest_indices]
                ).to(original_features.device)
                
                loss = criterion(original_features, nearest_reconstructed)
        else:
            # 坐标数量不匹配，使用chamfer距离的简化版本
            # 计算每个原始点到最近重建点的距离
            from scipy.spatial.distance import cdist
            import numpy as np
            
            original_coords_np = original_coords.detach().cpu().numpy()
            reconstruction_coords_np = reconstruction_coords.detach().cpu().numpy()
            reconstructed_features_np = reconstructed_features.detach().cpu().numpy()
            
            distances = cdist(original_coords_np, reconstruction_coords_np)
            nearest_indices = np.argmin(distances, axis=1)
            
            nearest_reconstructed = torch.from_numpy(
                reconstructed_features_np[nearest_indices]
            ).to(original_features.device)
            
            loss = criterion(original_features, nearest_reconstructed)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train 3D Geometric Branch')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--output-dir', type=str, default='outputs/geometric_3d',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size for point cloud quantization')
    parser.add_argument('--feature-dim', type=int, default=128,
                        help='Feature dimension')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据集
    print("Loading dataset...")
    train_dataset = NuScenesPointCloudDataset(
        data_root=args.data_root,
        split='train',
    )
    val_dataset = NuScenesPointCloudDataset(
        data_root=args.data_root,
        split='val',
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # 模型
    print("Building model...")
    model = Geometric3DBranch(
        in_channels=3,
        feature_dim=args.feature_dim,
        checkpoint_path=args.resume,
        freeze_backbone=False,
        voxel_size=args.voxel_size,
    )
    model = model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    print("=" * 50)
    print("Training 3D Geometric Branch (MinkUNet)")
    print("=" * 50)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Mask ratio: 0.3 (30% voxels masked)")  # 修复：显示mask比例
    print("=" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练（修复：传入mask_ratio参数）
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, mask_ratio=0.3
        )
        
        # 验证（修复：使用与训练相同的损失计算方式）
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                point_clouds = batch['point_clouds']
                valid_pcs = [pc for pc in point_clouds if len(pc) > 0]
                if len(valid_pcs) == 0:
                    continue
                
                # 修复：验证时也应用mask（模拟训练条件）
                results = model(
                    valid_pcs,
                    return_features=False,
                    return_reconstruction_error=False,
                    apply_mask=True,  # 验证时也使用mask
                    mask_ratio=0.3,
                )
                
                # 修复：计算重建损失（与训练时相同）
                original_tensor = results['sparse_tensor']
                reconstruction = results['reconstruction']
                original_features = original_tensor.F
                reconstructed_features = reconstruction.F
                
                # 简化：如果特征数量相同，直接计算L2损失
                if len(original_features) == len(reconstructed_features):
                    loss = criterion(original_features, reconstructed_features)
                else:
                    # 使用最近邻插值
                    
                    original_coords = original_tensor.C.float()[:, 1:].detach().cpu().numpy()
                    reconstruction_coords = reconstruction.C.float()[:, 1:].detach().cpu().numpy()
                    reconstructed_features_np = reconstructed_features.detach().cpu().numpy()
                    
                    distances = cdist(original_coords, reconstruction_coords)
                    nearest_indices = np.argmin(distances, axis=1)
                    nearest_reconstructed = torch.from_numpy(
                        reconstructed_features_np[nearest_indices]
                    ).to(original_features.device)
                    
                    loss = criterion(original_features, nearest_reconstructed)
                
                val_loss += loss.item()
                val_batches += 1
        
        val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        
        # 学习率调度
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存检查点
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        
        # 定期保存
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'model_final.pth')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': val_loss,
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    main()

