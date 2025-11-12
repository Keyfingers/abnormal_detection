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
):
    """训练一个epoch"""
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
        
        # 前向传播
        results = model(
            valid_pcs,
            return_features=False,
            return_reconstruction_error=True,
        )
        
        # 计算重建损失
        # 这里使用简化的损失：重建误差的均值
        reconstruction_error = results['reconstruction_error']
        loss = reconstruction_error.mean()
        
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
    print("=" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # 验证（简化版本）
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                point_clouds = batch['point_clouds']
                valid_pcs = [pc for pc in point_clouds if len(pc) > 0]
                if len(valid_pcs) == 0:
                    continue
                
                results = model(
                    valid_pcs,
                    return_features=False,
                    return_reconstruction_error=True,
                )
                loss = results['reconstruction_error'].mean()
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

