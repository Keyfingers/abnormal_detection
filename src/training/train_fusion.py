"""
训练融合模块（阶段四）
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.models.semantic_2d import Semantic2DBranch
from src.models.geometric_3d import Geometric3DBranch
from src.models.fusion import FusionHead, FusionModel
from src.data.anovox_dataset import AnoVoxDataset
from src.data.anovox_normality_dataset import AnoVoxNormalityDataset
from src.utils.losses import FocalLoss, DiceLoss, CombinedLoss


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
        images = batch['images'].to(device)
        point_clouds = batch['point_clouds']
        anomaly_masks = batch['anomaly_masks'].to(device)
        camera_intrinsics = batch['camera_intrinsics'].to(device)
        camera_extrinsics = batch['camera_extrinsics'].to(device)
        
        # 过滤空点云
        valid_indices = [i for i, pc in enumerate(point_clouds) if len(pc) > 0]
        if len(valid_indices) == 0:
            continue
        
        # 只处理有效样本
        images = images[valid_indices]
        point_clouds = [point_clouds[i] for i in valid_indices]
        anomaly_masks = anomaly_masks[valid_indices]
        camera_intrinsics = camera_intrinsics[valid_indices]
        camera_extrinsics = camera_extrinsics[valid_indices]
        
        optimizer.zero_grad()
        
        # 前向传播
        results = model(
            images=images,
            point_clouds=point_clouds,
            camera_intrinsic=camera_intrinsics,
            camera_extrinsic=camera_extrinsics,
            return_individual_scores=False,
        )
        
        fusion_score = results['fusion_score']  # (B, H, W)
        
        # 调整尺寸匹配
        if fusion_score.shape[-2:] != anomaly_masks.shape[-2:]:
            fusion_score = torch.nn.functional.interpolate(
                fusion_score.unsqueeze(1),
                size=anomaly_masks.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        
        # 修复：使用Focal Loss或Dice Loss替代BCELoss（规则要求）
        # 融合头已经包含Sigmoid，输出已经是[0,1]范围，无需再次归一化
        # Focal Loss能够聚焦于难分类样本，对异常检测特别有效
        loss = criterion(fusion_score, anomaly_masks)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['images'].to(device)
            point_clouds = batch['point_clouds']
            anomaly_masks = batch['anomaly_masks'].to(device)
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
            
            results = model(
                images=images,
                point_clouds=point_clouds,
                camera_intrinsic=camera_intrinsics,
                camera_extrinsic=camera_extrinsics,
                return_individual_scores=False,
            )
            
            fusion_score = results['fusion_score']
            if fusion_score.shape[-2:] != anomaly_masks.shape[-2:]:
                fusion_score = torch.nn.functional.interpolate(
                    fusion_score.unsqueeze(1),
                    size=anomaly_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)
            
            # 修复：融合头已经包含Sigmoid，无需再次归一化
            loss = criterion(fusion_score, anomaly_masks)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Fusion Module')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to AnoVox dataset root')
    parser.add_argument('--semantic-ckpt', type=str, required=True,
                        help='Path to semantic 2D branch checkpoint')
    parser.add_argument('--geometric-ckpt', type=str, required=True,
                        help='Path to geometric 3D branch checkpoint')
    parser.add_argument('--semantic-config', type=str, required=True,
                        help='Path to semantic 2D branch config')
    parser.add_argument('--output-dir', type=str, default='outputs/fusion',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--feature-2d-dim', type=int, default=256,
                        help='2D feature dimension')
    parser.add_argument('--feature-3d-dim', type=int, default=128,
                        help='3D feature dimension')
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
    # 修复：根据项目规则，阶段4使用"常态"训练集训练融合头
    # 如果数据格式是AnoVox_Normality格式，使用AnoVoxNormalityDataset
    # 否则使用标准的AnoVoxDataset
    normality_dir = os.path.join(args.data_root, 'AnoVox_Normality_Mono_Town03')
    if os.path.exists(normality_dir) or 'Normality' in args.data_root:
        print("检测到AnoVox Normality格式，使用AnoVoxNormalityDataset")
        train_dataset = AnoVoxNormalityDataset(
            data_root=args.data_root,
            split='train',
        )
        val_dataset = AnoVoxNormalityDataset(
            data_root=args.data_root,
            split='test',
        )
    else:
        print("使用标准AnoVoxDataset")
        train_dataset = AnoVoxDataset(
            data_root=args.data_root,
            split='train',
        )
        val_dataset = AnoVoxDataset(
            data_root=args.data_root,
            split='test',
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
    
    # 构建模型
    print("Building models...")
    
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
    
    # 完整融合模型
    model = FusionModel(
        semantic_2d_model=semantic_2d,
        geometric_3d_model=geometric_3d,
        fusion_head=fusion_head,
        freeze_2d=True,
        freeze_3d=True,
    )
    model = model.to(device)
    
    # 优化器（只优化融合头）
    optimizer = optim.Adam(model.fusion_head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 修复：使用Focal Loss替代BCELoss（规则要求）
    # Focal Loss能够聚焦于难分类样本，对异常检测任务特别有效
    # 也可以使用DiceLoss或CombinedLoss，根据实际效果选择
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    # 可选：使用组合损失
    # criterion = CombinedLoss(focal_alpha=0.25, focal_gamma=2.0, dice_weight=0.5)
    
    # 恢复训练（如果指定）
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.fusion_head.load_state_dict(checkpoint['fusion_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # 训练循环
    print("=" * 50)
    print("Training Fusion Module")
    print("=" * 50)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print("=" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
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
                'fusion_head_state_dict': model.fusion_head.state_dict(),
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
                'fusion_head_state_dict': model.fusion_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'model_final.pth')
    torch.save({
        'epoch': args.num_epochs,
        'fusion_head_state_dict': model.fusion_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': val_loss,
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    main()

