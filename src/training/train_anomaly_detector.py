"""
异常检测模型训练脚本
实现自监督合成异常训练策略
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.anomaly_detector import AnomalyDetector
from src.losses.anomaly_loss import AnomalyDetectionLoss
from src.utils.pseudo_anomaly import generate_pseudo_anomalies


def train_one_epoch(
    model: AnomalyDetector,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_func: AnomalyDetectionLoss,
    device: str = "cuda",
    anomaly_prob: float = 0.5,
    noise_type: str = "gaussian",
    noise_scale: float = 2.0
) -> Dict[str, float]:
    """
    训练一个epoch
    
    训练流程：
    1. 提取2D和3D特征（Backbone冻结）
    2. 在线生成伪异常（特征层噪声注入）
    3. 融合头前向传播
    4. 计算损失（Focal Loss + Dice Loss）
    5. 反向传播（仅更新FusionHead）
    
    Args:
        model: 异常检测模型
        dataloader: 数据加载器（仅Normal数据）
        optimizer: 优化器
        loss_func: 损失函数
        device: 计算设备
        anomaly_prob: 生成异常的概率，默认0.5
        noise_type: 噪声类型，'gaussian'或'shuffle'
        noise_scale: 噪声强度，默认2.0
        
    Returns:
        metrics: 训练指标字典
    """
    model.train()
    
    total_loss = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # 1. 准备数据（只有正常数据）
        images = batch['img'].to(device)  # (B, C, H, W)
        point_clouds = batch['points']  # List[Tensor]，每个元素是一个点云
        projection_matrices = batch['projection_matrix'].to(device)  # (B, 3, 4) 或 (B, 4, 4)
        
        B = images.shape[0]
        
        # 2. 提取特征（Backbone冻结）
        # 注意：由于每个样本可能有不同的投影矩阵，需要逐个处理
        batch_img_features = []
        batch_pts_features = []
        
        with torch.no_grad():
            for i in range(B):
                # 单个样本
                img_i = images[i:i+1]  # (1, C, H, W)
                points_i = [point_clouds[i]]  # List with one point cloud
                proj_i = projection_matrices[i]  # (3, 4) 或 (4, 4)
                
                # 提取特征
                img_feat, pts_feat = model.extract_features(
                    img_i, points_i, proj_i
                )
                
                batch_img_features.append(img_feat)
                batch_pts_features.append(pts_feat)
            
            # 拼接批次
            img_features = torch.cat(batch_img_features, dim=0)  # (B, C_img, H', W')
            pts_features_proj = torch.cat(batch_pts_features, dim=0)  # (B, C_pts, H', W')
        
        # 3. 关键步骤：在线生成伪异常（Pseudo-Anomaly Synthesis）
        # 注意：掩码尺寸是相对于特征图尺寸的（不是原图像素）
        img_features_corrupted, pts_features_corrupted, anomaly_mask = generate_pseudo_anomalies(
            img_features,
            pts_features_proj,
            anomaly_prob=anomaly_prob,
            num_boxes=1,
            noise_type=noise_type,
            noise_scale=noise_scale,
            min_size_ratio=0.05,  # 特征图的5%
            max_size_ratio=0.2    # 特征图的20%
        )
        
        # 4. 融合头前向传播（可训练）
        anomaly_map = model.fusion_head(img_features_corrupted, pts_features_corrupted)
        
        # 5. 计算损失
        loss_dict = loss_func(anomaly_map, anomaly_mask)
        loss = loss_dict['loss']
        
        # 6. 反向传播（仅更新FusionHead）
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（可选，防止梯度爆炸）
        trainable_params = list(model.get_trainable_parameters())
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        optimizer.step()
        
        # 更新指标
        total_loss += loss.item()
        total_focal_loss += loss_dict['focal_loss'].item()
        total_dice_loss += loss_dict['dice_loss'].item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'focal': f'{loss_dict["focal_loss"].item():.4f}',
            'dice': f'{loss_dict["dice_loss"].item():.4f}'
        })
    
    # 计算平均指标
    metrics = {
        'loss': total_loss / num_batches,
        'focal_loss': total_focal_loss / num_batches,
        'dice_loss': total_dice_loss / num_batches
    }
    
    return metrics


def validate(
    model: AnomalyDetector,
    dataloader: DataLoader,
    loss_func: AnomalyDetectionLoss,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    验证模型（使用真实的异常数据）
    
    Args:
        model: 异常检测模型
        dataloader: 数据加载器（包含真实异常）
        loss_func: 损失函数
        device: 计算设备
        
    Returns:
        metrics: 验证指标字典
    """
    model.eval()
    
    total_loss = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            # 准备数据
            images = batch['img'].to(device)
            point_clouds = batch['points']  # List[Tensor]
            projection_matrices = batch['projection_matrix'].to(device)  # (B, 3, 4)
            anomaly_mask = batch['anomaly_mask'].to(device)  # 真实异常标签 (B, 1, H, W)
            
            B = images.shape[0]
            
            # 前向传播（不生成伪异常）
            # 由于每个样本可能有不同的投影矩阵，需要逐个处理
            batch_anomaly_maps = []
            
            for i in range(B):
                img_i = images[i:i+1]  # (1, C, H, W)
                points_i = [point_clouds[i]]  # List with one point cloud
                proj_i = projection_matrices[i]  # (3, 4)
                
                output = model(img_i, points_i, proj_i)
                batch_anomaly_maps.append(output['anomaly_map'])
            
            anomaly_map = torch.cat(batch_anomaly_maps, dim=0)  # (B, 1, H', W')
            
            # 计算损失
            loss_dict = loss_func(anomaly_map, anomaly_mask)
            loss = loss_dict['loss']
            
            # 更新指标
            total_loss += loss.item()
            total_focal_loss += loss_dict['focal_loss'].item()
            total_dice_loss += loss_dict['dice_loss'].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'focal': f'{loss_dict["focal_loss"].item():.4f}',
                'dice': f'{loss_dict["dice_loss"].item():.4f}'
            })
    
    # 计算平均指标
    metrics = {
        'loss': total_loss / num_batches,
        'focal_loss': total_focal_loss / num_batches,
        'dice_loss': total_dice_loss / num_batches
    }
    
    return metrics


def train(
    model: AnomalyDetector,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    save_dir: str = "checkpoints/anomaly_detector",
    anomaly_prob: float = 0.5,
    noise_type: str = "gaussian",
    noise_scale: float = 2.0
):
    """
    完整训练流程
    
    Args:
        model: 异常检测模型
        train_loader: 训练数据加载器（仅Normal数据）
        val_loader: 验证数据加载器（包含真实异常）
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备
        save_dir: 模型保存目录
        anomaly_prob: 生成异常的概率
        noise_type: 噪声类型
        noise_scale: 噪声强度
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    loss_func = AnomalyDetectionLoss()
    
    # 训练历史
    train_history = []
    val_history = []
    
    best_val_loss = float('inf')
    
    print("="*60)
    print("开始训练异常检测模型")
    print("="*60)
    print(f"可训练参数量: {model.count_parameters()['trainable']:,}")
    print(f"可训练参数比例: {model.count_parameters()['trainable_ratio']*100:.2f}%")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"伪异常概率: {anomaly_prob}")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_func, device,
            anomaly_prob=anomaly_prob,
            noise_type=noise_type,
            noise_scale=noise_scale
        )
        train_history.append(train_metrics)
        
        # 验证
        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_func, device)
            val_history.append(val_metrics)
            
            print(f"\n验证指标:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Focal Loss: {val_metrics['focal_loss']:.4f}")
            print(f"  Dice Loss: {val_metrics['dice_loss']:.4f}")
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ 保存最佳模型 (val_loss: {best_val_loss:.4f})")
        
        # 学习率调度
        scheduler.step()
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


# 示例使用
if __name__ == "__main__":
    import argparse
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from src.datasets.anovox_dataset import AnoVoxDataset
    
    parser = argparse.ArgumentParser(description="训练异常检测模型")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="AnoVox数据集根目录路径"
    )
    parser.add_argument(
        "--mask2former_config",
        type=str,
        default="configs/mask2former_swin_l_cityscapes.yaml",
        help="Mask2Former配置文件路径"
    )
    parser.add_argument(
        "--mask2former_checkpoint",
        type=str,
        default="checkpoints/mask2former/model_final_064788.pkl",
        help="Mask2Former权重路径"
    )
    parser.add_argument(
        "--minkunet_checkpoint",
        type=str,
        default="checkpoints/mmdet3d/mmdet3d_placeholder.pth",
        help="MinkUNet权重路径"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="批次大小（根据显存调整，推荐2-4）"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="训练轮数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="学习率"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载器工作进程数"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备（cuda或cpu）"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/anomaly_detector",
        help="模型保存目录"
    )
    parser.add_argument(
        "--anomaly_prob",
        type=float,
        default=0.5,
        help="生成伪异常的概率"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "shuffle"],
        help="噪声类型"
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=2.0,
        help="噪声强度"
    )
    
    args = parser.parse_args()
    
    # 1. 图像预处理 (适配Mask2Former，通常Resize到短边800)
    transform = transforms.Compose([
        transforms.Resize((800, 1333)),  # Mask2Former常用尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. 创建数据集
    print(f"加载数据集: {args.data_root}")
    train_dataset = AnoVoxDataset(
        root_dir=args.data_root,
        transform=transform
    )
    
    # 3. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=AnoVoxDataset.collate_fn,  # 关键！处理点云List
        pin_memory=True if args.device == "cuda" else False
    )
    
    print(f"数据集大小: {len(train_dataset)}")
    print(f"批次数量: {len(train_loader)}")
    
    # 4. 初始化模型
    print("初始化模型...")
    model = AnomalyDetector(
        mask2former_config_path=args.mask2former_config,
        mask2former_checkpoint_path=args.mask2former_checkpoint,
        minkunet_checkpoint_path=args.minkunet_checkpoint,
        device=args.device
    )
    
    # 打印参数量信息
    param_info = model.count_parameters()
    print(f"总参数量: {param_info['total']:,}")
    print(f"可训练参数量: {param_info['trainable']:,}")
    print(f"可训练参数比例: {param_info['trainable_ratio']*100:.2f}%")
    
    # 5. 开始训练
    # 注意：val_loader设为None，因为我们只在训练集上跑伪异常
    # 真正的验证需要另外加载Anomaly目录
    train(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        save_dir=args.save_dir,
        anomaly_prob=args.anomaly_prob,
        noise_type=args.noise_type,
        noise_scale=args.noise_scale
    )

