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
        point_clouds = batch['points']  # 点云数据
        projection_matrix = batch['projection_matrix'].to(device)  # (3, 4) 或 (4, 4)
        
        # 2. 提取特征（Backbone冻结）
        with torch.no_grad():
            img_features, pts_features_proj = model.extract_features(
                images, point_clouds, projection_matrix
            )
        
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
            point_clouds = batch['points']
            projection_matrix = batch['projection_matrix'].to(device)
            anomaly_mask = batch['anomaly_mask'].to(device)  # 真实异常标签
            
            # 前向传播（不生成伪异常）
            output = model(images, point_clouds, projection_matrix)
            anomaly_map = output['anomaly_map']
            
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
    # 这里需要实现DataLoader
    # 示例代码结构，实际使用时需要根据AnoVox数据集格式实现
    
    print("训练脚本框架已创建")
    print("需要实现AnoVox数据集的DataLoader才能开始训练")

