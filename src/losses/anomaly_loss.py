"""
异常检测损失函数模块
实现Focal Loss和Dice Loss的组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss：解决类别不平衡问题
    
    公式：FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    其中：
    - p_t: 预测概率
    - α_t: 类别权重（平衡正负样本）
    - γ: 聚焦参数（focusing parameter），默认2.0
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重，默认0.25（正样本权重）
            gamma: 聚焦参数，默认2.0
            reduction: 损失归约方式，'mean'或'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            pred: 预测概率图 (B, 1, H, W) 或 (B, H, W)，值域[0, 1]
            target: 真实标签 (B, 1, H, W) 或 (B, H, W)，值域{0, 1}
            
        Returns:
            loss: Focal Loss标量值
        """
        # 确保维度一致
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # (B, H, W)
        if target.dim() == 4:
            target = target.squeeze(1).float()  # (B, H, W)
        
        # 数值稳定性：clamp预测值，避免log(0)或log(1)导致的NaN
        pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
        
        # 计算p_t
        p_t = pred * target + (1 - pred) * (1 - target)  # (B, H, W)
        
        # 计算alpha_t
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # 计算Focal Loss
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        focal_loss = focal_weight * ce_loss
        
        # 归约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss：优化分割边界
    
    公式：Dice = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
    
    适用于分割任务，对边界敏感
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean'
    ):
        """
        初始化Dice Loss
        
        Args:
            smooth: 平滑项，避免除零，默认1e-6
            reduction: 损失归约方式，'mean'或'sum'
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Dice Loss
        
        Args:
            pred: 预测概率图 (B, 1, H, W) 或 (B, H, W)，值域[0, 1]
            target: 真实标签 (B, 1, H, W) 或 (B, H, W)，值域{0, 1}
            
        Returns:
            loss: Dice Loss标量值
        """
        # 确保维度一致
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # (B, H, W)
        if target.dim() == 4:
            target = target.squeeze(1).float()  # (B, H, W)
        
        # 展平
        pred_flat = pred.contiguous().view(-1)  # (B*H*W,)
        target_flat = target.contiguous().view(-1)  # (B*H*W,)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss = 1 - Dice
        dice_loss = 1.0 - dice
        
        return dice_loss


class AnomalyDetectionLoss(nn.Module):
    """
    异常检测组合损失函数
    
    组合Focal Loss和Dice Loss：
    Loss = λ_focal * FocalLoss + λ_dice * DiceLoss
    
    设计理由：
    - Focal Loss：解决正负样本不平衡（异常区域通常很小）
    - Dice Loss：优化分割边界（异常检测本质上是分割任务）
    """
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        初始化组合损失函数
        
        Args:
            focal_weight: Focal Loss权重，默认1.0
            dice_weight: Dice Loss权重，默认1.0
            focal_alpha: Focal Loss的alpha参数，默认0.25
            focal_gamma: Focal Loss的gamma参数，默认2.0
        """
        super(AnomalyDetectionLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Args:
            pred: 预测概率图 (B, 1, H, W)，值域[0, 1]
            target: 真实标签 (B, 1, H, W)，值域{0, 1}
            
        Returns:
            loss_dict: 包含总损失和各项损失的字典
        """
        # 计算各项损失
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # 组合损失
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return {
            'loss': total_loss,
            'focal_loss': focal,
            'dice_loss': dice
        }

