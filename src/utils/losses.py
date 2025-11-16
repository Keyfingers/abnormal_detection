"""
损失函数：Focal Loss和Dice Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡问题
    
    修复说明：
    - 规则要求使用Focal Loss或Dice Loss替代BCELoss
    - Focal Loss能够聚焦于难分类样本，对异常检测任务特别有效
    - 公式：FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: 平衡因子，用于平衡正负样本
            gamma: 聚焦参数，gamma越大，对难样本的关注度越高
            reduction: 损失归约方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 预测分数 (B, H, W) 或 (B, 1, H, W)，值应在[0,1]
            targets: 真实标签 (B, H, W)，值应为0或1
        
        Returns:
            Focal Loss值
        """
        # 确保输入维度正确
        if inputs.dim() == 4:
            inputs = inputs.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 计算p_t（预测概率）
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # 计算focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # 计算focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss用于分割任务
    
    修复说明：
    - 规则要求使用Focal Loss或Dice Loss替代BCELoss
    - Dice Loss对分割边界敏感，适合异常检测的像素级预测
    - 公式：Dice = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: 平滑因子，避免除零
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss
        
        Args:
            inputs: 预测分数 (B, H, W) 或 (B, 1, H, W)，值应在[0,1]
            targets: 真实标签 (B, H, W)，值应为0或1
        
        Returns:
            Dice Loss值
        """
        # 确保输入维度正确
        if inputs.dim() == 4:
            inputs = inputs.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        
        # 展平
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # 计算交集和并集
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss = 1 - Dice
        dice_loss = 1 - dice
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    组合损失：Focal Loss + Dice Loss
    
    修复说明：
    - 结合Focal Loss和Dice Loss的优点
    - Focal Loss处理类别不平衡，Dice Loss优化分割边界
    """
    
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0, 
                 dice_weight: float = 0.5):
        """
        Args:
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            dice_weight: Dice Loss的权重（Focal Loss权重为1-dice_weight）
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            inputs: 预测分数 (B, H, W) 或 (B, 1, H, W)
            targets: 真实标签 (B, H, W)
        
        Returns:
            组合损失值
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        # 组合损失
        loss = (1 - self.dice_weight) * focal + self.dice_weight * dice
        
        return loss

