"""
伪异常生成工具模块
实现自监督合成异常（Self-Supervised Synthetic Anomaly）

训练策略：
- 仅使用Normal数据
- 在线生成伪异常（特征层噪声注入）
- 避免模型崩溃（Model Collapse）
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import random


def generate_random_box_mask(
    shape: Tuple[int, ...],
    num_boxes: int = 1,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    min_size_ratio: float = 0.05,  # 相对于特征图尺寸的最小比例
    max_size_ratio: float = 0.2,   # 相对于特征图尺寸的最大比例
    device: str = "cuda"
) -> torch.Tensor:
    """
    生成随机矩形掩码（用于伪异常区域）
    
    注意：尺寸参数是相对于特征图尺寸的，不是原图像素
    
    Args:
        shape: 特征图形状，可以是 (B, C, H, W) 或 (H, W, C)
        num_boxes: 矩形数量，默认1
        min_size: 最小矩形尺寸（像素），如果为None则使用min_size_ratio
        max_size: 最大矩形尺寸（像素），如果为None则使用max_size_ratio
        min_size_ratio: 最小矩形尺寸比例（相对于特征图），默认0.05（5%）
        max_size_ratio: 最大矩形尺寸比例（相对于特征图），默认0.2（20%）
        device: 计算设备
        
    Returns:
        mask: 掩码 (B, 1, H, W) 或 (H, W, 1)，值域{0, 1}
    """
    # 处理不同的输入格式
    if len(shape) == 4:
        B, C, H, W = shape
        squeeze_output = False
    elif len(shape) == 3:
        H, W, C = shape
        B = 1
        squeeze_output = True
    else:
        raise ValueError(f"不支持的形状: {shape}")
    
    # 计算相对于特征图尺寸的矩形大小
    if min_size is None:
        min_size = max(5, int(min(H, W) * min_size_ratio))  # 至少5像素
    if max_size is None:
        max_size = min(int(min(H, W) * max_size_ratio), min(H, W) // 2)  # 最多特征图的一半
    
    # 确保min_size <= max_size
    min_size = min(min_size, max_size)
    
    # 初始化掩码
    mask = torch.zeros(B, 1, H, W, device=device)
    
    # 为每个batch生成随机矩形
    for b in range(B):
        for _ in range(num_boxes):
            # 随机生成矩形位置和大小（相对于特征图尺寸）
            box_h = random.randint(min_size, min(max_size, H // 2))
            box_w = random.randint(min_size, min(max_size, W // 2))
            
            top = random.randint(0, max(1, H - box_h))
            left = random.randint(0, max(1, W - box_w))
            
            # 设置掩码
            mask[b, 0, top:top+box_h, left:left+box_w] = 1.0
    
    if squeeze_output:
        mask = mask.squeeze(0).permute(1, 2, 0)  # (H, W, 1)
    
    return mask


def inject_feature_noise(
    features: torch.Tensor,
    mask: torch.Tensor,
    noise_type: str = "gaussian",
    noise_scale: float = 2.0
) -> torch.Tensor:
    """
    在特征图上注入噪声（生成伪异常）
    
    Args:
        features: 特征图 (B, C, H, W) 或 (H, W, C)
        mask: 掩码 (B, 1, H, W) 或 (H, W, 1)，值域{0, 1}
        noise_type: 噪声类型，'gaussian'或'shuffle'
        noise_scale: 噪声强度，默认2.0
        
    Returns:
        corrupted_features: 被污染的特征图
    """
    # 处理输入格式
    if features.dim() == 3:
        # (H, W, C) -> (1, C, H, W)
        features = features.permute(2, 0, 1).unsqueeze(0)
        mask = mask.permute(2, 0, 1).unsqueeze(0) if mask.dim() == 3 else mask.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    corrupted_features = features.clone()
    
    # 生成噪声
    if noise_type == "gaussian":
        # 高斯噪声
        noise = torch.randn_like(features) * noise_scale
        corrupted_features = features * (1 - mask) + noise * mask
    
    elif noise_type == "shuffle":
        # 特征打乱（CutPaste风格）
        B, C, H, W = features.shape
        for b in range(B):
            mask_b = mask[b, 0] > 0.5  # (H, W)
            if mask_b.any():
                # 获取被掩码区域的索引
                masked_indices = torch.nonzero(mask_b, as_tuple=False)  # (N, 2)
                
                # 随机打乱这些位置的特征
                if len(masked_indices) > 0:
                    # 随机选择其他位置的特征替换
                    for idx in masked_indices:
                        h, w = idx[0].item(), idx[1].item()
                        # 随机选择另一个位置的特征
                        h_rand = random.randint(0, H - 1)
                        w_rand = random.randint(0, W - 1)
                        corrupted_features[b, :, h, w] = features[b, :, h_rand, w_rand]
    
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")
    
    # 恢复输出格式
    if squeeze_output:
        corrupted_features = corrupted_features.squeeze(0).permute(1, 2, 0)
    
    return corrupted_features


def generate_pseudo_anomalies(
    img_features: torch.Tensor,
    pts_features: torch.Tensor,
    anomaly_prob: float = 0.5,
    num_boxes: int = 1,
    noise_type: str = "gaussian",
    noise_scale: float = 2.0,
    min_size_ratio: float = 0.05,
    max_size_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成伪异常（在线合成）
    
    训练时调用此函数，随机在正常特征图上注入异常
    
    注意：掩码尺寸是相对于特征图尺寸的，不是原图像素
    
    Args:
        img_features: 图像特征 (B, C, H, W) 或 (H, W, C)
        pts_features: 点云特征 (B, C, H, W) 或 (H, W, C)
        anomaly_prob: 生成异常的概率，默认0.5
        num_boxes: 异常区域数量，默认1
        noise_type: 噪声类型，'gaussian'或'shuffle'
        noise_scale: 噪声强度，默认2.0
        min_size_ratio: 最小矩形尺寸比例（相对于特征图），默认0.05
        max_size_ratio: 最大矩形尺寸比例（相对于特征图），默认0.2
        
    Returns:
        img_features_corrupted: 被污染的图像特征
        pts_features_corrupted: 被污染的点云特征
        anomaly_mask: 异常掩码（标签），值域{0, 1}
    """
    # 决定是否生成异常
    if random.random() >= anomaly_prob:
        # 不生成异常，返回原始特征和全零掩码
        if img_features.dim() == 3:
            H, W, _ = img_features.shape
            anomaly_mask = torch.zeros(H, W, 1, device=img_features.device)
        else:
            B, _, H, W = img_features.shape
            anomaly_mask = torch.zeros(B, 1, H, W, device=img_features.device)
        
        return img_features, pts_features, anomaly_mask
    
    # 生成随机掩码（尺寸相对于特征图）
    if img_features.dim() == 3:
        shape = img_features.shape
    else:
        shape = img_features.shape
    
    mask = generate_random_box_mask(
        shape,
        num_boxes=num_boxes,
        min_size_ratio=min_size_ratio,
        max_size_ratio=max_size_ratio,
        device=img_features.device
    )
    
    # 注入噪声到两个特征图
    img_features_corrupted = inject_feature_noise(
        img_features, mask, noise_type=noise_type, noise_scale=noise_scale
    )
    pts_features_corrupted = inject_feature_noise(
        pts_features, mask, noise_type=noise_type, noise_scale=noise_scale
    )
    
    # 掩码作为标签（1=异常，0=正常）
    anomaly_mask = mask
    
    return img_features_corrupted, pts_features_corrupted, anomaly_mask

