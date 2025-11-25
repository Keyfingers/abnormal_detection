"""
图像预处理工具模块
提供与Cityscapes训练时一致的图像预处理功能
"""
import cv2
import numpy as np
import torch
from typing import Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    从文件路径加载图像
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        numpy数组格式的图像 (H, W, C)，BGR格式
        
    Raises:
        FileNotFoundError: 如果图像文件不存在
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    return image


def preprocess_image(
    image: np.ndarray,
    target_min_size: int = 800,
    target_max_size: int = 1333,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    预处理图像，使其符合Mask2Former的输入要求
    
    预处理步骤：
    1. BGR转RGB
    2. 调整尺寸（保持宽高比，最小边800，最大边1333）
    3. 归一化（ImageNet均值和标准差）
    4. 转换为torch.Tensor并添加batch维度
    
    Args:
        image: numpy数组格式的图像 (H, W, C)，BGR格式
        target_min_size: 目标最小尺寸，默认800
        target_max_size: 目标最大尺寸，默认1333
        mean: 归一化均值（RGB顺序），默认ImageNet均值
        std: 归一化标准差（RGB顺序），默认ImageNet标准差
        
    Returns:
        torch.Tensor: 预处理后的图像张量 (1, C, H', W')，已归一化
    """
    # BGR转RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # 获取原始尺寸
    h, w = image_rgb.shape[:2]
    
    # 计算缩放比例（保持宽高比）
    scale = target_min_size / min(h, w)
    if max(h, w) * scale > target_max_size:
        scale = target_max_size / max(h, w)
    
    # 调整尺寸
    new_h, new_w = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 转换为float32并归一化到[0, 1]
    image_float = image_resized.astype(np.float32) / 255.0
    
    # 转换为CHW格式
    image_chw = np.transpose(image_float, (2, 0, 1))
    
    # 归一化（ImageNet统计量）
    mean_array = np.array(mean).reshape(3, 1, 1)
    std_array = np.array(std).reshape(3, 1, 1)
    image_normalized = (image_chw - mean_array) / std_array
    
    # 转换为torch.Tensor并添加batch维度
    image_tensor = torch.from_numpy(image_normalized).float()
    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
    
    return image_tensor



