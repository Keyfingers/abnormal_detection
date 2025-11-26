"""
点云预处理工具模块
提供点云体素化和特征提取功能
"""
import numpy as np
import torch
from typing import Tuple, Optional, Union
from collections import defaultdict


def voxelize_pointcloud(
    points: Union[np.ndarray, torch.Tensor],
    voxel_size: float = 0.05,
    max_points_per_voxel: Optional[int] = None,
    return_indices: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    将点云体素化
    
    体素化步骤：
    1. 计算每个点所属的体素坐标
    2. 对每个体素内的点进行聚合（平均或采样）
    3. 返回体素坐标和体素特征
    
    Args:
        points: 点云数据 (N, 3) 或 (N, 4)，最后一维可以是强度/反射率
        voxel_size: 体素尺寸（米），默认0.05（5cm）
        max_points_per_voxel: 每个体素最大点数，如果超过则随机采样
        return_indices: 是否返回点云索引映射
        
    Returns:
        voxel_coords: 体素坐标 (M, 3)，整数坐标
        voxel_features: 体素特征 (M, C)，C为特征维度
        point_to_voxel: 点云到体素的映射索引 (N,)，如果return_indices=True
    """
    # 转换为numpy数组
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = np.asarray(points, dtype=np.float32)
    
    if points_np.shape[0] == 0:
        # 空点云
        empty_coords = torch.zeros((0, 3), dtype=torch.int32)
        empty_features = torch.zeros((0, points_np.shape[1] if points_np.shape[0] > 0 else 3), dtype=torch.float32)
        if return_indices:
            return empty_coords, empty_features, torch.zeros((0,), dtype=torch.int64)
        return empty_coords, empty_features
    
    # 计算体素坐标
    voxel_coords_np = np.floor(points_np[:, :3] / voxel_size).astype(np.int32)
    
    # 使用字典聚合体素内的点
    voxel_dict = defaultdict(list)
    point_indices = []
    
    for i, voxel_coord in enumerate(voxel_coords_np):
        voxel_key = tuple(voxel_coord)
        voxel_dict[voxel_key].append(i)
        point_indices.append(len(voxel_dict[voxel_key]) - 1)
    
    # 提取体素特征
    voxel_coords_list = []
    voxel_features_list = []
    point_to_voxel_map = np.zeros(len(points_np), dtype=np.int64)
    
    for voxel_idx, (voxel_key, point_indices_in_voxel) in enumerate(voxel_dict.items()):
        # 获取该体素内的点
        points_in_voxel = points_np[point_indices_in_voxel]
        
        # 如果超过最大点数，随机采样
        if max_points_per_voxel is not None and len(points_in_voxel) > max_points_per_voxel:
            indices = np.random.choice(len(points_in_voxel), max_points_per_voxel, replace=False)
            points_in_voxel = points_in_voxel[indices]
        
        # 计算体素特征（平均）
        if points_in_voxel.shape[1] > 3:
            # 有额外特征（如强度）
            voxel_feature = np.mean(points_in_voxel, axis=0)
        else:
            # 只有坐标，使用平均坐标作为特征
            voxel_feature = np.mean(points_in_voxel, axis=0)
        
        voxel_coords_list.append(voxel_key)
        voxel_features_list.append(voxel_feature)
        
        # 更新点云到体素的映射
        for point_idx in point_indices_in_voxel:
            point_to_voxel_map[point_idx] = voxel_idx
    
    # 转换为torch.Tensor
    voxel_coords = torch.from_numpy(np.array(voxel_coords_list, dtype=np.int32))
    voxel_features = torch.from_numpy(np.array(voxel_features_list, dtype=np.float32))
    
    if return_indices:
        point_to_voxel = torch.from_numpy(point_to_voxel_map)
        return voxel_coords, voxel_features, point_to_voxel
    
    return voxel_coords, voxel_features


def normalize_coordinates(
    points: Union[np.ndarray, torch.Tensor],
    center: Optional[Union[np.ndarray, torch.Tensor]] = None,
    scale: Optional[float] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], float]:
    """
    归一化点云坐标
    
    Args:
        points: 点云数据 (N, 3)
        center: 中心点，如果为None则自动计算
        scale: 缩放因子，如果为None则自动计算
        
    Returns:
        normalized_points: 归一化后的点云
        center: 使用的中心点
        scale: 使用的缩放因子
    """
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
        is_torch = True
    else:
        points_np = np.asarray(points, dtype=np.float32)
        is_torch = False
    
    if points_np.shape[0] == 0:
        if is_torch:
            return torch.from_numpy(points_np), np.array([0.0, 0.0, 0.0]), 1.0
        return points_np, np.array([0.0, 0.0, 0.0]), 1.0
    
    # 计算中心点
    if center is None:
        center = np.mean(points_np[:, :3], axis=0)
    else:
        if isinstance(center, torch.Tensor):
            center = center.cpu().numpy()
        center = np.asarray(center, dtype=np.float32)
    
    # 计算缩放因子
    if scale is None:
        # 使用最大距离作为缩放因子
        centered_points = points_np[:, :3] - center
        max_dist = np.max(np.linalg.norm(centered_points, axis=1))
        scale = max_dist if max_dist > 0 else 1.0
    else:
        scale = float(scale)
    
    # 归一化
    normalized_points = points_np.copy()
    normalized_points[:, :3] = (points_np[:, :3] - center) / scale
    
    if is_torch:
        normalized_points = torch.from_numpy(normalized_points)
    
    return normalized_points, center, scale


def extract_pointcloud_features(
    points: Union[np.ndarray, torch.Tensor],
    use_intensity: bool = True,
    use_normal: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    提取点云特征
    
    Args:
        points: 点云数据 (N, 3) 或 (N, 4+)
        use_intensity: 是否使用强度特征
        use_normal: 是否计算法向量（未实现）
        
    Returns:
        features: 点云特征 (N, C)
    """
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
        is_torch = True
    else:
        points_np = np.asarray(points, dtype=np.float32)
        is_torch = False
    
    if points_np.shape[0] == 0:
        if is_torch:
            return torch.zeros((0, 3), dtype=torch.float32)
        return np.zeros((0, 3), dtype=np.float32)
    
    features_list = []
    
    # 坐标特征（必需）
    features_list.append(points_np[:, :3])
    
    # 强度特征（如果有）
    if use_intensity and points_np.shape[1] > 3:
        intensity = points_np[:, 3:4]
        features_list.append(intensity)
    
    # 法向量特征（如果启用，需要额外计算）
    if use_normal:
        # TODO: 实现法向量计算
        pass
    
    # 拼接特征
    features = np.concatenate(features_list, axis=1)
    
    if is_torch:
        features = torch.from_numpy(features)
    
    return features


