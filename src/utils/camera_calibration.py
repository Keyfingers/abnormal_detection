"""
相机标定工具模块
处理相机内参、外参（LiDAR到相机）的加载和转换
"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Union
from pathlib import Path


def load_kitti_calibration(calib_file: str) -> Dict[str, np.ndarray]:
    """
    加载KITTI格式的标定文件
    
    KITTI标定文件格式：
    P0: 3x4 投影矩阵（相机0）
    P1: 3x4 投影矩阵（相机1）
    P2: 3x4 投影矩阵（相机2，通常用于目标检测）
    P3: 3x4 投影矩阵（相机3）
    R0_rect: 3x3 校正旋转矩阵
    Tr_velo_to_cam: 3x4 LiDAR到相机的变换矩阵
    Tr_imu_to_velo: 3x4 IMU到LiDAR的变换矩阵
    
    Args:
        calib_file: 标定文件路径
        
    Returns:
        calib_dict: 包含所有标定参数的字典
    """
    calib_dict = {}
    
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) < 2:
            continue
        
        key = parts[0].rstrip(':')
        values = [float(v) for v in parts[1:]]
        
        if key.startswith('P'):
            # 投影矩阵 3x4
            calib_dict[key] = np.array(values).reshape(3, 4)
        elif key == 'R0_rect':
            # 校正旋转矩阵 3x3
            calib_dict[key] = np.array(values).reshape(3, 3)
        elif key.startswith('Tr_'):
            # 变换矩阵 3x4
            calib_dict[key] = np.array(values).reshape(3, 4)
    
    return calib_dict


def build_projection_matrix(
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    rect: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    构建完整的投影矩阵
    
    投影矩阵 P = K @ R_rect @ T_lidar2cam
    
    Args:
        intrinsic: 相机内参矩阵 (3, 3) 或 (3, 4)
        extrinsic: LiDAR到相机的变换矩阵 (3, 4) 或 (4, 4)
        rect: 校正旋转矩阵 (3, 3) 或 (4, 4)，可选
        
    Returns:
        projection_matrix: 投影矩阵 (3, 4)
    """
    # 处理内参矩阵
    if intrinsic.shape == (3, 3):
        K = intrinsic
    elif intrinsic.shape == (3, 4):
        K = intrinsic[:, :3]
    else:
        raise ValueError(f"内参矩阵形状错误: {intrinsic.shape}")
    
    # 处理外参矩阵
    if extrinsic.shape == (3, 4):
        T = extrinsic
    elif extrinsic.shape == (4, 4):
        T = extrinsic[:3, :]
    else:
        raise ValueError(f"外参矩阵形状错误: {extrinsic.shape}")
    
    # 处理校正矩阵
    if rect is not None:
        if rect.shape == (3, 3):
            R_rect = np.eye(4)
            R_rect[:3, :3] = rect
        elif rect.shape == (4, 4):
            R_rect = rect
        else:
            raise ValueError(f"校正矩阵形状错误: {rect.shape}")
        
        # 扩展T到4x4
        T_4x4 = np.eye(4)
        T_4x4[:3, :] = T
        
        # 组合：P = K @ R_rect @ T_lidar2cam
        projection_matrix = K @ R_rect[:3, :3] @ T_4x4[:3, :]
    else:
        # 如果没有校正矩阵，直接组合
        T_4x4 = np.eye(4)
        T_4x4[:3, :] = T
        projection_matrix = K @ T_4x4[:3, :]
    
    return projection_matrix


def get_projection_matrix_from_kitti(
    calib_file: str,
    cam_idx: int = 2
) -> np.ndarray:
    """
    从KITTI标定文件获取指定相机的投影矩阵
    
    Args:
        calib_file: KITTI标定文件路径
        cam_idx: 相机索引（0, 1, 2, 3），默认2（通常用于目标检测）
        
    Returns:
        projection_matrix: 投影矩阵 (3, 4)
    """
    calib_dict = load_kitti_calibration(calib_file)
    
    # 获取投影矩阵（已经包含内参和外参）
    P_key = f'P{cam_idx}'
    if P_key not in calib_dict:
        raise ValueError(f"标定文件中没有找到 {P_key}")
    
    return calib_dict[P_key]


def get_projection_matrix_from_nuscenes(
    cam_intrinsic: np.ndarray,
    lidar2cam: np.ndarray
) -> np.ndarray:
    """
    从NuScenes格式构建投影矩阵
    
    Args:
        cam_intrinsic: 相机内参 (3, 3)
        lidar2cam: LiDAR到相机的变换矩阵 (4, 4)
        
    Returns:
        projection_matrix: 投影矩阵 (3, 4)
    """
    if cam_intrinsic.shape != (3, 3):
        raise ValueError(f"内参矩阵形状错误: {cam_intrinsic.shape}")
    if lidar2cam.shape != (4, 4):
        raise ValueError(f"变换矩阵形状错误: {lidar2cam.shape}")
    
    # P = K @ T_lidar2cam[:3, :]
    projection_matrix = cam_intrinsic @ lidar2cam[:3, :]
    
    return projection_matrix


def create_default_projection_matrix(
    image_width: int = 1333,
    image_height: int = 800,
    fov: float = 90.0
) -> np.ndarray:
    """
    创建默认的投影矩阵（用于测试）
    
    Args:
        image_width: 图像宽度
        image_height: 图像高度
        fov: 视场角（度）
        
    Returns:
        projection_matrix: 默认投影矩阵 (3, 4)
    """
    # 计算焦距
    f = (image_width / 2.0) / np.tan(np.radians(fov / 2.0))
    
    # 内参矩阵
    K = np.array([
        [f, 0, image_width / 2.0],
        [0, f, image_height / 2.0],
        [0, 0, 1]
    ])
    
    # 外参矩阵（假设LiDAR和相机坐标系对齐，只有平移）
    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    
    # 投影矩阵
    projection_matrix = K @ T
    
    return projection_matrix


def projection_matrix_to_torch(
    projection_matrix: np.ndarray,
    device: str = "cuda"
) -> torch.Tensor:
    """
    将投影矩阵转换为torch.Tensor
    
    Args:
        projection_matrix: 投影矩阵 (3, 4) 或 (4, 4)
        device: 设备
        
    Returns:
        projection_tensor: torch.Tensor格式的投影矩阵
    """
    return torch.from_numpy(projection_matrix).float().to(device)

