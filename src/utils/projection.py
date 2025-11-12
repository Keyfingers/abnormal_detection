"""
3D-2D投影工具：将3D点云特征投影到2D图像坐标
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import MinkowskiEngine as ME


def project_3d_to_2d(
    features_3d: ME.SparseTensor,
    coords_3d: List[np.ndarray],
    camera_intrinsic: np.ndarray,
    camera_extrinsic: Optional[np.ndarray] = None,
    image_shape: Tuple[int, int] = (1024, 2048),
    voxel_size: float = 0.05,
) -> torch.Tensor:
    """
    将3D特征投影到2D图像坐标
    
    Args:
        features_3d: 3D稀疏特征张量 (Minkowski SparseTensor)
        coords_3d: 3D坐标列表，每个元素是 (N, 3) 的numpy数组（原始坐标，未量化）
        camera_intrinsic: 相机内参矩阵 (3, 3)
        camera_extrinsic: 相机外参矩阵 (4, 4)，如果None则假设3D坐标已在相机坐标系
        image_shape: 图像尺寸 (H, W)
        voxel_size: 体素大小（用于坐标转换）
    
    Returns:
        投影后的2D特征图 (B, C_3D, H, W)
    """
    batch_size = len(coords_3d)
    feature_dim = features_3d.F.shape[1]
    H, W = image_shape
    
    # 初始化2D特征图
    device = features_3d.device
    feature_map_2d = torch.zeros(
        (batch_size, feature_dim, H, W),
        dtype=features_3d.F.dtype,
        device=device,
    )
    
    # 获取3D特征的坐标和特征值
    batch_coords, batch_feats = features_3d.decomposed_coordinates_and_features
    
    for batch_idx in range(batch_size):
        # 获取该batch的3D坐标和特征
        coords_3d_batch = coords_3d[batch_idx]  # (N, 3) 原始坐标
        if batch_idx < len(batch_coords):
            coords_sparse = batch_coords[batch_idx]  # (M, 3) 量化后的坐标
            feats_sparse = batch_feats[batch_idx]  # (M, C_3D)
        else:
            continue
        
        # 将量化坐标转换回原始坐标
        coords_3d_quantized = coords_sparse.cpu().numpy() * voxel_size
        
        # 应用外参变换（如果提供）
        if camera_extrinsic is not None:
            # 将3D坐标转换为齐次坐标
            coords_homo = np.column_stack([
                coords_3d_quantized,
                np.ones(len(coords_3d_quantized))
            ])  # (M, 4)
            
            # 应用外参变换
            coords_camera = (camera_extrinsic @ coords_homo.T).T[:, :3]  # (M, 3)
        else:
            coords_camera = coords_3d_quantized
        
        # 投影到2D图像坐标
        # 使用相机内参：u = fx * X/Z + cx, v = fy * Y/Z + cy
        X, Y, Z = coords_camera[:, 0], coords_camera[:, 1], coords_camera[:, 2]
        
        # 避免除零
        valid_mask = Z > 0.1  # 只处理Z > 0.1的点
        
        if valid_mask.sum() == 0:
            continue
        
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        Z_valid = Z[valid_mask]
        feats_valid = feats_sparse[valid_mask]
        
        # 投影到像素坐标
        u = (camera_intrinsic[0, 0] * X_valid / Z_valid + camera_intrinsic[0, 2]).astype(np.int32)
        v = (camera_intrinsic[1, 1] * Y_valid / Z_valid + camera_intrinsic[1, 2]).astype(np.int32)
        
        # 过滤超出图像范围的点
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_valid = u[in_bounds]
        v_valid = v[in_bounds]
        feats_valid = feats_valid[in_bounds]
        
        if len(u_valid) == 0:
            continue
        
        # 将特征填充到2D特征图
        # 如果有多个点投影到同一像素，使用平均值
        for i, (ui, vi) in enumerate(zip(u_valid, v_valid)):
            feature_map_2d[batch_idx, :, vi, ui] += feats_valid[i]
        
        # 计算每个像素的点数（用于平均）
        count_map = torch.zeros((H, W), device=device)
        for ui, vi in zip(u_valid, v_valid):
            count_map[vi, ui] += 1
        
        # 避免除零，只对非零像素求平均
        non_zero_mask = count_map > 0
        if non_zero_mask.sum() > 0:
            feature_map_2d[batch_idx, :, non_zero_mask] /= count_map[non_zero_mask].unsqueeze(0)
    
    return feature_map_2d


def project_3d_to_2d_bilinear(
    features_3d: ME.SparseTensor,
    coords_3d: List[np.ndarray],
    camera_intrinsic: np.ndarray,
    camera_extrinsic: Optional[np.ndarray] = None,
    image_shape: Tuple[int, int] = (1024, 2048),
    voxel_size: float = 0.05,
) -> torch.Tensor:
    """
    使用双线性插值将3D特征投影到2D图像坐标（更平滑的版本）
    
    参数同上，但使用双线性插值而不是最近邻
    """
    batch_size = len(coords_3d)
    feature_dim = features_3d.F.shape[1]
    H, W = image_shape
    
    device = features_3d.device
    feature_map_2d = torch.zeros(
        (batch_size, feature_dim, H, W),
        dtype=features_3d.F.dtype,
        device=device,
    )
    
    batch_coords, batch_feats = features_3d.decomposed_coordinates_and_features
    
    for batch_idx in range(batch_size):
        coords_3d_batch = coords_3d[batch_idx]
        if batch_idx >= len(batch_coords):
            continue
        
        coords_sparse = batch_coords[batch_idx]
        feats_sparse = batch_feats[batch_idx]
        
        coords_3d_quantized = coords_sparse.cpu().numpy() * voxel_size
        
        if camera_extrinsic is not None:
            coords_homo = np.column_stack([
                coords_3d_quantized,
                np.ones(len(coords_3d_quantized))
            ])
            coords_camera = (camera_extrinsic @ coords_homo.T).T[:, :3]
        else:
            coords_camera = coords_3d_quantized
        
        X, Y, Z = coords_camera[:, 0], coords_camera[:, 1], coords_camera[:, 2]
        valid_mask = Z > 0.1
        
        if valid_mask.sum() == 0:
            continue
        
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        Z_valid = Z[valid_mask]
        feats_valid = feats_sparse[valid_mask]
        
        # 投影到像素坐标（浮点数）
        u_float = camera_intrinsic[0, 0] * X_valid / Z_valid + camera_intrinsic[0, 2]
        v_float = camera_intrinsic[1, 1] * Y_valid / Z_valid + camera_intrinsic[1, 2]
        
        # 过滤超出范围的点
        in_bounds = (u_float >= 0) & (u_float < W) & (v_float >= 0) & (v_float < H)
        u_float = u_float[in_bounds]
        v_float = v_float[in_bounds]
        feats_valid = feats_valid[in_bounds]
        
        if len(u_float) == 0:
            continue
        
        # 双线性插值
        u0 = np.floor(u_float).astype(np.int32)
        u1 = np.minimum(u0 + 1, W - 1)
        v0 = np.floor(v_float).astype(np.int32)
        v1 = np.minimum(v0 + 1, H - 1)
        
        wu = u_float - u0
        wv = v_float - v0
        
        # 计算权重
        w00 = (1 - wu) * (1 - wv)
        w01 = (1 - wu) * wv
        w10 = wu * (1 - wv)
        w11 = wu * wv
        
        # 加权填充
        for i in range(len(u_float)):
            feat = feats_valid[i]
            feature_map_2d[batch_idx, :, v0[i], u0[i]] += feat * w00[i]
            feature_map_2d[batch_idx, :, v1[i], u0[i]] += feat * w01[i]
            feature_map_2d[batch_idx, :, v0[i], u1[i]] += feat * w10[i]
            feature_map_2d[batch_idx, :, v1[i], u1[i]] += feat * w11[i]
    
    return feature_map_2d

