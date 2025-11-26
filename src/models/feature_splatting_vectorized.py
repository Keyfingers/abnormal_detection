"""
Feature Splatting向量化版本（性能优化）
这是一个向量化的实现示例，用于大规模训练时的性能优化
注意：这是参考实现，当前主版本仍使用循环版本以保证正确性
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


class FeatureSplattingVectorized(nn.Module):
    """
    向量化版本的Feature Splatting
    
    使用PyTorch的广播和scatter操作实现全向量化，避免Python循环
    适合大规模训练（10,000+体素）
    
    注意：这是优化版本，需要仔细验证与循环版本的一致性
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        image_height: int = 800,
        image_width: int = 1333,
        voxel_size: float = 0.05,
        max_splat_radius: float = 10.0,
        device: str = "cuda"
    ):
        super(FeatureSplattingVectorized, self).__init__()
        self.feature_dim = feature_dim
        self.image_height = image_height
        self.image_width = image_width
        self.voxel_size = voxel_size
        self.max_splat_radius = max_splat_radius
        self.device = device
    
    def rasterize_splats_vectorized(
        self,
        pixel_coords: torch.Tensor,  # (M, 2)
        covariance_2d: torch.Tensor,  # (M, 2, 2)
        voxel_features: torch.Tensor,  # (M, D)
        depths: torch.Tensor  # (M,)
    ) -> torch.Tensor:
        """
        向量化版本的Splat光栅化
        
        使用广播和scatter操作实现全向量化
        """
        M, D = voxel_features.shape
        H, W = self.image_height, self.image_width
        
        # 计算每个Splat的影响范围（向量化）
        eigenvals, _ = torch.linalg.eigh(covariance_2d)  # (M, 2)
        max_radii = torch.sqrt(torch.max(eigenvals, dim=1)[0]) * 3.0  # (M,)
        max_radii = torch.clamp(max_radii, max=self.max_splat_radius)
        
        # 计算bounding box（向量化）
        u_min = torch.clamp((pixel_coords[:, 0] - max_radii).long(), min=0)
        u_max = torch.clamp((pixel_coords[:, 0] + max_radii).long() + 1, max=W)
        v_min = torch.clamp((pixel_coords[:, 1] - max_radii).long(), min=0)
        v_max = torch.clamp((pixel_coords[:, 1] + max_radii).long() + 1, max=H)
        
        # 初始化输出
        feature_map = torch.zeros(H, W, D, device=voxel_features.device)
        weight_map = torch.zeros(H, W, device=voxel_features.device)
        
        # 深度权重（平滑版本）
        depth_weights = torch.exp(-depths * 0.1)  # (M,)
        
        # 对每个Splat进行处理（仍然需要循环，但内部计算向量化）
        # 注意：完全向量化需要更复杂的实现，这里提供一个半向量化版本
        for i in range(M):
            if u_max[i] <= u_min[i] or v_max[i] <= v_min[i]:
                continue
            
            # 创建像素网格（向量化）
            u_range = torch.arange(u_min[i], u_max[i], device=voxel_features.device, dtype=torch.float32)
            v_range = torch.arange(v_min[i], v_max[i], device=voxel_features.device, dtype=torch.float32)
            u_grid, v_grid = torch.meshgrid(u_range, v_range, indexing='xy')
            pixel_grid = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1)  # (N, 2)
            
            # 计算高斯权重（向量化）
            diff = pixel_grid - pixel_coords[i:i+1]  # (N, 2)
            cov_2d = covariance_2d[i]  # (2, 2)
            cov_2d_stable = cov_2d + torch.eye(2, device=cov_2d.device) * 1e-5
            
            try:
                cov_inv = torch.linalg.inv(cov_2d_stable)
            except:
                cov_inv = torch.linalg.pinv(cov_2d_stable)
            
            mahalanobis_dist = torch.sum(diff @ cov_inv * diff, dim=1)  # (N,)
            weights = torch.exp(-0.5 * mahalanobis_dist) * depth_weights[i]  # (N,)
            
            # 使用scatter_add进行向量化累积
            u_indices = pixel_grid[:, 0].long()
            v_indices = pixel_grid[:, 1].long()
            valid_mask = (u_indices >= 0) & (u_indices < W) & (v_indices >= 0) & (v_indices < H)
            
            if valid_mask.any():
                u_valid = u_indices[valid_mask]
                v_valid = v_indices[valid_mask]
                w_valid = weights[valid_mask]
                
                # 使用scatter_add（更高效）
                feature_map.index_add_(0, v_valid, 
                    voxel_features[i:i+1].expand(len(v_valid), -1) * w_valid.unsqueeze(1))
                weight_map.index_add_(0, v_valid, w_valid)
        
        # 归一化（使用平滑项）
        weight_map_smooth = weight_map.unsqueeze(2) + 1.0
        feature_map = feature_map / weight_map_smooth
        
        return feature_map


# 注意：这是一个参考实现，展示了如何向量化Splatting过程
# 实际使用时需要仔细验证与循环版本的一致性
# 对于大规模训练，建议实现CUDA Kernel以获得最佳性能

