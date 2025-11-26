"""
Feature Splatting模块
实现3D体素特征到2D特征图的可微投影和聚合
核心创新：使用高斯Splatting机制实现稀疏到稠密的特征对齐
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union
import math


class FeatureSplatting(nn.Module):
    """
    Feature Splatting模块：将3D体素特征投影到2D特征图
    
    核心思想：
    1. 将每个体素建模为3D高斯分布（椭球）
    2. 通过相机投影变换，将3D高斯投影到2D平面
    3. 使用2D高斯权重进行特征聚合，生成稠密的2D特征图
    
    技术细节：
    - 输入：体素特征 (M, D) 和体素坐标 (M, 3)
    - 输出：2D特征图 (H, W, D)
    - 投影矩阵：相机内参 + 外参（LiDAR到相机）
    - 可微性：整个过程对位置和协方差可导
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        image_height: int = 800,
        image_width: int = 1333,
        voxel_size: float = 0.05,
        splat_radius: float = 2.0,  # Splat的像素半径
        learnable_covariance: bool = True,  # 是否学习协方差
        device: str = "cuda"
    ):
        """
        初始化Feature Splatting模块
        
        Args:
            feature_dim: 特征维度，默认128（与Geometric3DBranch输出一致）
            image_height: 图像高度，默认800
            image_width: 图像宽度，默认1333
            voxel_size: 体素尺寸（米），默认0.05（5cm）
            splat_radius: Splat的像素半径，默认2.0
            learnable_covariance: 是否学习协方差矩阵，默认True
            device: 计算设备
        """
        super(FeatureSplatting, self).__init__()
        
        self.feature_dim = feature_dim
        self.image_height = image_height
        self.image_width = image_width
        self.voxel_size = voxel_size
        self.splat_radius = splat_radius
        self.learnable_covariance = learnable_covariance
        self.device = device
        
        # 如果可学习，初始化协方差参数
        if learnable_covariance:
            # 为每个体素学习一个缩放因子
            # 实际协方差 = base_covariance * scale
            self.covariance_scale = nn.Parameter(torch.ones(1) * 1.0)
        else:
            self.register_buffer('covariance_scale', torch.ones(1))
    
    def compute_3d_covariance(
        self,
        voxel_coords: torch.Tensor,
        local_points: Optional[torch.Tensor] = None,
        road_aware: bool = True
    ) -> torch.Tensor:
        """
        计算3D体素的协方差矩阵
        
        将体素建模为3D高斯分布，协方差矩阵描述其形状和方向
        
        Args:
            voxel_coords: 体素坐标 (M, 3)，整数坐标
            local_points: 局部点云 (可选)，用于计算几何协方差
            road_aware: 是否考虑道路扁平特点（高度轴方差小），默认True
            
        Returns:
            covariance_3d: 3D协方差矩阵 (M, 3, 3)
        """
        M = voxel_coords.shape[0]
        
        if local_points is not None:
            # 如果有局部点云，计算几何协方差
            # 这里简化处理，实际可以从点云分布计算
            base_cov = torch.eye(3, device=voxel_coords.device).unsqueeze(0) * (self.voxel_size ** 2)
        else:
            # 默认：考虑道路扁平特点
            base_cov = torch.eye(3, device=voxel_coords.device).unsqueeze(0).repeat(M, 1, 1)
            
            if road_aware:
                # 道路通常是扁平的：高度轴（通常是Y轴或Z轴，取决于坐标系）方差小
                # 水平面（X, Z）方差大
                # 假设坐标系：X前，Y左，Z上（KITTI LiDAR标准）
                # 高度轴是Z轴，水平面是X-Y平面
                base_cov[:, 0, 0] = (self.voxel_size / 2.0) ** 2  # X轴（前）
                base_cov[:, 1, 1] = (self.voxel_size / 2.0) ** 2  # Y轴（左）
                base_cov[:, 2, 2] = (self.voxel_size / 4.0) ** 2  # Z轴（上，高度轴，方差更小）
            else:
                # 各向同性高斯
                base_cov = base_cov * (self.voxel_size / 2.0) ** 2
        
        # 应用可学习的缩放
        covariance_3d = base_cov * self.covariance_scale
        
        return covariance_3d
    
    def project_3d_to_2d(
        self,
        voxel_coords: torch.Tensor,
        projection_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将3D体素坐标投影到2D图像平面
        
        Args:
            voxel_coords: 体素坐标 (M, 3)，世界坐标系（米）
            projection_matrix: 投影矩阵 (3, 4) 或 (4, 4)，包含内参和外参
            
        Returns:
            pixel_coords: 2D像素坐标 (M, 2)，浮点数
            depths: 深度值 (M,)，用于深度排序
            cam_coords: 相机坐标系坐标 (M, 3)，用于雅可比矩阵计算
        """
        M = voxel_coords.shape[0]
        
        # 将体素坐标转换为世界坐标（体素中心）
        world_coords = voxel_coords.float() * self.voxel_size  # (M, 3)
        
        # 转换为齐次坐标
        ones = torch.ones(M, 1, device=voxel_coords.device)
        world_coords_homo = torch.cat([world_coords, ones], dim=1)  # (M, 4)
        
        # 处理投影矩阵维度
        if projection_matrix.shape == (3, 4):
            proj_mat = projection_matrix
        elif projection_matrix.shape == (4, 4):
            proj_mat = projection_matrix[:3, :]
        else:
            raise ValueError(f"投影矩阵形状错误: {projection_matrix.shape}")
        
        # 投影到相机坐标系
        cam_coords = (proj_mat @ world_coords_homo.T).T  # (M, 3)
        
        # 提取深度（z坐标，相机坐标系中Z是深度）
        depths = cam_coords[:, 2]  # (M,)
        
        # 过滤相机背后的点（深度 <= 0 的点没有意义）
        # 注意：这里不直接clamp，而是在forward中过滤
        # depths = torch.clamp(depths, min=1e-6)  # 移除这行，改为在forward中过滤
        
        # 投影到2D像素坐标
        # 避免除零，但只对有效点计算
        valid_mask = depths > 0.1  # 最小深度阈值（10cm）
        pixel_coords = torch.zeros(M, 2, device=voxel_coords.device)
        if valid_mask.any():
            pixel_coords[valid_mask] = cam_coords[valid_mask, :2] / depths[valid_mask].unsqueeze(1)
        
        return pixel_coords, depths, cam_coords
    
    def compute_2d_covariance(
        self,
        covariance_3d: torch.Tensor,
        cam_coords: torch.Tensor,
        projection_matrix: torch.Tensor,
        depths: torch.Tensor
    ) -> torch.Tensor:
        """
        将3D协方差矩阵投影到2D平面
        
        使用雅可比矩阵进行线性近似：
        Σ_2d = J @ Σ_3d @ J^T
        
        修正：使用真实的相机坐标 (X, Y, Z) 而不是像素坐标 (u, v)
        
        Args:
            covariance_3d: 3D协方差矩阵 (M, 3, 3)
            cam_coords: 相机坐标系坐标 (M, 3)，包含 [X, Y, Z]
            projection_matrix: 投影矩阵 (3, 4)
            depths: 深度值 (M,)，即 cam_coords[:, 2]
            
        Returns:
            covariance_2d: 2D协方差矩阵 (M, 2, 2)
        """
        M = covariance_3d.shape[0]
        
        # 提取投影矩阵的内参部分（前3x3）
        if projection_matrix.shape == (3, 4):
            K = projection_matrix[:, :3]  # 内参矩阵
        else:
            K = projection_matrix[:3, :3]
        
        # 提取内参
        fx = K[0, 0]
        fy = K[1, 1]
        
        # 提取相机坐标
        X = cam_coords[:, 0]  # (M,)
        Y = cam_coords[:, 1]  # (M,)
        Z = depths  # (M,)，即 cam_coords[:, 2]
        
        # 构建雅可比矩阵 (M, 2, 3)
        # 对于透视投影：u = fx * X/Z + cx, v = fy * Y/Z + cy
        # 雅可比矩阵：
        # J = [fx/Z,    0,      -fx*X/Z^2]
        #     [0,        fy/Z,   -fy*Y/Z^2]
        J = torch.zeros(M, 2, 3, device=covariance_3d.device)
        
        # 避免除零
        Z_safe = torch.clamp(Z, min=1e-6)
        
        # 填充雅可比矩阵
        J[:, 0, 0] = fx / Z_safe  # ∂u/∂X
        J[:, 0, 2] = -(fx * X) / (Z_safe ** 2)  # ∂u/∂Z
        J[:, 1, 1] = fy / Z_safe  # ∂v/∂Y
        J[:, 1, 2] = -(fy * Y) / (Z_safe ** 2)  # ∂v/∂Z
        
        # 投影协方差：Σ_2d = J @ Σ_3d @ J^T
        covariance_2d = torch.bmm(torch.bmm(J, covariance_3d), J.transpose(1, 2))  # (M, 2, 2)
        
        # 添加最小协方差以确保数值稳定性（改进：使用更小的epsilon）
        min_cov = torch.eye(2, device=covariance_2d.device).unsqueeze(0) * 1e-5
        covariance_2d = covariance_2d + min_cov
        
        return covariance_2d
    
    def rasterize_splats(
        self,
        pixel_coords: torch.Tensor,
        covariance_2d: torch.Tensor,
        voxel_features: torch.Tensor,
        depths: torch.Tensor
    ) -> torch.Tensor:
        """
        光栅化Splats，生成稠密的2D特征图
        
        对图像上的每个像素，计算所有覆盖该像素的Splat的加权和
        
        性能说明：
        - 当前实现使用Python循环，适合MVP版本
        - 对于大规模训练（10,000+体素），建议使用向量化实现或CUDA Kernel
        - 可以通过设置较小的max_splat_radius来限制计算量
        
        Args:
            pixel_coords: 2D像素坐标 (M, 2)
            covariance_2d: 2D协方差矩阵 (M, 2, 2)
            voxel_features: 体素特征 (M, D)
            depths: 深度值 (M,)，用于深度排序和混合
            
        Returns:
            feature_map_2d: 2D特征图 (H, W, D)
        """
        M, D = voxel_features.shape
        H, W = self.image_height, self.image_width
        
        # 初始化输出特征图
        feature_map = torch.zeros(H, W, D, device=voxel_features.device)
        weight_map = torch.zeros(H, W, device=voxel_features.device)
        
        # 按深度排序（从远到近，后处理时从近到远）
        sorted_indices = torch.argsort(depths, descending=True)
        
        # 性能优化：限制最大Splat半径，避免过大的计算量
        # 注意：对于大规模训练，这个Python循环会成为瓶颈
        # 建议后续实现向量化版本或CUDA Kernel
        max_splat_radius = min(self.splat_radius * 3, 10.0)  # 最大10像素
        
        # 对每个体素进行Splat
        # TODO: 性能优化 - 实现向量化版本或CUDA Kernel以加速大规模训练
        for idx in sorted_indices:
            pixel = pixel_coords[idx]  # (2,)
            cov_2d = covariance_2d[idx]  # (2, 2)
            feat = voxel_features[idx]  # (D,)
            depth = depths[idx]
            
            # 计算Splat的影响范围（3σ原则）
            # 计算协方差矩阵的特征值，确定影响范围
            eigenvals, _ = torch.linalg.eigh(cov_2d)
            max_radius = min(torch.sqrt(torch.max(eigenvals)) * 3.0, max_splat_radius)  # 限制最大半径
            
            # 计算像素范围
            u, v = pixel[0].item(), pixel[1].item()
            u_min = max(0, int(u - max_radius))
            u_max = min(W, int(u + max_radius) + 1)
            v_min = max(0, int(v - max_radius))
            v_max = min(H, int(v + max_radius) + 1)
            
            if u_max <= u_min or v_max <= v_min:
                continue
            
            # 创建像素网格（向量化）
            u_range = torch.arange(u_min, u_max, device=voxel_features.device, dtype=torch.float32)
            v_range = torch.arange(v_min, v_max, device=voxel_features.device, dtype=torch.float32)
            u_grid, v_grid = torch.meshgrid(u_range, v_range, indexing='xy')
            pixel_grid = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1)  # (N, 2)
            
            # 计算高斯权重（向量化）
            diff = pixel_grid - pixel.unsqueeze(0)  # (N, 2)
            
            # 计算2D高斯值：exp(-0.5 * (x-μ)^T @ Σ^(-1) @ (x-μ))
            # 改进数值稳定性：添加小的正则项
            cov_2d_stable = cov_2d + torch.eye(2, device=cov_2d.device) * 1e-5
            try:
                cov_inv = torch.linalg.inv(cov_2d_stable)  # (2, 2)
            except:
                # 如果矩阵奇异，使用伪逆
                cov_inv = torch.linalg.pinv(cov_2d_stable)
            
            # 计算马氏距离（向量化）
            mahalanobis_dist = torch.sum(diff @ cov_inv * diff, dim=1)  # (N,)
            
            # 计算高斯权重
            weights = torch.exp(-0.5 * mahalanobis_dist)  # (N,)
            
            # 深度权重处理（修正：避免数值不稳定）
            # 原方案：depth_weight = 1.0 / (depth + 1e-6) 会导致深度很小时权重爆炸
            # 新方案：使用exp(-depth)或直接去掉，因为高斯权重已经通过协方差矩阵反映了投影大小
            # 透视投影本身已经隐含了近大远小的关系，不需要显式除以深度
            # 如果需要遮挡处理，应该使用Alpha Blending或Softmax，而不是简单的深度权重
            # 这里使用平滑的深度权重，避免数值爆炸
            depth_weight = torch.exp(-depth * 0.1)  # 平滑的深度权重，避免爆炸
            weights = weights * depth_weight
            
            # 累积特征和权重（向量化索引）
            u_indices = pixel_grid[:, 0].long()  # (N,)
            v_indices = pixel_grid[:, 1].long()  # (N,)
            
            # 过滤有效索引
            valid_mask = (u_indices >= 0) & (u_indices < W) & (v_indices >= 0) & (v_indices < H)
            if valid_mask.any():
                u_valid = u_indices[valid_mask]
                v_valid = v_indices[valid_mask]
                w_valid = weights[valid_mask]
                
                # 使用循环累积（当前版本）
                # TODO: 性能优化 - 可以使用scatter_add进行向量化，但需要仔细处理维度
                # 参考实现见 feature_splatting_vectorized.py
                for j in range(len(u_valid)):
                    feature_map[v_valid[j], u_valid[j]] += feat * w_valid[j]
                    weight_map[v_valid[j], u_valid[j]] += w_valid[j]
        
        # 特征归一化（修正：避免特征值幅度波动）
        # 原方案：feature_map = feature_map / weight_map 会导致加权平均
        # 问题：如果像素只有一个点在边缘（权重0.1），除以权重后特征值会被放大10倍
        # 这会导致特征值的幅度随投影位置剧烈波动，不利于后续训练
        # 
        # 新方案：使用加权累加 + 平滑项，避免特征值幅度波动
        # 类似Smooth L1，确保特征值幅度稳定
        weight_map_smooth = weight_map.unsqueeze(2) + 1.0  # 平滑项，避免除零和幅度波动
        feature_map = feature_map / weight_map_smooth
        
        return feature_map
    
    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        projection_matrix: torch.Tensor,
        local_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播：将3D体素特征投影到2D特征图
        
        Args:
            voxel_features: 体素特征 (M, D)
            voxel_coords: 体素坐标 (M, 3)，整数坐标（体素索引）
            projection_matrix: 投影矩阵 (3, 4) 或 (4, 4)
            local_points: 局部点云（可选），用于计算几何协方差
            
        Returns:
            feature_map_2d: 2D特征图 (H, W, D)
        """
        # 确保投影矩阵在正确的设备上，并确保是float类型
        if isinstance(projection_matrix, np.ndarray):
            projection_matrix = torch.from_numpy(projection_matrix).float().to(voxel_features.device)
        else:
            projection_matrix = projection_matrix.float().to(voxel_features.device)
        
        # 1. 计算3D协方差矩阵
        covariance_3d = self.compute_3d_covariance(voxel_coords, local_points)  # (M, 3, 3)
        
        # 2. 投影3D坐标到2D像素坐标
        pixel_coords, depths, cam_coords = self.project_3d_to_2d(voxel_coords, projection_matrix)  # (M, 2), (M,), (M, 3)
        
        # 3. 过滤无效体素：深度过滤 + 图像范围过滤
        # 深度过滤：只保留相机前方的点（深度 > 0.1米，避免数值问题）
        depth_mask = depths > 0.1
        
        # 图像范围过滤
        image_mask = (
            (pixel_coords[:, 0] >= 0) & 
            (pixel_coords[:, 0] < self.image_width) &
            (pixel_coords[:, 1] >= 0) & 
            (pixel_coords[:, 1] < self.image_height)
        )
        
        valid_mask = depth_mask & image_mask
        
        if not valid_mask.any():
            # 如果没有有效体素，返回零特征图
            return torch.zeros(
                self.image_height, 
                self.image_width, 
                self.feature_dim,
                device=voxel_features.device
            )
        
        voxel_features = voxel_features[valid_mask]
        voxel_coords = voxel_coords[valid_mask]
        pixel_coords = pixel_coords[valid_mask]
        depths = depths[valid_mask]
        cam_coords = cam_coords[valid_mask]
        covariance_3d = covariance_3d[valid_mask]
        
        # 4. 计算2D协方差矩阵（使用真实的相机坐标）
        covariance_2d = self.compute_2d_covariance(
            covariance_3d, cam_coords, projection_matrix, depths
        )  # (M, 2, 2)
        
        # 5. 光栅化Splats，生成稠密特征图
        feature_map_2d = self.rasterize_splats(
            pixel_coords, covariance_2d, voxel_features, depths
        )  # (H, W, D)
        
        return feature_map_2d

