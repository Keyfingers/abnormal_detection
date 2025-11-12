"""
融合模块：融合2D语义特征和3D几何特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FusionHead(nn.Module):
    """
    融合头：轻量级的2D卷积网络，融合2D和3D特征
    
    输入：
    - 2D语义特征图 (B, C_2D, H, W)
    - 3D投影特征图 (B, C_3D, H, W)
    
    输出：
    - 融合异常分数图 (B, 1, H, W)
    """
    
    def __init__(
        self,
        feature_2d_dim: int = 256,
        feature_3d_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        """
        Args:
            feature_2d_dim: 2D特征维度
            feature_3d_dim: 3D特征维度
            hidden_dim: 隐藏层维度
            num_layers: 卷积层数量
        """
        super().__init__()
        
        input_dim = feature_2d_dim + feature_3d_dim
        
        layers = []
        in_channels = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_channels = hidden_dim
        
        # 最终输出层：输出单通道异常分数
        layers.append(
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features_2d: 2D语义特征图 (B, C_2D, H, W)
            features_3d: 3D投影特征图 (B, C_3D, H, W)
        
        Returns:
            融合异常分数图 (B, 1, H, W)
        """
        # 确保特征图尺寸一致
        if features_2d.shape[-2:] != features_3d.shape[-2:]:
            # 上采样3D特征图到2D特征图尺寸
            features_3d = F.interpolate(
                features_3d,
                size=features_2d.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        
        # 沿通道维度拼接
        fused_features = torch.cat([features_2d, features_3d], dim=1)  # (B, C_2D+C_3D, H, W)
        
        # 通过融合网络
        anomaly_score = self.network(fused_features)  # (B, 1, H, W)
        
        return anomaly_score


class FusionModel(nn.Module):
    """
    完整的融合模型：包含2D分支、3D分支和融合头
    """
    
    def __init__(
        self,
        semantic_2d_model: nn.Module,
        geometric_3d_model: nn.Module,
        fusion_head: nn.Module,
        freeze_2d: bool = True,
        freeze_3d: bool = True,
    ):
        """
        Args:
            semantic_2d_model: 2D语义分支模型
            geometric_3d_model: 3D几何分支模型
            fusion_head: 融合头
            freeze_2d: 是否冻结2D分支
            freeze_3d: 是否冻结3D分支
        """
        super().__init__()
        
        self.semantic_2d = semantic_2d_model
        self.geometric_3d = geometric_3d_model
        self.fusion_head = fusion_head
        
        # 冻结分支（只训练融合头）
        if freeze_2d:
            for param in self.semantic_2d.parameters():
                param.requires_grad = False
        
        if freeze_3d:
            for param in self.geometric_3d.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
        point_clouds: list,
        camera_intrinsic: torch.Tensor,
        camera_extrinsic: Optional[torch.Tensor] = None,
        return_individual_scores: bool = False,
    ) -> dict:
        """
        前向传播
        
        Args:
            images: 输入图像 (B, 3, H, W)
            point_clouds: 点云列表
            camera_intrinsic: 相机内参 (B, 3, 3) 或 (3, 3)
            camera_extrinsic: 相机外参 (B, 4, 4) 或 (4, 4)，可选
            return_individual_scores: 是否返回单独的2D和3D评分
        
        Returns:
            Dict包含：
            - 'fusion_score': 融合异常分数 (B, 1, H, W)
            - 'rba_score': RbA评分 (B, H, W) [可选]
            - 'reconstruction_error': 重建误差 [可选]
        """
        # 2D语义分支
        semantic_results = self.semantic_2d(
            images,
            return_features=True,
            return_rba_score=return_individual_scores,
        )
        features_2d = semantic_results['features_2d']
        
        # 3D几何分支
        geometric_results = self.geometric_3d(
            point_clouds,
            return_features=True,
            return_reconstruction_error=return_individual_scores,
        )
        features_3d_sparse = geometric_results['features_3d']
        coords_3d = geometric_results['coords_list']
        
        # 3D-2D投影
        from ..utils.projection import project_3d_to_2d_bilinear
        
        if isinstance(camera_intrinsic, torch.Tensor):
            camera_intrinsic_np = camera_intrinsic.cpu().numpy()
        else:
            camera_intrinsic_np = camera_intrinsic
        
        if camera_extrinsic is not None:
            if isinstance(camera_extrinsic, torch.Tensor):
                camera_extrinsic_np = camera_extrinsic.cpu().numpy()
            else:
                camera_extrinsic_np = camera_extrinsic
        else:
            camera_extrinsic_np = None
        
        features_3d_2d = project_3d_to_2d_bilinear(
            features_3d_sparse,
            coords_3d,
            camera_intrinsic_np,
            camera_extrinsic_np,
            image_shape=(images.shape[2], images.shape[3]),
            voxel_size=self.geometric_3d.voxel_size,
        )
        
        # 融合
        fusion_score = self.fusion_head(features_2d, features_3d_2d)
        
        results = {
            'fusion_score': fusion_score.squeeze(1),  # (B, H, W)
        }
        
        if return_individual_scores:
            results['rba_score'] = semantic_results['rba_score']
            results['reconstruction_error'] = geometric_results['reconstruction_error']
        
        return results

