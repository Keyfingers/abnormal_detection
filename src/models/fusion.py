"""
融合模块：融合2D语义特征和3D几何特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class GatedAdapterFusionHead(nn.Module):
    """
    门控适配器融合头：使用门控机制和瓶颈适配器融合特征
    符合 "Feature Splatting + Adapter" 范式
    """
    def __init__(
        self,
        feature_2d_dim: int = 256,
        feature_3d_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # 1. 特征对齐 (Projectors)
        self.align_2d = nn.Conv2d(feature_2d_dim, hidden_dim, kernel_size=1)
        self.align_3d = nn.Conv2d(feature_3d_dim, hidden_dim, kernel_size=1)
        self.bn_2d = nn.BatchNorm2d(hidden_dim)
        self.bn_3d = nn.BatchNorm2d(hidden_dim)
        
        # 2. 门控网络 (Gating Mechanism)
        # 决定在每个像素点更信任哪个模态
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()  # 输出 [0, 1] 权重
        )
        
        # 3. 瓶颈适配器 (Bottleneck Adapter)
        # 用于参数高效微调 (PEFT)
        self.adapter = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1), # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1), # 升维
            nn.Sigmoid() # 输出作为缩放因子
        )
        
        # 4. 预测头 (Prediction Head)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid() # 输出异常概率
        )

    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
    ) -> torch.Tensor:
        # 确保尺寸一致
        if features_2d.shape[-2:] != features_3d.shape[-2:]:
            features_3d = F.interpolate(
                features_3d,
                size=features_2d.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            
        # 对齐特征
        f2d = F.relu(self.bn_2d(self.align_2d(features_2d)))
        f3d = F.relu(self.bn_3d(self.align_3d(features_3d)))
        
        # 计算门控权重
        # Gate 接近 1 表示信任 2D，接近 0 表示信任 3D
        gate_input = torch.cat([f2d, f3d], dim=1)
        gate = self.gate_net(gate_input)
        
        # 加权融合
        f_fused = gate * f2d + (1 - gate) * f3d
        
        # 通过适配器增强 (Residual Connection)
        # Adapter 输出一个缩放因子，调制融合特征
        adapter_scale = self.adapter(f_fused)
        f_enhanced = f_fused * (1 + adapter_scale)
        
        # 预测
        return self.head(f_enhanced)


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
        
        # 3D-2D投影（使用 Gaussian Splatting）
        from ..utils.projection import project_3d_to_2d_gaussian
        
        # 修复：确保相机参数在正确的设备和格式
        device = images.device
        
        if isinstance(camera_intrinsic, torch.Tensor):
            camera_intrinsic_np = camera_intrinsic.cpu().numpy()
        else:
            camera_intrinsic_np = np.array(camera_intrinsic)
        
        if camera_extrinsic is not None:
            if isinstance(camera_extrinsic, torch.Tensor):
                camera_extrinsic_np = camera_extrinsic.cpu().numpy()
            else:
                camera_extrinsic_np = np.array(camera_extrinsic)
        else:
            camera_extrinsic_np = None
        
        # 调用 Gaussian Splatting
        features_3d_2d = project_3d_to_2d_gaussian(
            features_3d_sparse,
            coords_3d,
            camera_intrinsic_np,
            camera_extrinsic_np,
            image_shape=(images.shape[2], images.shape[3]),
            voxel_size=self.geometric_3d.voxel_size,
            kernel_radius=3, # 可配置：对应高斯核半径
            sigma=1.0        # 可配置：高斯方差
        )
        
        # 确保特征图在正确的设备上
        if features_3d_2d.device != device:
            features_3d_2d = features_3d_2d.to(device)
        
        # 融合
        fusion_score = self.fusion_head(features_2d, features_3d_2d)
        
        results = {
            'fusion_score': fusion_score.squeeze(1),  # (B, H, W)
        }
        
        if return_individual_scores:
            results['rba_score'] = semantic_results['rba_score']
            results['reconstruction_error'] = geometric_results['reconstruction_error']
        
        return results
