"""
端到端异常检测模型
整合所有阶段：Semantic2D + Geometric3D + FeatureSplatting + FusionHead
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.semantic_2d import Semantic2DBranch
from src.models.geometric_3d import Geometric3DBranch
from src.models.feature_splatting import FeatureSplatting
from src.models.fusion_head import FusionHead


class AnomalyDetector(nn.Module):
    """
    端到端异常检测模型
    
    架构流程：
    1. Semantic2DBranch: 提取2D语义特征（冻结）
    2. Geometric3DBranch: 提取3D几何特征（冻结）
    3. FeatureSplatting: 3D到2D投影（冻结）
    4. FusionHead: 特征融合和异常判定（可训练）
    
    训练策略：
    - 仅训练FusionHead（<5%参数量）
    - 使用伪异常生成避免模型崩溃
    """
    
    def __init__(
        self,
        # Semantic2D配置（必需参数）
        mask2former_config_path: str,
        mask2former_checkpoint_path: str,
        
        # Geometric3D配置（必需参数）
        minkunet_checkpoint_path: str,
        
        # 可选参数（有默认值）
        img_feature_dim: int = 256,
        pts_feature_dim: int = 128,
        voxel_size: float = 0.05,
        image_height: int = 800,
        image_width: int = 1333,
        fusion_hidden_dim: int = 64,
        use_cross_attention: bool = False,
        use_gating: bool = True,
        device: str = "cuda"
    ):
        """
        初始化端到端模型
        
        Args:
            mask2former_config_path: Mask2Former配置文件路径
            mask2former_checkpoint_path: Mask2Former权重路径
            img_feature_dim: 图像特征维度，默认256
            minkunet_checkpoint_path: MinkUNet权重路径
            pts_feature_dim: 点云特征维度，默认128
            voxel_size: 体素尺寸，默认0.05
            image_height: 图像高度，默认800
            image_width: 图像宽度，默认1333
            fusion_hidden_dim: 融合头隐藏维度，默认64
            use_cross_attention: 是否使用交叉注意力，默认False
            use_gating: 是否使用门控机制，默认True
            device: 计算设备
        """
        super(AnomalyDetector, self).__init__()
        
        self.device = device
        
        # 阶段一：2D语义分支（冻结）
        self.semantic_2d = Semantic2DBranch(
            config_path=mask2former_config_path,
            checkpoint_path=mask2former_checkpoint_path,
            freeze_backbone=True,
            feature_dim=img_feature_dim,
            device=device
        )
        
        # 阶段二：3D几何分支（冻结）
        self.geometric_3d = Geometric3DBranch(
            checkpoint_path=minkunet_checkpoint_path,
            freeze_backbone=True,
            feature_dim=pts_feature_dim,
            voxel_size=voxel_size,
            device=device
        )
        
        # 阶段三：Feature Splatting（冻结）
        self.feature_splatting = FeatureSplatting(
            feature_dim=pts_feature_dim,
            image_height=image_height,
            image_width=image_width,
            voxel_size=voxel_size,
            device=device
        )
        
        # 阶段四：融合头（可训练）
        self.fusion_head = FusionHead(
            img_feature_dim=img_feature_dim,
            pts_feature_dim=pts_feature_dim,
            hidden_dim=fusion_hidden_dim,
            use_cross_attention=use_cross_attention,
            use_gating=use_gating,
            device=device
        )
        
        # 冻结前三个阶段
        self._freeze_backbones()
    
    def _freeze_backbones(self):
        """冻结前三个阶段的参数"""
        # Semantic2D已经冻结
        # Geometric3D已经冻结
        # FeatureSplatting冻结（如果learnable_covariance=False）
        for param in self.semantic_2d.parameters():
            param.requires_grad = False
        for param in self.geometric_3d.parameters():
            param.requires_grad = False
        # FeatureSplatting的协方差参数可能需要训练，这里先冻结
        for name, param in self.feature_splatting.named_parameters():
            if 'covariance_scale' in name:
                # 协方差缩放可以训练（可选）
                param.requires_grad = False  # 默认冻结
            else:
                param.requires_grad = False
    
    def extract_features(
        self,
        images: torch.Tensor,
        point_clouds,
        projection_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取2D和3D特征（用于训练时的伪异常生成）
        
        Args:
            images: 输入图像 (B, C, H, W)
            point_clouds: 点云数据
            projection_matrix: 投影矩阵 (3, 4) 或 (4, 4)
            
        Returns:
            img_features: 2D特征 (B, C_img, H', W')
            pts_features_proj: 3D投影特征 (B, C_pts, H', W')
        """
        with torch.no_grad():
            # 提取2D特征
            img_features = self.semantic_2d(images)  # (B, C_img, H', W')
            
            # 提取3D特征
            output_3d = self.geometric_3d(point_clouds)
            voxel_features = output_3d['voxel_features']  # (M, C_pts)
            voxel_coords = output_3d['voxel_coords']  # (M, 3)
            
            # Feature Splatting投影
            pts_features_proj = self.feature_splatting(
                voxel_features,
                voxel_coords,
                projection_matrix
            )  # (H, W, C_pts)
            
            # 转换为CHW格式
            if pts_features_proj.dim() == 3:
                pts_features_proj = pts_features_proj.permute(2, 0, 1).unsqueeze(0)  # (1, C_pts, H, W)
            
            # 调整尺寸匹配（如果需要）
            if img_features.shape[2:] != pts_features_proj.shape[2:]:
                pts_features_proj = F.interpolate(
                    pts_features_proj,
                    size=img_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
        
        return img_features, pts_features_proj
    
    def forward(
        self,
        images: torch.Tensor,
        point_clouds,
        projection_matrix: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：端到端异常检测
        
        Args:
            images: 输入图像 (B, C, H, W)
            point_clouds: 点云数据
            projection_matrix: 投影矩阵 (3, 4) 或 (4, 4)
            return_features: 是否返回中间特征，默认False
            
        Returns:
            output_dict: 包含异常概率图的字典
        """
        # 提取特征
        img_features, pts_features_proj = self.extract_features(
            images, point_clouds, projection_matrix
        )
        
        # 融合和判定
        anomaly_map = self.fusion_head(img_features, pts_features_proj)  # (B, 1, H', W')
        
        output_dict = {
            'anomaly_map': anomaly_map
        }
        
        if return_features:
            output_dict['img_features'] = img_features
            output_dict['pts_features'] = pts_features_proj
        
        return output_dict
    
    def get_trainable_parameters(self):
        """获取可训练参数（仅FusionHead）"""
        return self.fusion_head.parameters()
    
    def count_parameters(self):
        """统计参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }

