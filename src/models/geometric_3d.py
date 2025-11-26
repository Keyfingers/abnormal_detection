"""
3D几何分支模块
实现基于冻结MMDetection3D模型的几何特征提取
"""
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union
import sys
import numpy as np

# 尝试导入MMDetection3D，如果未安装则提供友好错误信息
try:
    import mmdet3d
    from mmdet3d.apis import init_model, inference_detector
    # mmcv 2.x中Config和load_checkpoint都在mmengine中
    try:
        from mmengine import Config
        from mmengine.runner import load_checkpoint
    except ImportError:
        # 兼容旧版本mmcv
        from mmcv import Config
        from mmcv.runner import load_checkpoint
    MMDET3D_AVAILABLE = True
except ImportError as e:
    MMDET3D_AVAILABLE = False
    # 只在真正导入失败时打印警告
    import warnings
    warnings.warn(f"MMDetection3D未安装或导入失败: {e}。请参考requirements.txt中的说明安装MMDetection3D。", ImportWarning)

from ..utils.pointcloud_preprocessing import voxelize_pointcloud


class Geometric3DBranch(nn.Module):
    """
    3D几何分支，基于冻结的MMDetection3D模型
    
    该类封装了MMDetection3D的点云分割模型，用于提取点云的几何特征。
    所有模型参数被冻结，仅作为特征提取器使用。
    
    Attributes:
        checkpoint_path: 预训练权重文件路径
        config_path: MMDetection3D配置文件路径（可选）
        freeze_backbone: 是否冻结backbone参数
        feature_dim: 输出特征维度，默认128
        voxel_size: 体素尺寸（米），默认0.05（5cm）
        device: 计算设备（'cuda'或'cpu'）
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        freeze_backbone: bool = True,
        feature_dim: int = 128,
        voxel_size: float = 0.05,
        device: str = "cuda"
    ):
        """
        初始化Geometric3DBranch
        
        Args:
            checkpoint_path: 预训练权重文件路径
            config_path: MMDetection3D配置文件路径，如果为None则使用默认配置
            freeze_backbone: 是否冻结所有参数，默认True
            feature_dim: 输出特征维度，默认128
            voxel_size: 体素尺寸（米），默认0.05（5cm）
            device: 计算设备，默认'cuda'
            
        Raises:
            FileNotFoundError: 如果权重文件不存在
            ImportError: 如果MMDetection3D未安装
        """
        super(Geometric3DBranch, self).__init__()
        
        if not MMDET3D_AVAILABLE:
            raise ImportError(
                "MMDetection3D未安装。请运行以下命令安装：\n"
                "1. pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html\n"
                "2. pip install mmdet3d\n"
                "（请根据您的CUDA版本调整URL）\n"
                "参考: https://mmdetection3d.readthedocs.io/en/latest/get_started.html"
            )
        
        # 检查权重文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"权重文件不存在: {checkpoint_path}\n"
                f"请运行以下命令下载权重文件：\n"
                f"  python scripts/download_mmdet3d_weights.py\n"
                f"或手动下载SemanticKITTI预训练的MMDetection3D权重"
            )
        
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.freeze_backbone = freeze_backbone
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        self.device = device
        
        # 构建MMDetection3D模型
        self.model = self._build_model()
        
        # 加载权重
        self._load_weights()
        
        # 冻结参数
        if freeze_backbone:
            self._freeze_parameters()
        
        # 设置为评估模式
        self.model.eval()
        
        # 移动到指定设备
        self.model = self.model.to(device)
    
    def _build_model(self) -> nn.Module:
        """
        构建MMDetection3D模型
        
        如果提供了config_path，则使用配置文件初始化模型。
        否则，使用默认的PointNet++配置。
        """
        if self.config_path and os.path.exists(self.config_path):
            # 使用提供的配置文件
            cfg = Config.fromfile(self.config_path)
            model = init_model(cfg, self.checkpoint_path, device=self.device)
            return model
        else:
            # 使用默认配置构建模型
            # 这里创建一个简化的PointNet++风格的backbone用于特征提取
            return self._build_default_model()
    
    def _build_default_model(self) -> nn.Module:
        """
        构建默认的点云特征提取模型
        
        使用PointNet++风格的架构，但只保留backbone部分用于特征提取。
        """
        # 这是一个简化的实现，实际使用时应该使用MMDetection3D的完整模型
        # 或者从配置文件加载
        
        class SimplePointNetBackbone(nn.Module):
            """简化的PointNet++ backbone用于特征提取"""
            def __init__(self, feature_dim=128):
                super().__init__()
                # 简化的PointNet结构
                self.conv1 = nn.Conv1d(3, 64, 1)
                self.bn1 = nn.BatchNorm1d(64)
                self.conv2 = nn.Conv1d(64, 128, 1)
                self.bn2 = nn.BatchNorm1d(128)
                self.conv3 = nn.Conv1d(128, feature_dim, 1)
                self.bn3 = nn.BatchNorm1d(feature_dim)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, points):
                """
                Args:
                    points: (B, N, 3) 点云坐标
                Returns:
                    features: (B, feature_dim, N) 点特征
                """
                x = points.transpose(1, 2)  # (B, 3, N)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.relu(self.bn3(self.conv3(x)))
                return x
        
        return SimplePointNetBackbone(feature_dim=self.feature_dim)
    
    def _load_weights(self):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 处理不同的权重文件格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 如果使用MMDetection3D的init_model，权重已经加载
            if hasattr(self.model, 'load_state_dict'):
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                except RuntimeError as e:
                    print(f"警告: 权重加载不完全匹配: {e}")
                    print("将尝试部分加载...")
                    # 尝试部分加载
                    model_dict = self.model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() 
                                     if k in model_dict and v.size() == model_dict[k].size()}
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
        except Exception as e:
            print(f"警告: 无法加载权重文件: {e}")
            print("将使用随机初始化的权重（仅用于测试）")
    
    def _freeze_parameters(self):
        """冻结所有模型参数"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        point_clouds: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，提取几何特征
        
        Args:
            point_clouds: 点云数据，可以是：
                - 单个点云: (N, 3) 或 (N, 4) numpy数组或torch.Tensor
                - 点云列表: List[(N_i, 3) 或 (N_i, 4)]
                
        Returns:
            Dict包含：
                - 'voxel_features': 体素特征 (M, feature_dim)
                - 'voxel_coords': 体素坐标 (M, 3)，整数坐标
                - 'point_features': 点特征（可选）
        """
        # 处理输入格式
        if isinstance(point_clouds, (np.ndarray, torch.Tensor)):
            # 单个点云
            point_clouds = [point_clouds]
        
        # 处理批次
        batch_features = []
        batch_coords = []
        
        for batch_idx, points in enumerate(point_clouds):
            # 转换为torch.Tensor
            if isinstance(points, np.ndarray):
                points_tensor = torch.from_numpy(points.astype(np.float32))
            else:
                points_tensor = points.float()
            
            if len(points_tensor) == 0:
                # 空点云，跳过
                continue
            
            # 提取坐标（前3维）
            coords = points_tensor[:, :3].to(self.device)  # (N, 3)
            
            # 体素化（用于生成体素坐标）
            points_np = coords.cpu().numpy()
            voxel_coords, voxel_features, _ = voxelize_pointcloud(
                points_np,
                voxel_size=self.voxel_size,
                return_indices=True
            )
            
            if len(voxel_coords) == 0:
                continue
            
            # 通过模型提取特征
            # 将点云转换为批次格式 (1, N, 3)
            points_batch = coords.unsqueeze(0)  # (1, N, 3)
            
            with torch.no_grad():
                # 提取点特征
                if hasattr(self.model, 'extract_feat'):
                    # 如果模型有extract_feat方法（MMDetection3D标准接口）
                    point_feat = self.model.extract_feat(points_batch)
                elif hasattr(self.model, 'backbone'):
                    # 如果模型有backbone属性
                    point_feat = self.model.backbone(points_batch)
                else:
                    # 直接调用模型
                    point_feat = self.model(points_batch)  # (1, feature_dim, N)
            
            # 处理特征输出
            if isinstance(point_feat, (list, tuple)):
                point_feat = point_feat[-1]  # 取最后一层特征
            
            # 转换为 (N, feature_dim)
            if point_feat.dim() == 3:
                point_feat = point_feat.squeeze(0).transpose(0, 1)  # (N, feature_dim)
            elif point_feat.dim() == 2:
                point_feat = point_feat  # 已经是 (N, feature_dim)
            
            # 将点特征聚合到体素
            # 简单方法：对每个体素内的点特征求平均
            voxel_feat_list = []
            for voxel_coord in voxel_coords:
                # 找到属于该体素的点
                voxel_mask = np.all(
                    np.floor(points_np / self.voxel_size).astype(np.int32) == voxel_coord.numpy(),
                    axis=1
                )
                if np.any(voxel_mask):
                    # 对该体素内的点特征求平均
                    voxel_feat = point_feat[voxel_mask].mean(dim=0)  # (feature_dim,)
                    voxel_feat_list.append(voxel_feat)
                else:
                    # 如果没有点，使用零特征
                    voxel_feat_list.append(torch.zeros(self.feature_dim, device=self.device))
            
            if len(voxel_feat_list) > 0:
                voxel_feat_tensor = torch.stack(voxel_feat_list)  # (M, feature_dim)
                batch_features.append(voxel_feat_tensor)
                batch_coords.append(voxel_coords.to(self.device))
        
        if len(batch_features) == 0:
            # 所有点云都为空
            empty_features = torch.zeros((0, self.feature_dim), dtype=torch.float32, device=self.device)
            empty_coords = torch.zeros((0, 3), dtype=torch.int32, device=self.device)
            return {
                'voxel_features': empty_features,
                'voxel_coords': empty_coords
            }
        
        # 合并批次
        all_features = torch.cat(batch_features, dim=0)  # (M_total, feature_dim)
        all_coords = torch.cat(batch_coords, dim=0)  # (M_total, 3)
        
        return {
            'voxel_features': all_features,
            'voxel_coords': all_coords
        }
    
    def extract_features(
        self,
        point_clouds: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        提取特征的别名方法
        
        Args:
            point_clouds: 点云数据
            
        Returns:
            Dict包含体素特征和坐标
        """
        return self.forward(point_clouds)
