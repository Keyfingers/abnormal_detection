"""
3D几何分支：基于MinkUNet的点云重建和异常检测
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock

# 添加MinkowskiEngine示例路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../MinkowskiEngine-master/examples'))
from minkunet import MinkUNetBase


class MinkUNetAutoEncoder(MinkUNetBase):
    """
    基于MinkUNet的自编码器，用于点云重建
    """
    BLOCK = BasicBlock
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1
    
    def __init__(self, in_channels: int = 3, feature_dim: int = 128, D: int = 3):
        """
        Args:
            in_channels: 输入特征维度（通常是3，表示xyz坐标）
            feature_dim: 输出特征维度（用于融合）
            D: 空间维度（3D点云为3）
        """
        # 修改输出通道数为feature_dim
        super().__init__(in_channels, feature_dim, D)
        self.feature_dim = feature_dim
    
    def forward_with_features(self, x: ME.SparseTensor) -> Tuple[ME.SparseTensor, ME.SparseTensor]:
        """
        前向传播，同时返回重建结果和特征
        
        Returns:
            - reconstruction: 重建的点云特征 (用于计算重建误差)
            - features: 解码器特征 (用于融合)
        """
        # 编码器部分
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        
        # 最深层（编码器输出）
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        encoded = self.block4(out)
        
        # 解码器部分
        # tensor_stride=8
        out = self.convtr4p16s2(encoded)
        out = self.bntr4(out)
        out = self.relu(out)
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)
        
        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        decoder_features = self.block6(out)  # 保存解码器特征
        
        # tensor_stride=2
        out = self.convtr6p4s2(decoder_features)
        out = self.bntr6(out)
        out = self.relu(out)
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)
        
        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = ME.cat(out, out_p1)
        out = self.block8(out)
        
        # 最终重建输出
        reconstruction = self.final(out)
        
        return reconstruction, decoder_features


class Geometric3DBranch(nn.Module):
    """
    3D几何分支模型封装
    
    功能：
    1. 使用MinkUNet进行点云重建
    2. 提取解码器的3D特征（用于融合）
    3. 计算重建误差作为异常评分（用于基线）
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 128,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        voxel_size: float = 0.05,
    ):
        """
        Args:
            in_channels: 输入特征维度
            feature_dim: 输出特征维度
            checkpoint_path: 预训练模型权重路径
            freeze_backbone: 是否冻结主干网络
            voxel_size: 体素大小（用于点云量化）
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim
        
        # 构建MinkUNet自编码器
        self.model = MinkUNetAutoEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            D=3,
        )
        
        # 加载权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        # 冻结主干网络（如果指定）
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.eval()
    
    def forward(
        self,
        point_clouds: List[np.ndarray],
        return_features: bool = True,
        return_reconstruction_error: bool = True,
    ) -> Dict:
        """
        前向传播
        
        Args:
            point_clouds: 点云列表，每个元素是 (N, 3) 的numpy数组
            return_features: 是否返回3D特征
            return_reconstruction_error: 是否返回重建误差
        
        Returns:
            Dict包含：
            - 'features_3d': 3D特征（稀疏张量）[可选]
            - 'reconstruction_error': 重建误差 (B, N) [可选]
            - 'sparse_tensor': 输入的稀疏张量
        """
        # 将点云转换为稀疏张量
        sparse_tensor, coords_list = self._point_clouds_to_sparse_tensor(point_clouds)
        
        # 前向传播
        reconstruction, decoder_features = self.model.forward_with_features(sparse_tensor)
        
        results = {
            'sparse_tensor': sparse_tensor,
            'reconstruction': reconstruction,
        }
        
        # 提取3D特征（用于融合）
        if return_features:
            results['features_3d'] = decoder_features
            results['coords_list'] = coords_list
        
        # 计算重建误差（用于基线）
        if return_reconstruction_error:
            # 重建误差：原始特征与重建特征的L2距离
            # 注意：这里我们假设输入特征是xyz坐标
            original_features = sparse_tensor.F
            reconstructed_features = reconstruction.F
            
            # 由于稀疏张量的坐标可能不完全匹配，我们需要在相同坐标上比较
            # 简化版本：使用chamfer距离或L2损失
            reconstruction_error = self._compute_reconstruction_error(
                sparse_tensor, reconstruction
            )
            results['reconstruction_error'] = reconstruction_error
        
        return results
    
    def _point_clouds_to_sparse_tensor(
        self,
        point_clouds: List[np.ndarray],
    ) -> Tuple[ME.SparseTensor, List]:
        """
        将点云列表转换为Minkowski稀疏张量
        
        Args:
            point_clouds: 点云列表，每个元素是 (N, 3) 的numpy数组
        
        Returns:
            sparse_tensor: Minkowski稀疏张量
            coords_list: 坐标列表（用于后续投影）
        """
        batch_coords = []
        batch_feats = []
        coords_list = []
        
        for batch_idx, pc in enumerate(point_clouds):
            # 量化坐标
            coords, feats, inds = ME.utils.sparse_quantize(
                pc / self.voxel_size,
                features=pc,  # 使用原始坐标作为特征
                return_index=True,
            )
            
            # 添加batch索引
            batch_coord = np.column_stack([np.full(len(coords), batch_idx), coords])
            batch_coords.append(torch.from_numpy(batch_coord).int())
            batch_feats.append(torch.from_numpy(feats).float())
            coords_list.append(coords * self.voxel_size)  # 保存原始坐标
        
        # 合并batch
        coords = torch.cat(batch_coords, dim=0)
        feats = torch.cat(batch_feats, dim=0)
        
        # 创建稀疏张量
        sparse_tensor = ME.SparseTensor(
            features=feats,
            coordinates=coords,
            device=feats.device if hasattr(feats, 'device') else torch.device('cpu'),
        )
        
        return sparse_tensor, coords_list
    
    def _compute_reconstruction_error(
        self,
        original: ME.SparseTensor,
        reconstruction: ME.SparseTensor,
    ) -> torch.Tensor:
        """
        计算重建误差
        
        Args:
            original: 原始稀疏张量
            reconstruction: 重建的稀疏张量
        
        Returns:
            重建误差（每个点的L2距离）
        """
        # 获取原始特征（xyz坐标）
        original_features = original.F  # (N, 3)
        
        # 将重建特征投影回原始坐标
        # 使用插值获取重建特征在原始坐标位置的值
        original_coords = original.C.float()  # (N, 4) [batch, x, y, z]
        
        # 从重建结果中提取特征
        # 简化版本：直接比较特征
        # 实际应用中，需要将重建特征插值回原始坐标
        reconstructed_features = reconstruction.F  # (M, feature_dim)
        
        # 由于坐标可能不完全匹配，我们使用最近邻或插值
        # 这里使用简化的方法：计算重建特征的L2范数作为异常分数
        # 对于正常几何，重建误差应该很小；对于异常几何，重建误差很大
        
        # 使用重建特征的L2范数作为异常分数
        # 如果重建特征接近原始特征，则误差小
        # 这里我们假设重建特征的前3个通道对应xyz坐标
        if reconstructed_features.shape[1] >= 3:
            recon_xyz = reconstructed_features[:, :3]
            # 计算L2距离
            error = torch.norm(original_features - recon_xyz, dim=1)
        else:
            # 如果特征维度不匹配，使用特征本身的L2范数
            error = torch.norm(reconstructed_features, dim=1)
        
        return error
    
    def get_geometric_features(
        self,
        point_clouds: List[np.ndarray],
    ) -> Tuple[ME.SparseTensor, List]:
        """获取3D几何特征（用于融合）"""
        results = self.forward(
            point_clouds,
            return_features=True,
            return_reconstruction_error=False,
        )
        return results['features_3d'], results['coords_list']
    
    def get_reconstruction_error(
        self,
        point_clouds: List[np.ndarray],
    ) -> torch.Tensor:
        """获取重建误差（用于基线）"""
        results = self.forward(
            point_clouds,
            return_features=False,
            return_reconstruction_error=True,
        )
        return results['reconstruction_error']

