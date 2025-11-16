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
from scipy.spatial.distance import cdist

# 添加MinkowskiEngine示例路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../MinkowskiEngine-master/examples'))
from minkunet import MinkUNetBase


class MinkUNetAutoEncoder(MinkUNetBase):
    """
    基于MinkUNet的自编码器，用于点云重建
    
    修复说明：
    - 重建任务需要输出与输入相同的维度（3维xyz），而不是feature_dim
    - 解码器特征用于融合，应提取中间层特征（feature_dim维度）
    - 最终重建输出必须是3维xyz，用于计算重建误差
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
            feature_dim: 解码器特征维度（用于融合），注意：最终重建输出仍然是3维xyz
            D: 空间维度（3D点云为3）
        """
        # 修复：重建输出必须是3维xyz，但解码器中间特征可以是feature_dim
        # 因此需要两个输出：1) 重建的xyz (3维) 2) 解码器特征 (feature_dim维)
        super().__init__(in_channels, feature_dim, D)
        self.feature_dim = feature_dim
        self.in_channels = in_channels
        
        # 修复：添加最终重建层，输出3维xyz坐标
        # 这是必要的，因为重建任务的目标是恢复原始点云坐标
        self.reconstruction_head = ME.MinkowskiConvolution(
            feature_dim, in_channels, kernel_size=1, stride=1, dimension=D
        )
    
    def forward_with_features(self, x: ME.SparseTensor, mask: Optional[ME.SparseTensor] = None) -> Tuple[ME.SparseTensor, ME.SparseTensor]:
        """
        前向传播，同时返回重建结果和特征
        
        修复说明：
        - 添加mask参数支持随机mask训练（30%体素被mask）
        - 重建输出必须是3维xyz，用于计算重建误差
        - 解码器特征（decoder_features）用于融合，保持feature_dim维度
        
        Args:
            x: 输入稀疏张量（可能已被mask）
            mask: 可选的mask张量，指示哪些体素被mask（用于训练）
        
        Returns:
            - reconstruction: 重建的点云xyz坐标 (3维，用于计算重建误差)
            - features: 解码器特征 (feature_dim维，用于融合)
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
        # 修复：提取解码器中间特征用于融合（规则要求：U-Net上采样部分的特征）
        # 这个位置的特征既包含足够的语义信息，又保持了相对较高的空间分辨率
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        decoder_features = self.block6(out)  # 保存解码器特征（feature_dim维，用于融合）
        
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
        
        # 修复：最终重建输出必须是3维xyz坐标
        # 使用reconstruction_head将feature_dim维特征映射回3维xyz
        # 这是必要的，因为重建误差需要比较原始xyz和重建xyz
        reconstruction_xyz = self.reconstruction_head(out)
        
        return reconstruction_xyz, decoder_features


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
        apply_mask: bool = False,
        mask_ratio: float = 0.3,
    ) -> Dict:
        """
        前向传播
        
        修复说明：
        - 添加apply_mask参数支持随机mask训练（规则要求：随机mask掉30%体素）
        - 这是自监督重建任务的核心：模型需要从部分可见的点云重建完整点云
        - 训练时apply_mask=True，评估时apply_mask=False
        
        Args:
            point_clouds: 点云列表，每个元素是 (N, 3) 的numpy数组
            return_features: 是否返回3D特征
            return_reconstruction_error: 是否返回重建误差
            apply_mask: 是否应用随机mask（用于训练）
            mask_ratio: mask比例（默认0.3，即30%）
        
        Returns:
            Dict包含：
            - 'features_3d': 3D特征（稀疏张量）[可选]
            - 'reconstruction_error': 重建误差 (B, N) [可选]
            - 'sparse_tensor': 输入的稀疏张量
            - 'masked_tensor': 被mask后的稀疏张量（如果apply_mask=True）
        """
        # 将点云转换为稀疏张量
        sparse_tensor, coords_list = self._point_clouds_to_sparse_tensor(point_clouds)
        
        # 修复：实现随机mask机制（规则要求：随机mask掉30%体素）
        # 这是自监督重建任务的关键：模型学习从部分可见的点云重建完整点云
        # 对于异常物体（如AnoVox中的奶牛），模型从未在nuScenes中学习过如何重建，
        # 因此重建误差会非常高，可以作为异常检测的依据
        masked_tensor = sparse_tensor
        if apply_mask and self.training:
            masked_tensor = self._apply_random_mask(sparse_tensor, mask_ratio)
        
        # 前向传播（使用masked_tensor作为输入）
        reconstruction, decoder_features = self.model.forward_with_features(masked_tensor)
        
        results = {
            'sparse_tensor': sparse_tensor,  # 原始完整点云
            'reconstruction': reconstruction,  # 重建的点云（3维xyz）
        }
        
        if apply_mask:
            results['masked_tensor'] = masked_tensor
        
        # 提取3D特征（用于融合）
        if return_features:
            results['features_3d'] = decoder_features
            results['coords_list'] = coords_list
        
        # 计算重建误差（用于基线）
        if return_reconstruction_error:
            # 修复：重建误差应该比较原始完整点云和重建点云
            # 即使输入是masked的，重建误差应该基于完整点云计算
            reconstruction_error = self._compute_reconstruction_error(
                sparse_tensor, reconstruction
            )
            results['reconstruction_error'] = reconstruction_error
        
        return results
    
    def _apply_random_mask(
        self,
        sparse_tensor: ME.SparseTensor,
        mask_ratio: float = 0.3,
    ) -> ME.SparseTensor:
        """
        应用随机mask到稀疏张量
        
        修复说明：
        - 这是规则要求的核心训练机制：随机mask掉30%的体素
        - 模型需要学习从剩余70%的体素重建被mask的部分
        - 这种自监督学习使得模型能够学习正常几何的表示
        
        Args:
            sparse_tensor: 输入稀疏张量
            mask_ratio: mask比例（0.3表示30%）
        
        Returns:
            被mask后的稀疏张量（部分体素被移除）
        """
        # 获取所有体素
        num_voxels = len(sparse_tensor.F)
        num_mask = int(num_voxels * mask_ratio)
        
        # 随机选择要mask的体素索引
        mask_indices = torch.randperm(num_voxels, device=sparse_tensor.device)[:num_mask]
        
        # 创建保留索引（未被mask的体素）
        keep_indices = torch.ones(num_voxels, dtype=torch.bool, device=sparse_tensor.device)
        keep_indices[mask_indices] = False
        
        # 提取保留的坐标和特征
        kept_coords = sparse_tensor.C[keep_indices]
        kept_feats = sparse_tensor.F[keep_indices]
        
        # 创建新的稀疏张量（只包含未被mask的体素）
        masked_tensor = ME.SparseTensor(
            features=kept_feats,
            coordinates=kept_coords,
            device=sparse_tensor.device,
        )
        
        return masked_tensor
    
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
        
        # 修复：统一设备管理
        # 确保所有张量在同一设备上
        # 合并batch
        coords = torch.cat(batch_coords, dim=0)
        feats = torch.cat(batch_feats, dim=0)
        
        # 修复：确定设备（优先使用GPU，如果可用）
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # 确保张量在正确的设备上
        coords = coords.to(device)
        feats = feats.to(device)
        
        # 创建稀疏张量
        sparse_tensor = ME.SparseTensor(
            features=feats,
            coordinates=coords,
            device=device,
        )
        
        return sparse_tensor, coords_list
    
    def _compute_reconstruction_error(
        self,
        original: ME.SparseTensor,
        reconstruction: ME.SparseTensor,
    ) -> torch.Tensor:
        """
        计算重建误差
        
        修复说明：
        - 重建输出现在保证是3维xyz，可以直接与原始xyz比较
        - 需要处理稀疏张量坐标不匹配的问题（使用插值或最近邻）
        - 重建误差 = ||原始xyz - 重建xyz||_2
        
        Args:
            original: 原始稀疏张量（包含原始xyz坐标）
            reconstruction: 重建的稀疏张量（包含重建的xyz坐标，3维）
        
        Returns:
            重建误差（每个点的L2距离）
        """
        # 修复：重建特征现在保证是3维xyz
        original_features = original.F  # (N, 3) 原始xyz坐标
        reconstructed_features = reconstruction.F  # (M, 3) 重建xyz坐标
        
        # 修复：由于稀疏张量的坐标可能不完全匹配（量化后可能不同），
        # 需要将重建特征插值回原始坐标位置
        # 使用Minkowski Engine的插值功能
        original_coords = original.C.float()[:, 1:]  # (N, 3) 去除batch索引，只保留xyz
        reconstruction_coords = reconstruction.C.float()[:, 1:]  # (M, 3)
        
        # 如果坐标完全匹配，直接计算L2距离
        if len(original_coords) == len(reconstruction_coords):
            # 检查坐标是否匹配（允许小的数值误差）
            coords_match = torch.allclose(original_coords, reconstruction_coords, atol=1e-3)
            if coords_match:
                # 直接计算L2距离
                error = torch.norm(original_features - reconstructed_features, dim=1)
                return error
        
        # 修复：坐标不匹配时，使用最近邻插值
        # 对于每个原始点，找到最近的重建点
        
        original_coords_np = original_coords.detach().cpu().numpy()
        reconstruction_coords_np = reconstruction_coords.detach().cpu().numpy()
        reconstructed_features_np = reconstructed_features.detach().cpu().numpy()
        
        # 计算距离矩阵
        distances = cdist(original_coords_np, reconstruction_coords_np)
        nearest_indices = np.argmin(distances, axis=1)
        
        # 获取最近邻的重建特征
        nearest_reconstructed = torch.from_numpy(
            reconstructed_features_np[nearest_indices]
        ).to(original_features.device)
        
        # 计算L2距离
        error = torch.norm(original_features - nearest_reconstructed, dim=1)
        
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

