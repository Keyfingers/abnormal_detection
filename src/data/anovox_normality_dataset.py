"""
AnoVox Normality数据集加载器（适配实际格式）

适配AnoVox_Normality_Mono_Town03数据集的格式：
- 点云格式：.npy文件（Nx4，包含xyz和intensity）
- 目录结构：按Scenario组织
- 训练数据：所有样本都是"常态"（异常掩码全0）
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2
import glob
import random


class AnoVoxNormalityDataset(Dataset):
    """
    AnoVox Normality数据集加载器
    
    用于阶段四（融合模块）的训练
    通过CutPaste策略生成伪异常（Pseudo-Anomaly）
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (1024, 2048),
        use_placeholder_image: bool = True,
    ):
        """
        Args:
            data_root: AnoVox数据根目录（包含AnoVox_Normality_Mono_Town03目录）
            split: 数据集分割（'train' 或 'test'），用于划分数据
            image_size: 图像尺寸 (H, W)
            use_placeholder_image: 如果True，创建占位符图像（因为数据集可能没有图像）
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.use_placeholder_image = use_placeholder_image
        
        # 查找所有.npy点云文件
        normality_dir = os.path.join(data_root, 'AnoVox_Normality_Mono_Town03')
        if not os.path.exists(normality_dir):
            # 尝试直接在data_root查找
            normality_dir = data_root
        
        # 查找所有点云文件
        pointcloud_files = glob.glob(os.path.join(normality_dir, '**', '*.npy'), recursive=True)
        pointcloud_files = [f for f in pointcloud_files if 'LIDAR' in f]
        
        if len(pointcloud_files) == 0:
            raise ValueError(f"未找到点云文件在: {normality_dir}")
        
        # 按文件名排序，确保可重复性
        pointcloud_files.sort()
        
        # 划分训练/测试集（80/20）
        split_idx = int(len(pointcloud_files) * 0.8)
        if split == 'train':
            self.pointcloud_files = pointcloud_files[:split_idx]
        else:
            self.pointcloud_files = pointcloud_files[split_idx:]
        
        print(f"AnoVox Normality数据集 ({split}): {len(self.pointcloud_files)} 个样本")
    
    def __len__(self) -> int:
        return len(self.pointcloud_files)
    
    def _apply_cutpaste(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 CutPaste 数据增强来模拟异常
        
        原理：将图像的一部分剪切并粘贴到另一位置，制造视觉与几何的不一致性
        """
        h, w = image.shape[1], image.shape[2]
        
        # 随机选择 patch 大小 (图像面积的 2% 到 15%)
        area_ratio = random.uniform(0.02, 0.15)
        aspect_ratio = random.uniform(0.3, 3.3)
        
        patch_area = h * w * area_ratio
        patch_h = int((patch_area * aspect_ratio) ** 0.5)
        patch_w = int((patch_area / aspect_ratio) ** 0.5)
        
        # 边界检查
        patch_h = min(max(patch_h, 10), h - 1)
        patch_w = min(max(patch_w, 10), w - 1)
        
        # 源位置
        src_y = random.randint(0, h - patch_h)
        src_x = random.randint(0, w - patch_w)
        
        # 目标位置
        dst_y = random.randint(0, h - patch_h)
        dst_x = random.randint(0, w - patch_w)
        
        # 提取 patch
        patch = image[:, src_y:src_y+patch_h, src_x:src_x+patch_w].clone()
        
        # 可选：对 patch 进行颜色增强，使其更像"异物"
        if random.random() < 0.5:
            patch = patch * random.uniform(0.8, 1.2)
            
        # 粘贴
        image[:, dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = patch
        
        # 更新 mask (粘贴区域标记为异常)
        mask[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = 1.0
        
        return image, mask

    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Returns:
            Dict包含：
            - 'image': 图像 (3, H, W)
            - 'point_cloud': 点云 (N, 3)
            - 'anomaly_mask': 异常掩码 (H, W)
            - 'camera_intrinsic': 相机内参 (3, 3)
            - 'camera_extrinsic': 相机外参 (4, 4)
            - 'sample_id': 样本ID
        """
        pc_path = self.pointcloud_files[idx]
        sample_id = os.path.splitext(os.path.basename(pc_path))[0]
        
        results = {
            'sample_id': sample_id,
        }
        
        # 加载点云（.npy格式）
        try:
            point_cloud = np.load(pc_path)  # (N, 4)
            # 只使用前3列（xyz），忽略intensity
            point_cloud = point_cloud[:, :3].astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load point cloud {pc_path}: {e}")
            point_cloud = np.zeros((0, 3), dtype=np.float32)
        
        results['point_cloud'] = point_cloud
        
        # 创建占位符图像（因为数据集可能没有图像）
        if self.use_placeholder_image:
            # 注意：如果是占位符图像（全黑），CutPaste效果可能不明显
            # 但为了流程完整性，我们仍然执行。
            # 理想情况下应该加载真实图像。
            image = torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            # 如果是全黑图像，加点噪声以便CutPaste有东西可切
            image = torch.randn_like(image) * 0.1
        else:
            # 尝试查找图像文件（如果存在）
            image_path = pc_path.replace('.npy', '.jpg').replace('LIDAR', 'CAMERA')
            if not os.path.exists(image_path):
                image_path = pc_path.replace('.npy', '.png').replace('LIDAR', 'CAMERA')
            
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
                image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
                image = torch.from_numpy(image)
            else:
                image = torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)
                image = torch.randn_like(image) * 0.1
        
        # 异常掩码初始化
        anomaly_mask = torch.zeros((self.image_size[0], self.image_size[1]), dtype=torch.float32)
        
        # 修复：引入伪异常生成 (CutPaste)
        # 50% 概率生成伪异常，仅在训练时
        if self.split == 'train' and random.random() < 0.5:
            image, anomaly_mask = self._apply_cutpaste(image, anomaly_mask)
            
        results['image'] = image
        results['anomaly_mask'] = anomaly_mask
        
        # 相机标定：使用默认值
        # 注意：AnoVox数据集可能包含标定信息，但当前数据集没有
        # 在实际应用中，需要从数据集元数据或配置文件获取
        camera_intrinsic = np.array([
            [1000, 0, self.image_size[1] / 2],
            [0, 1000, self.image_size[0] / 2],
            [0, 0, 1],
        ], dtype=np.float32)
        
        # 外参：单位矩阵（假设相机坐标系和世界坐标系一致）
        camera_extrinsic = np.eye(4, dtype=np.float32)
        
        results['camera_intrinsic'] = torch.from_numpy(camera_intrinsic).float()
        results['camera_extrinsic'] = torch.from_numpy(camera_extrinsic).float()
        
        return results






