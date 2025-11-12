"""
nuScenes数据集加载器
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
import cv2


class NuScenesDataset(Dataset):
    """
    nuScenes数据集加载器
    
    用于阶段二（2D语义分支）和阶段三（3D几何分支）的预训练
    """
    
    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-trainval',
        split: str = 'train',
        image_size: Tuple[int, int] = (1024, 2048),
        load_lidar: bool = True,
        load_semantic_labels: bool = True,
    ):
        """
        Args:
            data_root: nuScenes数据根目录
            version: 数据集版本
            split: 数据集分割（'train' 或 'val'）
            image_size: 图像尺寸 (H, W)
            load_lidar: 是否加载LiDAR点云
            load_semantic_labels: 是否加载语义分割标签
        """
        self.data_root = data_root
        self.version = version
        self.split = split
        self.image_size = image_size
        self.load_lidar = load_lidar
        self.load_semantic_labels = load_semantic_labels
        
        # 初始化nuScenes API
        self.nusc = NuScenes(
            version=version,
            dataroot=data_root,
            verbose=True,
        )
        
        # 获取样本列表
        if split == 'train':
            self.scenes = self.nusc.scene[:int(len(self.nusc.scene) * 0.9)]
        else:
            self.scenes = self.nusc.scene[int(len(self.nusc.scene) * 0.9):]
        
        # 收集所有样本token
        self.sample_tokens = []
        for scene in self.scenes:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                self.sample_tokens.append(sample_token)
                sample_token = sample['next']
    
    def __len__(self) -> int:
        return len(self.sample_tokens)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Returns:
            Dict包含：
            - 'image': 图像 (3, H, W)
            - 'point_cloud': 点云 (N, 3) [可选]
            - 'semantic_label': 语义标签 (H, W) [可选]
            - 'camera_intrinsic': 相机内参 (3, 3)
            - 'camera_extrinsic': 相机外参 (4, 4)
            - 'sample_token': 样本token
        """
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        
        results = {
            'sample_token': sample_token,
        }
        
        # 加载图像（使用前向相机）
        cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        image_path = os.path.join(self.data_root, cam_front_data['filename'])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        results['image'] = image
        
        # 加载相机标定
        cam_front_calib = self.nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(cam_front_calib['camera_intrinsic'])
        results['camera_intrinsic'] = torch.from_numpy(camera_intrinsic).float()
        
        # 获取相机外参（从ego到相机）
        camera_extrinsic = np.eye(4)
        camera_extrinsic[:3, :3] = np.array(cam_front_calib['rotation'])
        camera_extrinsic[:3, 3] = np.array(cam_front_calib['translation'])
        results['camera_extrinsic'] = torch.from_numpy(camera_extrinsic).float()
        
        # 加载LiDAR点云（如果指定）
        if self.load_lidar:
            lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_path = os.path.join(self.data_root, lidar_data['filename'])
            
            # 加载点云
            pc = LidarPointCloud.from_file(lidar_path)
            points = pc.points[:3, :].T  # (N, 3)
            
            # 转换到相机坐标系（简化版本）
            # 实际应用中需要更精确的坐标变换
            results['point_cloud'] = points.astype(np.float32)
        
        # 加载语义分割标签（如果指定）
        if self.load_semantic_labels:
            # nuScenes没有直接的语义分割标签
            # 这里需要根据实际的数据格式加载
            # 可能需要使用nuScenes的panoptic分割或自己标注的数据
            # 暂时返回None，需要根据实际情况实现
            results['semantic_label'] = None
        
        return results


class NuScenesSemanticDataset(NuScenesDataset):
    """
    nuScenes语义分割数据集（用于2D分支训练）
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, load_semantic_labels=True, **kwargs)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        
        # 这里需要根据实际的语义标签格式加载
        # 如果使用nuScenes的panoptic分割，需要转换
        # 暂时返回一个占位符
        if sample.get('semantic_label') is None:
            # 创建一个假的标签（实际应用中需要替换）
            sample['semantic_label'] = torch.zeros(
                (self.image_size[0], self.image_size[1]),
                dtype=torch.long,
            )
        
        return sample


class NuScenesPointCloudDataset(NuScenesDataset):
    """
    nuScenes点云数据集（用于3D分支训练）
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, load_lidar=True, load_semantic_labels=False, **kwargs)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        
        # 确保点云存在
        if 'point_cloud' not in sample:
            # 创建一个空的点云
            sample['point_cloud'] = np.zeros((0, 3), dtype=np.float32)
        
        return sample

