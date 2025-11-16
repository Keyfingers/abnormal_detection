"""
AnoVox数据集加载器
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2


class AnoVoxDataset(Dataset):
    """
    AnoVox数据集加载器
    
    用于阶段四（融合模块）的训练和评估
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (1024, 2048),
    ):
        """
        Args:
            data_root: AnoVox数据根目录
            split: 数据集分割（'train' 或 'test'）
            image_size: 图像尺寸 (H, W)
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # 构建数据路径
        self.split_dir = os.path.join(data_root, split)
        self.image_dir = os.path.join(self.split_dir, 'images')
        self.pointcloud_dir = os.path.join(self.split_dir, 'pointclouds')
        self.label_dir = os.path.join(self.split_dir, 'anomaly_masks')
        
        # 获取所有样本ID
        if os.path.exists(self.image_dir):
            self.sample_ids = [
                f.replace('.jpg', '').replace('.png', '')
                for f in os.listdir(self.image_dir)
                if f.endswith(('.jpg', '.png'))
            ]
        else:
            self.sample_ids = []
        
        self.sample_ids.sort()
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
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
        sample_id = self.sample_ids[idx]
        
        results = {
            'sample_id': sample_id,
        }
        
        # 加载图像
        image_path = os.path.join(self.image_dir, f'{sample_id}.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f'{sample_id}.png')
        
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            image = torch.from_numpy(image)
        else:
            # 创建占位符
            image = torch.zeros((3, self.image_size[0], self.image_size[1]))
        
        results['image'] = image
        
        # 加载点云
        pc_path = os.path.join(self.pointcloud_dir, f'{sample_id}.bin')
        if os.path.exists(pc_path):
            # 假设点云是.bin格式（numpy数组）
            point_cloud = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        else:
            # 创建占位符
            point_cloud = np.zeros((0, 3), dtype=np.float32)
        
        results['point_cloud'] = point_cloud.astype(np.float32)
        
        # 加载异常掩码
        mask_path = os.path.join(self.label_dir, f'{sample_id}.png')
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))
            mask = (mask > 128).astype(np.float32)  # 二值化
            mask = torch.from_numpy(mask)
        else:
            # 创建占位符
            mask = torch.zeros((self.image_size[0], self.image_size[1]))
        
        results['anomaly_mask'] = mask
        
        # 修复：实现正确的相机标定加载（规则要求：确保数据集中包含正确的内参和外参）
        # 这是3D-2D投影的命脉，必须准确
        calib_path = os.path.join(self.split_dir, 'calibrations', f'{sample_id}.txt')
        if os.path.exists(calib_path):
            # 修复：根据AnoVox数据集的实际格式解析相机标定
            # 假设格式为：
            # 内参矩阵（3x3）：
            # fx 0 cx
            # 0 fy cy
            # 0 0 1
            # 外参矩阵（4x4）：
            # R11 R12 R13 tx
            # R21 R22 R23 ty
            # R31 R32 R33 tz
            # 0   0   0   1
            try:
                with open(calib_path, 'r') as f:
                    lines = f.readlines()
                
                # 解析内参（前3行）
                camera_intrinsic = np.array([
                    [float(x) for x in lines[0].split()],
                    [float(x) for x in lines[1].split()],
                    [float(x) for x in lines[2].split()],
                ])
                
                # 解析外参（后4行）
                camera_extrinsic = np.array([
                    [float(x) for x in lines[3].split()],
                    [float(x) for x in lines[4].split()],
                    [float(x) for x in lines[5].split()],
                    [float(x) for x in lines[6].split()],
                ])
            except Exception as e:
                print(f"Warning: Failed to parse calibration file {calib_path}: {e}")
                # 使用默认值
                camera_intrinsic = np.array([
                    [1000, 0, self.image_size[1] / 2],
                    [0, 1000, self.image_size[0] / 2],
                    [0, 0, 1],
                ])
                camera_extrinsic = np.eye(4)
        else:
            # 修复：如果标定文件不存在，尝试从其他位置加载
            # 例如：可能存储在JSON或YAML文件中
            calib_json_path = os.path.join(self.split_dir, 'calibrations', f'{sample_id}.json')
            if os.path.exists(calib_json_path):
                import json
                try:
                    with open(calib_json_path, 'r') as f:
                        calib_data = json.load(f)
                    camera_intrinsic = np.array(calib_data.get('intrinsic', np.eye(3)))
                    camera_extrinsic = np.array(calib_data.get('extrinsic', np.eye(4)))
                except Exception as e:
                    print(f"Warning: Failed to parse JSON calibration file {calib_json_path}: {e}")
                    camera_intrinsic = np.array([
                        [1000, 0, self.image_size[1] / 2],
                        [0, 1000, self.image_size[0] / 2],
                        [0, 0, 1],
                    ])
                    camera_extrinsic = np.eye(4)
            else:
                # 修复：如果完全没有标定文件，使用合理的默认值
                # 但应该警告用户，因为标定参数对3D-2D投影至关重要
                print(f"Warning: No calibration file found for {sample_id}, using default values")
                camera_intrinsic = np.array([
                    [1000, 0, self.image_size[1] / 2],
                    [0, 1000, self.image_size[0] / 2],
                    [0, 0, 1],
                ])
                camera_extrinsic = np.eye(4)
        
        results['camera_intrinsic'] = torch.from_numpy(camera_intrinsic).float()
        results['camera_extrinsic'] = torch.from_numpy(camera_extrinsic).float()
        
        return results

