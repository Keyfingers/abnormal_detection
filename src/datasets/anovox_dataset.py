"""
AnoVox数据集加载器
针对AnoVox_Normality_Mono_Town03目录结构

功能:
1. 读取RGB图像 (.png)
2. 读取LiDAR点云 (.npy)
3. 解析sensor_setup.json获取相机内参
4. 从文件夹名解析传感器外参
5. 生成投影矩阵 (用于Feature Splatting)
"""
import os
import json
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.camera_calibration import build_projection_matrix


def parse_sensor_params_from_dirname(dirname: str) -> Dict[str, float]:
    """
    从文件夹名解析传感器参数
    
    示例: RGB-CAM(0, 0, 1.8)(0, 0, 0)
    第一个括号: 位置 (x, y, z)
    第二个括号: 旋转 (roll, pitch, yaw)
    
    Args:
        dirname: 文件夹名
        
    Returns:
        params: 包含位置和旋转的字典
    """
    # 匹配模式: (x, y, z)(r, p, y)
    pattern = r'\(([^)]+)\)\(([^)]+)\)'
    matches = re.findall(pattern, dirname)
    
    if len(matches) >= 2:
        # 解析位置
        pos_str = matches[0]
        rot_str = matches[1]
        
        pos = [float(x.strip()) for x in pos_str.split(',')]
        rot = [float(x.strip()) for x in rot_str.split(',')]
        
        return {
            'position': pos,  # [x, y, z]
            'rotation': rot   # [roll, pitch, yaw]
        }
    else:
        # 默认值（假设传感器在原点，无旋转）
        return {
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0]
        }


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    将欧拉角转换为旋转矩阵 (ZYX顺序)
    
    Args:
        roll: 绕X轴旋转（弧度）
        pitch: 绕Y轴旋转（弧度）
        yaw: 绕Z轴旋转（弧度）
        
    Returns:
        R: 旋转矩阵 (3, 3)
    """
    # 转换为弧度（如果输入是度数）
    if abs(roll) > 2 * np.pi or abs(pitch) > 2 * np.pi or abs(yaw) > 2 * np.pi:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    # ZYX顺序的旋转矩阵
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    R = np.array([
        [cos_y * cos_p, -sin_y * cos_r + cos_y * sin_p * sin_r, sin_y * sin_r + cos_y * sin_p * cos_r],
        [sin_y * cos_p, cos_y * cos_r + sin_y * sin_p * sin_r, -cos_y * sin_r + sin_y * sin_p * cos_r],
        [-sin_p, cos_p * sin_r, cos_p * cos_r]
    ])
    
    return R


def build_extrinsic_matrix(position: List[float], rotation: List[float]) -> np.ndarray:
    """
    构建外参矩阵（LiDAR到相机的变换）
    
    Args:
        position: 位置 [x, y, z]
        rotation: 旋转 [roll, pitch, yaw]（度数或弧度）
        
    Returns:
        extrinsic: 外参矩阵 (4, 4)
    """
    # 转换为旋转矩阵
    R = euler_to_rotation_matrix(rotation[0], rotation[1], rotation[2])
    
    # 构建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    
    return T


def parse_sensor_setup_json(json_path: Path) -> Dict:
    """
    解析sensor_setup.json文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        config: 传感器配置字典
    """
    if not json_path.exists():
        return {}
    
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"警告: 无法解析 {json_path}: {e}")
        return {}


def extract_intrinsic_from_config(config: Dict, image_width: int, image_height: int, rgb_sensor_key: Optional[str] = None) -> np.ndarray:
    """
    从配置中提取相机内参矩阵
    
    Args:
        config: 传感器配置字典（sensor_setup.json的内容）
        image_width: 图像宽度（如果从config中找不到，使用此值）
        image_height: 图像高度（如果从config中找不到，使用此值）
        rgb_sensor_key: RGB传感器在config中的key（可选）
        
    Returns:
        K: 内参矩阵 (3, 3)
    """
    # AnoVox的sensor_setup.json格式：
    # {
    #   "RGB-CAM(...)": {
    #     "args": {
    #       "image_height": 512,
    #       "image_width": 768,
    #       "camera_fov": 90.0
    #     }
    #   }
    # }
    
    # 尝试从config中找到RGB-CAM的配置
    rgb_config = None
    if rgb_sensor_key and rgb_sensor_key in config:
        rgb_config = config[rgb_sensor_key]
    else:
        # 查找第一个RGB-CAM配置
        for key, value in config.items():
            if isinstance(value, dict) and value.get('sensor_type') == 'RGB-CAM':
                rgb_config = value
                break
    
    if rgb_config and 'args' in rgb_config:
        args = rgb_config['args']
        # 从args中提取参数
        img_h = args.get('image_height', image_height)
        img_w = args.get('image_width', image_width)
        fov = args.get('camera_fov', 90.0)  # 度
        
        # 计算焦距：f = (w/2) / tan(FOV/2)
        f = (img_w / 2.0) / np.tan(np.radians(fov / 2.0))
        
        # 内参矩阵
        K = np.array([
            [f, 0, img_w / 2.0],
            [0, f, img_h / 2.0],
            [0, 0, 1]
        ])
        
        return K
    
    # 如果找不到配置，使用默认值
    # 默认内参（CARLA默认90度FOV）
    fov = 90.0  # 度
    f = (image_width / 2.0) / np.tan(np.radians(fov / 2.0))
    K = np.array([
        [f, 0, image_width / 2.0],
        [0, f, image_height / 2.0],
        [0, 0, 1]
    ])
    
    return K


class AnoVoxDataset(Dataset):
    """
    AnoVox数据集加载器 (针对Normality_Mono_Town03结构)
    
    目录结构:
    AnoVox_Normality_Mono_Town03/
    ├── Scenario_000/
    │   ├── RGB-CAM(0, 0, 1.8)(0, 0, 0)/
    │   │   └── *.png
    │   ├── LIDAR(0, 0, 1.8)(0, 0, 0)/
    │   │   └── *.npy
    │   └── sensor_setup.json
    └── ...
    """
    
    def __init__(
        self, 
        root_dir: str, 
        transform=None,
        voxel_size: float = 0.05,
        max_points: int = 100000
    ):
        """
        Args:
            root_dir: 数据集根目录 (例如 "path/to/AnoVox_Normality_Mono_Town03")
            transform: 图像预处理 (Resize, Normalize 等)
            voxel_size: 体素大小 (用于下采样点云，MinkUNet会自己处理)
            max_points: 最大点数 (用于限制显存)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.voxel_size = voxel_size
        self.max_points = max_points
        
        if not self.root_dir.exists():
            raise ValueError(f"数据集根目录不存在: {root_dir}")
        
        # 1. 扫描所有场景
        self.scenarios = sorted([
            d for d in self.root_dir.iterdir() 
            if d.is_dir() and d.name.startswith("Scenario")
        ])
        
        if len(self.scenarios) == 0:
            raise ValueError(f"在 {root_dir} 中未找到任何Scenario目录")
        
        self.samples = []
        self.sensor_configs = {}  # 缓存每个场景的传感器配置
        
        print(f"扫描到 {len(self.scenarios)} 个场景...")
        
        # 2. 建立索引 (Pairing RGB and LiDAR)
        for scenario in self.scenarios:
            # 找到RGB和LiDAR目录
            rgb_dirs = list(scenario.glob("RGB-CAM*"))
            lidar_dirs = list(scenario.glob("LIDAR*"))
            
            if not rgb_dirs or not lidar_dirs:
                print(f"跳过场景 {scenario.name}: 缺少传感器目录")
                continue
            
            rgb_dir = rgb_dirs[0]
            lidar_dir = lidar_dirs[0]
            
            # 读取sensor_setup.json
            setup_file = scenario / "sensor_setup.json"
            if setup_file.exists():
                self.sensor_configs[scenario.name] = parse_sensor_setup_json(setup_file)
            
            # 解析传感器参数（从文件夹名）
            rgb_params = parse_sensor_params_from_dirname(rgb_dir.name)
            lidar_params = parse_sensor_params_from_dirname(lidar_dir.name)
            
            # 匹配文件
            # 文件名格式：传感器名_帧ID.扩展名
            # 例如：RGB-CAM(0, 0, 1.8)(0, 0, 0)_1932154471836802243_4884.png
            rgb_files = sorted(list(rgb_dir.glob("*.png")))
            
            for rgb_path in rgb_files:
                # 从文件名中提取帧ID（最后一个下划线后的数字）
                rgb_stem = rgb_path.stem  # 不带扩展名的文件名
                # 提取帧ID：假设格式是 传感器名_帧ID 或 传感器名_其他_帧ID
                # 使用最后一个下划线后的部分作为帧ID
                parts = rgb_stem.rsplit('_', 1)
                if len(parts) == 2:
                    frame_id = parts[1]  # 帧ID
                    # 构建LIDAR文件名：LIDAR传感器名_帧ID.npy
                    # 需要从LIDAR目录中找到匹配的文件
                    lidar_pattern = f"*_{frame_id}.npy"
                    lidar_matches = list(lidar_dir.glob(lidar_pattern))
                    
                    if lidar_matches:
                        lidar_path = lidar_matches[0]
                        self.samples.append({
                            "scenario": scenario.name,
                            "rgb_path": str(rgb_path),
                            "lidar_path": str(lidar_path),
                            "frame_id": frame_id,
                            "rgb_params": rgb_params,
                            "lidar_params": lidar_params
                        })
                    else:
                        # 添加警告信息，方便调试数据完整性
                        print(f"警告: 未找到匹配的LiDAR文件 for RGB: {rgb_stem}, 帧ID: {frame_id}")
                else:
                    # 如果文件名格式不符合预期，尝试直接匹配
                    # 假设文件名就是帧ID（向后兼容）
                    frame_id = rgb_stem
                    lidar_path = lidar_dir / f"{frame_id}.npy"
                    if lidar_path.exists():
                        self.samples.append({
                            "scenario": scenario.name,
                            "rgb_path": str(rgb_path),
                            "lidar_path": str(lidar_path),
                            "frame_id": frame_id,
                            "rgb_params": rgb_params,
                            "lidar_params": lidar_params
                        })
                    else:
                        print(f"警告: 未找到匹配的LiDAR文件 for RGB: {rgb_stem}, 尝试帧ID: {frame_id}")
        
        if len(self.samples) == 0:
            raise ValueError(f"未找到任何匹配的数据对（RGB + LiDAR）")
        
        print(f"共加载 {len(self.samples)} 帧数据对。")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # ---------------------------------------------------
        # 1. 读取RGB图像
        # ---------------------------------------------------
        img = Image.open(sample['rgb_path']).convert('RGB')
        orig_w, orig_h = img.size
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # 默认转Tensor
            import torchvision.transforms as T
            img_tensor = T.ToTensor()(img)
        
        # ---------------------------------------------------
        # 2. 读取LiDAR点云 (.npy)
        # ---------------------------------------------------
        # AnoVox .npy通常是 (N, 4) -> x, y, z, intensity
        # 或者 (N, 3) -> x, y, z
        points = np.load(sample['lidar_path'])
        
        # 确保是2D数组
        if points.ndim == 1:
            points = points.reshape(-1, points.shape[0])
        
        # 限制点数（可选，为了节省显存）
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        
        # 处理点云通道数
        # MinkUNet预训练权重（SemanticKITTI）通常期望4维 (x, y, z, intensity)
        if points.shape[1] == 3:
            # 只有x, y, z，需要补全intensity通道
            intensity = np.ones((points.shape[0], 1), dtype=points.dtype)
            points = np.hstack((points, intensity))
        elif points.shape[1] > 4:
            # 超过4维，只保留前4维
            points = points[:, :4]
        # 如果已经是4维，保持不变
        
        # 转换为Tensor
        points_tensor = torch.from_numpy(points).float()
        
        # ---------------------------------------------------
        # 3. 计算投影矩阵 (Projection Matrix)
        # ---------------------------------------------------
        # 获取当前场景的配置
        config = self.sensor_configs.get(sample['scenario'], {})
        
        # 从RGB路径中提取传感器key（用于查找config中的RGB配置）
        rgb_path_obj = Path(sample['rgb_path'])
        rgb_sensor_key = rgb_path_obj.parent.name  # RGB目录名
        
        # 提取内参
        K = extract_intrinsic_from_config(config, orig_w, orig_h, rgb_sensor_key)
        
        # 计算外参（LiDAR到相机的变换）
        # 关键：即使物理位置重合，LiDAR和相机的坐标系定义通常不同！
        # LiDAR (CARLA/KITTI): X-前, Y-右, Z-上 (Right-handed)
        # Camera (OpenCV/Mask2Former): X-右, Y-下, Z-前
        lidar_pos = np.array(sample['lidar_params']['position'])
        rgb_pos = np.array(sample['rgb_params']['position'])
        
        # 计算相对位置
        relative_pos = rgb_pos - lidar_pos
        
        # 标准变换: LiDAR(前,右,上) -> Camera(右,下,前)
        # x_cam = y_lidar
        # y_cam = -z_lidar
        # z_cam = x_lidar
        # 这是CARLA/AnoVox的标准坐标轴转换
        rotation_lidar2cam = np.array([
            [0, 1, 0],   # x_cam = y_lidar
            [0, 0, -1],  # y_cam = -z_lidar
            [1, 0, 0]    # z_cam = x_lidar
        ])
        
        # 构建外参矩阵
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation_lidar2cam  # 旋转部分
        extrinsic[:3, 3] = relative_pos  # 平移部分
        
        # 注意：即使位置重合（relative_pos = 0），也需要坐标轴转换！
        # 如果直接用单位矩阵，点云投影会旋转90度并发生翻转
        
        # 构建投影矩阵: P = K @ T_lidar2cam[:3, :]
        projection_matrix = build_projection_matrix(K, extrinsic)
        
        # 转换为Tensor (3, 4)
        projection_tensor = torch.from_numpy(projection_matrix).float()
        
        return {
            "img": img_tensor,
            "points": points_tensor,  # MinkUNet需要这个
            "projection_matrix": projection_tensor,
            "meta": {
                "scenario": sample['scenario'],
                "frame_id": sample['frame_id'],
                "original_size": (orig_h, orig_w)
            }
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        自定义collate_fn用于处理点云列表 (MinkowskiEngine需要list或batch coords)
        
        Args:
            batch: 批次数据列表
            
        Returns:
            batched_data: 批处理后的数据字典
        """
        images = torch.stack([b['img'] for b in batch])
        projection_matrices = torch.stack([b['projection_matrix'] for b in batch])
        
        # MinkowskiEngine通常要求点云作为List[Tensor]或(BatchIdx, x, y, z)格式
        # 这里我们简单地把points作为一个list返回，让Model内部处理
        points_list = [b['points'] for b in batch]
        
        # 元数据
        meta_list = [b['meta'] for b in batch]
        
        return {
            "img": images,
            "points": points_list,  # List[Tensor]
            "projection_matrix": projection_matrices,
            "meta": meta_list
        }

