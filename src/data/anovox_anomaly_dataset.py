"""
AnoVox Anomaly Dataset Loader (Raw Scenario Format)
用于评估阶段，读取真实的异常数据（AnoVox_Static_Mono_Town10_1）
并利用 SEMANTIC-LIDAR 生成 2D Ground Truth Mask
"""
import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional

class AnoVoxAnomalyDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = 'test', # 'test' or 'val' or 'all'
        image_size: Tuple[int, int] = (512, 1024), # Default output size (H, W)
        mask_downsample: int = 1, # Downsample factor for mask (to save memory if needed)
    ):
        """
        Args:
            data_root: Path to AnoVox dataset root (e.g., .../AnoVox_Static_Mono_Town10_1)
            split: 'test' loads all data (since this dataset is for testing)
            image_size: Target image size (H, W). Original is likely 512x768 or similar.
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.mask_downsample = mask_downsample
        
        # Find all Scenarios
        self.scenarios = glob.glob(os.path.join(data_root, 'AnoVox_Dynamic_Multi_Town10', 'Scenario_*'))
        if not self.scenarios:
             # Try one level up if structure is different
             self.scenarios = glob.glob(os.path.join(data_root, 'Scenario_*'))
             
        self.samples = [] # List of dicts with file paths
        
        print(f"Found {len(self.scenarios)} scenarios. Indexing samples...")
        
        for scenario_dir in self.scenarios:
            self._index_scenario(scenario_dir)
            
        print(f"Total samples indexed: {len(self.samples)}")
        
    def _index_scenario(self, scenario_dir):
        """Index a single scenario directory"""
        # 1. Parse sensor_setup.json for calibration (if available)
        sensor_setup_path = os.path.join(scenario_dir, 'sensor_setup.json')
        sensor_setup = {}
        if os.path.exists(sensor_setup_path):
            with open(sensor_setup_path, 'r') as f:
                sensor_setup = json.load(f)
        
        # 2. Find sensor directories
        # We need RGB-CAM, LIDAR, SEMANTIC-LIDAR
        # Assuming "LIDAR(0, 0, 1.8)(0, 0, 0)_..." naming convention
        # We take the first available RGB camera and corresponding LIDAR
        
        rgb_dirs = glob.glob(os.path.join(scenario_dir, 'RGB-CAM*'))
        lidar_dirs = glob.glob(os.path.join(scenario_dir, 'LIDAR*'))
        sem_lidar_dirs = glob.glob(os.path.join(scenario_dir, 'SEMANTIC-LIDAR*'))
        anomaly_dir = os.path.join(scenario_dir, 'ANOMALY')
        
        if not rgb_dirs or not lidar_dirs or not sem_lidar_dirs or not os.path.exists(anomaly_dir):
            return

        # Use the first found sensor set (usually (0,0,1.8))
        rgb_dir = rgb_dirs[0]
        lidar_dir = lidar_dirs[0]
        sem_lidar_dir = sem_lidar_dirs[0]
        
        sensor_id_rgb = os.path.basename(rgb_dir)
        sensor_id_lidar = os.path.basename(lidar_dir)
        
        # Parse calibration from sensor_setup or folder name
        # We prefer sensor_setup
        rgb_calib = sensor_setup.get(sensor_id_rgb, {})
        lidar_calib = sensor_setup.get(sensor_id_lidar, {})
        
        # Find all frames in RGB dir
        rgb_files = glob.glob(os.path.join(rgb_dir, '*.png')) + glob.glob(os.path.join(rgb_dir, '*.jpg'))
        rgb_files.sort()
        
        for rgb_path in rgb_files:
            # Extract frame ID
            # Name format: ..._FRAMEID.png
            basename = os.path.basename(rgb_path)
            frame_id_str = basename.split('_')[-1].split('.')[0]
            
            # Find corresponding LIDAR and SEMANTIC-LIDAR
            # LIDAR format: ..._FRAMEID.npy
            lidar_path = os.path.join(lidar_dir, f"{sensor_id_lidar}_{frame_id_str}.npy")
            sem_lidar_path = os.path.join(sem_lidar_dir, f"{os.path.basename(sem_lidar_dir)}_{frame_id_str}.npy")
            anomaly_csv_path = os.path.join(anomaly_dir, f"ANOMALY_{frame_id_str}.csv")
            
            if os.path.exists(lidar_path) and os.path.exists(sem_lidar_path) and os.path.exists(anomaly_csv_path):
                self.samples.append({
                    'rgb_path': rgb_path,
                    'lidar_path': lidar_path,
                    'sem_lidar_path': sem_lidar_path,
                    'anomaly_csv_path': anomaly_csv_path,
                    'rgb_calib': rgb_calib,
                    'lidar_calib': lidar_calib,
                    'frame_id': frame_id_str,
                    'scenario_id': os.path.basename(scenario_dir)
                })

    def __len__(self):
        return len(self.samples)

    def _parse_calibration(self, calib_dict, folder_name):
        """
        Parse calibration from dictionary or fallback to parsing folder name
        """
        intrinsic = np.eye(3, dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)
        
        # Default intrinsic
        H, W = self.image_size
        intrinsic[0, 0] = W  # fx approx W (90 deg FOV)
        intrinsic[1, 1] = W
        intrinsic[0, 2] = W / 2
        intrinsic[1, 2] = H / 2
        
        if 'args' in calib_dict:
            args = calib_dict['args']
            img_w = args.get('image_width', 800)
            img_h = args.get('image_height', 600)
            fov = args.get('camera_fov', 90.0)
            
            # Compute focal length from FOV
            # f = (W/2) / tan(FOV/2)
            f = (img_w / 2.0) / np.tan(np.deg2rad(fov / 2.0))
            
            intrinsic[0, 0] = f
            intrinsic[1, 1] = f
            intrinsic[0, 2] = img_w / 2.0
            intrinsic[1, 2] = img_h / 2.0
            
            # Scale intrinsic if we resize image
            scale_x = self.image_size[1] / img_w
            scale_y = self.image_size[0] / img_h
            intrinsic[0, 0] *= scale_x
            intrinsic[0, 2] *= scale_x
            intrinsic[1, 1] *= scale_y
            intrinsic[1, 2] *= scale_y
            
        if 'location' in calib_dict and 'rotation' in calib_dict:
            loc = calib_dict['location'] # x, y, z
            rot = calib_dict['rotation'] # pitch, yaw, roll (CARLA)
            
            # Convert CARLA rotation to matrix
            # CARLA: Pitch(y), Yaw(z), Roll(x)
            # But order? Usually R = Rz(yaw) * Ry(pitch) * Rx(roll)
            # Need to verify. For now assume identity R if mostly straight.
            # The dataset seems to use (0,0,0) mostly.
            pass 
            
        # NOTE: Since the dataset uses co-located sensors at (0,0,1.8), 
        # and we project LIDAR (also at 0,0,1.8) to Camera, 
        # the relative extrinsic is Identity! (assuming same orientation)
        # CARLA LIDAR and Camera have different axis conventions.
        # LIDAR: x-forward, y-right, z-up
        # Camera: x-right, y-down, z-forward
        # We need T_lidar_to_cam.
        # x_cam = y_lidar
        # y_cam = -z_lidar
        # z_cam = x_lidar
        # Matrix:
        # [[0, 1, 0, 0],
        #  [0, 0, -1, 0],
        #  [1, 0, 0, 0],
        #  [0, 0, 0, 1]]
        
        extrinsic = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        return torch.from_numpy(intrinsic), torch.from_numpy(extrinsic)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Image
        rgb_path = sample['rgb_path']
        image = Image.open(rgb_path).convert('RGB')
        image = np.array(image) # (H_orig, W_orig, 3)
        
        # Resize
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # 2. Load Point Cloud
        lidar_path = sample['lidar_path']
        # (N, 4) -> x, y, z, i
        point_cloud = np.load(lidar_path)[:, :3].astype(np.float32)
        
        # 3. Load Semantic LIDAR and Generate Mask
        sem_lidar_path = sample['sem_lidar_path']
        sem_lidar = np.load(sem_lidar_path) # (N, 4) -> x, y, z, semantic_tag (float)
        sem_points = sem_lidar[:, :3].astype(np.float32)
        sem_tags = sem_lidar[:, 3].astype(int)
        
        # Read Anomaly CSV to get target tags
        anomaly_tags = []
        with open(sample['anomaly_csv_path'], 'r') as f:
            for line in f:
                if 'semantic_tags' in line:
                    tags_str = line.split(';')[1].strip().replace('[','').replace(']','')
                    if tags_str:
                        anomaly_tags = [int(x) for x in tags_str.split(',')]
                    break
        
        # Filter anomaly points using SEMANTIC-LIDAR points
        anomaly_mask_3d = np.isin(sem_tags, anomaly_tags)
        anomaly_points = sem_points[anomaly_mask_3d]
        
        # 4. Project to 2D to get Ground Truth Mask
        # Get calibration
        camera_intrinsic, camera_extrinsic = self._parse_calibration(sample['rgb_calib'], "")
        
        mask = torch.zeros((self.image_size[0], self.image_size[1]), dtype=torch.float32)
        
        if len(anomaly_points) > 0:
            # Project
            # Transform to Camera Frame
            points_homo = np.hstack((anomaly_points, np.ones((len(anomaly_points), 1))))
            points_cam = (camera_extrinsic.numpy() @ points_homo.T).T # (N, 4)
            
            # Project to Image Plane
            points_cam = points_cam[:, :3]
            x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
            
            # Filter z > 0
            valid_z = z > 0.1
            x = x[valid_z]
            y = y[valid_z]
            z = z[valid_z]
            
            if len(z) > 0:
                K = camera_intrinsic.numpy()
                u = (K[0, 0] * x / z + K[0, 2]).astype(int)
                v = (K[1, 1] * y / z + K[1, 2]).astype(int)
                
                # Filter bounds
                H, W = self.image_size
                valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                u = u[valid_uv]
                v = v[valid_uv]
                
                # Draw on mask
                # Use cv2 to draw circles to dilate (simulate object size)
                mask_np = mask.numpy()
                for i in range(len(u)):
                    # Radius 2-3 pixels to cover gaps
                    cv2.circle(mask_np, (u[i], v[i]), 3, 1.0, -1)
                mask = torch.from_numpy(mask_np)
        
        # Downsample mask if requested (for memory efficiency)
        if self.mask_downsample > 1:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0), 
                scale_factor=1/self.mask_downsample, 
                mode='nearest'
            ).squeeze()
            
        return {
            'image': image_tensor,
            'point_cloud': point_cloud,
            'anomaly_mask': mask,
            'camera_intrinsic': camera_intrinsic,
            'camera_extrinsic': camera_extrinsic,
            'sample_id': f"{sample['scenario_id']}_{sample['frame_id']}"
        }

