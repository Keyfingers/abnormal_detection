import numpy as np
import os

base_dir = '/root/autodl-tmp/dataset/anovox/AnoVox_Static_Mono_Town10_1/AnoVox_Dynamic_Multi_Town10/Scenario_31144b1f-2345-4c7d-919b-3997409c024a'
lidar_path = os.path.join(base_dir, 'LIDAR(0, 0, 1.8)(0, 0, 0)_8567646834588022690/LIDAR(0, 0, 1.8)(0, 0, 0)_8567646834588022690_1000.npy')
voxel_path = os.path.join(base_dir, 'VOXEL_GRID/VOXEL_GRID_1000.npy')

# Load LIDAR
if os.path.exists(lidar_path):
    lidar = np.load(lidar_path)
    print(f"LIDAR shape: {lidar.shape}, dtype: {lidar.dtype}")
    print(f"LIDAR min: {lidar[:, :3].min(axis=0)}")
    print(f"LIDAR max: {lidar[:, :3].max(axis=0)}")
else:
    print("LIDAR file not found")

# Load Voxel Grid
if os.path.exists(voxel_path):
    voxel = np.load(voxel_path)
    print(f"VOXEL shape: {voxel.shape}, dtype: {voxel.dtype}")
    print(f"VOXEL min: {voxel[:, :3].min(axis=0)}")
    print(f"VOXEL max: {voxel[:, :3].max(axis=0)}")
    
    # Try to correlate
    # Assume origin -100, -100, -100 ? 
    # Or maybe center 0,0,0 matches center of grid? 
    # Usually grid indices start at 0.


