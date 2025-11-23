import numpy as np
import os

base_dir = '/root/autodl-tmp/dataset/anovox/AnoVox_Static_Mono_Town10_1/AnoVox_Dynamic_Multi_Town10/Scenario_31144b1f-2345-4c7d-919b-3997409c024a'
lidar_path = os.path.join(base_dir, 'LIDAR(0, 0, 1.8)(0, 0, 0)_8567646834588022690/LIDAR(0, 0, 1.8)(0, 0, 0)_8567646834588022690_1000.npy')
sem_lidar_path = os.path.join(base_dir, 'SEMANTIC-LIDAR(0, 0, 1.8)(0, 0, 0)_8567646834588022690/SEMANTIC-LIDAR(0, 0, 1.8)(0, 0, 0)_8567646834588022690_1000.npy')

# Load LIDAR
lidar = np.load(lidar_path)
print(f"LIDAR shape: {lidar.shape}")

# Load SEMANTIC-LIDAR
if os.path.exists(sem_lidar_path):
    sem_lidar = np.load(sem_lidar_path)
    print(f"SEMANTIC-LIDAR shape: {sem_lidar.shape}")
    print(f"Sample: {sem_lidar[:5]}")
    
    # Check if shapes match
    if lidar.shape[0] == sem_lidar.shape[0]:
        print("Shapes match! Can use Semantic LIDAR for Ground Truth.")
        # Check for label 15
        # Semantic LIDAR usually contains [x, y, z, cos(angle), instance_id, semantic_id]
        # Need to check format.
        pass
else:
    print("SEMANTIC-LIDAR not found")


