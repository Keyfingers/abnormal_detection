import numpy as np
import os
import json

base_dir = '/root/autodl-tmp/dataset/anovox/AnoVox_Static_Mono_Town10_1/AnoVox_Dynamic_Multi_Town10/Scenario_31144b1f-2345-4c7d-919b-3997409c024a'
path = os.path.join(base_dir, 'VOXEL_GRID/VOXEL_GRID_1000.npy')
json_path = os.path.join(base_dir, 'sensor_setup.json')

# Load Voxel Grid
data = np.load(path)
labels = data[:, 3]
print(f"Unique labels in Voxel Grid: {np.unique(labels)}")

# Load Sensor Setup
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        setup = json.load(f)
    print(json.dumps(setup, indent=2))
else:
    print("sensor_setup.json not found")
