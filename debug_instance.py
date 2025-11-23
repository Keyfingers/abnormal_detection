import os
import cv2
import numpy as np
from PIL import Image

base_dir = '/root/autodl-tmp/dataset/anovox/AnoVox_Static_Mono_Town10_1/AnoVox_Dynamic_Multi_Town10/Scenario_31144b1f-2345-4c7d-919b-3997409c024a'
semantic_dir = os.path.join(base_dir, 'SEMANTIC-CAM(0, 0, 1.8)(0, 0, 0)_8112341084538454858')
csv_path = os.path.join(base_dir, 'ANOMALY', 'ANOMALY_1000.csv')
img_path = os.path.join(semantic_dir, 'SEMANTIC-CAM(0, 0, 1.8)(0, 0, 0)_8112341084538454858_1000.png')

# Read CSV
anomaly_tags = []
with open(csv_path, 'r') as f:
    for line in f:
        if 'semantic_tags' in line:
            # Parse "[15]" -> [15]
            tags_str = line.split(';')[1].strip().replace('[','').replace(']','')
            anomaly_tags = [int(x) for x in tags_str.split(',')]
            print(f"Anomaly Tags from CSV: {anomaly_tags}")
            break

# Read Image
if os.path.exists(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Read as is (BGR or BGRA or 16bit)
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    # CARLA Semantic is usually in R channel (or just single channel)
    if len(img.shape) == 3:
        # Usually encoded in Red channel
        semantic_ids = img[:,:,2] # BGR -> R is index 2
    else:
        semantic_ids = img
        
    print(f"Unique IDs in image: {np.unique(semantic_ids)}")
    
    found = False
    for tag in anomaly_tags:
        if tag in semantic_ids:
            print(f"Anomaly Tag {tag} found in Semantic Image!")
            mask = (semantic_ids == tag).astype(np.uint8)
            print(f"Anomaly pixels: {mask.sum()}")
            found = True
    
    if not found:
        print("Anomaly Tag NOT found in Semantic Image.")
else:
    print("Image not found")

