#!/usr/bin/env python
"""
快速测试3D分支训练（不依赖detectron2）
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

# 直接导入需要的模块（避免detectron2依赖）
from src.data.nuscenes_dataset import NuScenesPointCloudDataset
from src.models.geometric_3d import Geometric3DBranch
import torch.nn as nn
import torch.optim as optim


def collate_fn(batch):
    """自定义collate函数"""
    images = torch.stack([item['image'] for item in batch])
    point_clouds = [item['point_cloud'] for item in batch]
    return {
        'images': images,
        'point_clouds': point_clouds,
    }


def main():
    print("=" * 60)
    print("快速测试3D分支训练")
    print("=" * 60)
    
    # 配置
    data_root = "/root/autodl-tmp/dataset/nuscenes"
    output_dir = "outputs/geometric_3d_test"
    batch_size = 2
    num_epochs = 1  # 只测试1个epoch
    num_workers = 0  # 避免多进程问题
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据集
    print("\n加载数据集...")
    try:
        train_dataset = NuScenesPointCloudDataset(
            data_root=data_root,
            version=None,  # 自动检测
            split='train',
            image_size=(512, 1024),  # 使用较小尺寸加快测试
        )
        print(f"训练集大小: {len(train_dataset)}")
        
        val_dataset = NuScenesPointCloudDataset(
            data_root=data_root,
            version=None,
            split='val',
            image_size=(512, 1024),
        )
        print(f"验证集大小: {len(val_dataset)}")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    # 模型
    print("\n创建模型...")
    try:
        model = Geometric3DBranch(
            in_channels=3,
            feature_dim=128,
            checkpoint_path=None,
            freeze_backbone=False,
            voxel_size=0.05,
        )
        model = model.to(device)
        print("✓ 模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 训练一个epoch
    print("\n开始训练...")
    model.train()
    
    try:
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # 只测试前3个batch
                break
            
            point_clouds = batch['point_clouds']
            
            # 过滤空点云
            valid_pcs = [pc for pc in point_clouds if len(pc) > 0]
            if len(valid_pcs) == 0:
                print(f"  Batch {batch_idx}: 跳过（无有效点云）")
                continue
            
            optimizer.zero_grad()
            
            # 前向传播（应用mask）
            results = model(
                valid_pcs,
                return_features=False,
                return_reconstruction_error=False,
                apply_mask=True,
                mask_ratio=0.3,
            )
            
            # 计算损失
            original_tensor = results['sparse_tensor']
            reconstruction = results['reconstruction']
            original_features = original_tensor.F
            
            # 修复：简化损失计算（确保梯度流）
            reconstructed_features = reconstruction.F
            
            if len(original_features) == len(reconstructed_features):
                # 直接计算L2损失
                loss = criterion(original_features, reconstructed_features)
            else:
                # 使用最近邻插值（简化版本）
                # 注意：需要保持梯度，所以不能detach
                from scipy.spatial.distance import cdist
                original_coords = original_tensor.C.float()[:, 1:].cpu().numpy()
                reconstruction_coords = reconstruction.C.float()[:, 1:].cpu().numpy()
                reconstructed_features_np = reconstructed_features.detach().cpu().numpy()
                
                distances = cdist(original_coords, reconstruction_coords)
                nearest_indices = np.argmin(distances, axis=1)
                
                # 使用gather保持梯度
                nearest_reconstructed = reconstructed_features[nearest_indices]
                
                loss = criterion(original_features, nearest_reconstructed)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        print("✓ 训练成功")
        
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 验证
    print("\n开始验证...")
    model.eval()
    
    try:
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 2:  # 只测试前2个batch
                    break
                
                point_clouds = batch['point_clouds']
                valid_pcs = [pc for pc in point_clouds if len(pc) > 0]
                
                if len(valid_pcs) == 0:
                    continue
                
                results = model(
                    valid_pcs,
                    return_features=False,
                    return_reconstruction_error=False,
                    apply_mask=True,
                    mask_ratio=0.3,
                )
                
                original_tensor = results['sparse_tensor']
                reconstruction = results['reconstruction']
                original_features = original_tensor.F
                
                if len(original_features) == len(reconstruction.F):
                    loss = criterion(original_features, reconstruction.F)
                else:
                    from scipy.spatial.distance import cdist
                    original_coords = original_tensor.C.float()[:, 1:].detach().cpu().numpy()
                    reconstruction_coords = reconstruction.C.float()[:, 1:].detach().cpu().numpy()
                    reconstructed_features_np = reconstruction.F.detach().cpu().numpy()
                    
                    distances = cdist(original_coords, reconstruction_coords)
                    nearest_indices = np.argmin(distances, axis=1)
                    nearest_reconstructed = torch.from_numpy(
                        reconstructed_features_np[nearest_indices]
                    ).to(original_features.device)
                    
                    loss = criterion(original_features, nearest_reconstructed)
                
                val_loss += loss.item()
                val_batches += 1
        
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            print(f"✓ 验证成功: 平均损失 = {avg_val_loss:.4f}")
        else:
            print("⚠ 验证集为空")
        
    except Exception as e:
        print(f"✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！代码可以正常运行训练。")
    print("=" * 60)
    print("\n可以运行完整训练命令:")
    print(f"python src/training/train_3d_branch.py \\")
    print(f"    --data-root {data_root} \\")
    print(f"    --output-dir outputs/geometric_3d \\")
    print(f"    --batch-size 2 \\")
    print(f"    --num-epochs 10 \\")
    print(f"    --num-workers 0")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

