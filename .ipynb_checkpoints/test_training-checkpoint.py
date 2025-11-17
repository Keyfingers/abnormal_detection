#!/usr/bin/env python
"""
测试训练脚本：验证代码能否正常运行
"""
import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

def test_data_loading():
    """测试数据加载"""
    print("=" * 50)
    print("测试1: 数据加载")
    print("=" * 50)
    
    try:
        from src.data.nuscenes_dataset import NuScenesPointCloudDataset
        
        data_root = "/root/autodl-tmp/dataset/nuscenes"
        
        print(f"数据路径: {data_root}")
        print(f"路径存在: {os.path.exists(data_root)}")
        
        # 创建数据集
        dataset = NuScenesPointCloudDataset(
            data_root=data_root,
            version=None,  # 自动检测
            split='train',
            image_size=(512, 1024),  # 使用较小的尺寸以加快测试
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试加载一个样本
            sample = dataset[0]
            print(f"样本键: {sample.keys()}")
            print(f"图像形状: {sample['image'].shape}")
            print(f"点云形状: {sample['point_cloud'].shape}")
            print(f"点云数量: {len(sample['point_cloud'])}")
            print("✓ 数据加载成功")
            return True
        else:
            print("✗ 数据集为空")
            return False
            
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "=" * 50)
    print("测试2: 模型前向传播")
    print("=" * 50)
    
    try:
        # 直接导入3D模型（避免detectron2依赖）
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "geometric_3d", 
            os.path.join(os.path.dirname(__file__), "src/models/geometric_3d.py")
        )
        geometric_3d_module = importlib.util.module_from_spec(spec)
        # 临时禁用semantic_2d导入
        import sys
        original_modules = sys.modules.copy()
        sys.modules['src.models.semantic_2d'] = None
        spec.loader.exec_module(geometric_3d_module)
        Geometric3DBranch = geometric_3d_module.Geometric3DBranch
        
        # 创建模型
        model = Geometric3DBranch(
            in_channels=3,
            feature_dim=128,
            checkpoint_path=None,
            freeze_backbone=False,
            voxel_size=0.05,
        )
        
        print(f"模型创建成功")
        
        # 创建测试点云
        test_point_clouds = [
            np.random.randn(1000, 3).astype(np.float32),
            np.random.randn(1500, 3).astype(np.float32),
        ]
        
        print(f"测试点云数量: {len(test_point_clouds)}")
        
        # 前向传播（训练模式，应用mask）
        model.train()
        results = model(
            test_point_clouds,
            return_features=True,
            return_reconstruction_error=True,
            apply_mask=True,
            mask_ratio=0.3,
        )
        
        print(f"结果键: {results.keys()}")
        print(f"3D特征类型: {type(results['features_3d'])}")
        print(f"重建误差形状: {results['reconstruction_error'].shape}")
        print("✓ 模型前向传播成功")
        return True
        
    except Exception as e:
        print(f"✗ 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n" + "=" * 50)
    print("测试3: 训练步骤")
    print("=" * 50)
    
    try:
        from src.models.geometric_3d import Geometric3DBranch
        from torch.utils.data import DataLoader
        from src.data.nuscenes_dataset import NuScenesPointCloudDataset
        import torch.nn as nn
        import torch.optim as optim
        
        data_root = "/root/autodl-tmp/dataset/nuscenes"
        
        # 创建数据集
        dataset = NuScenesPointCloudDataset(
            data_root=data_root,
            version=None,
            split='train',
            image_size=(512, 1024),
        )
        
        if len(dataset) == 0:
            print("✗ 数据集为空，跳过训练步骤测试")
            return False
        
        # 创建数据加载器
        def collate_fn(batch):
            images = torch.stack([item['image'] for item in batch])
            point_clouds = [item['point_cloud'] for item in batch]
            return {
                'images': images,
                'point_clouds': point_clouds,
            }
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 使用0避免多进程问题
            collate_fn=collate_fn,
        )
        
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        model = Geometric3DBranch(
            in_channels=3,
            feature_dim=128,
            checkpoint_path=None,
            freeze_backbone=False,
            voxel_size=0.05,
        )
        model = model.to(device)
        
        # 创建优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # 测试一个batch
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # 只测试第一个batch
                break
            
            point_clouds = batch['point_clouds']
            valid_pcs = [pc for pc in point_clouds if len(pc) > 0]
            
            if len(valid_pcs) == 0:
                print("✗ 没有有效的点云")
                return False
            
            optimizer.zero_grad()
            
            # 前向传播
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
            
            # 简化损失计算（如果坐标匹配）
            if len(original_features) == len(reconstruction.F):
                loss = criterion(original_features, reconstruction.F)
            else:
                # 使用平均误差作为简化损失
                loss = torch.mean(torch.norm(original_features, dim=1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
            print("✓ 训练步骤成功")
            return True
        
        print("✗ 没有可用的batch")
        return False
        
    except Exception as e:
        print(f"✗ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("开始测试训练流程")
    print("=" * 50)
    
    results = []
    
    # 测试1: 数据加载
    results.append(("数据加载", test_data_loading()))
    
    # 测试2: 模型前向传播
    results.append(("模型前向传播", test_model_forward()))
    
    # 测试3: 训练步骤（如果数据加载成功）
    if results[0][1]:
        results.append(("训练步骤", test_training_step()))
    
    # 打印总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ 所有测试通过！代码可以正常运行训练。")
        print("\n建议运行命令:")
        print("python src/training/train_3d_branch.py \\")
        print("    --data-root /root/autodl-tmp/dataset/nuscenes \\")
        print("    --output-dir outputs/geometric_3d \\")
        print("    --batch-size 2 \\")
        print("    --num-epochs 10 \\")
        print("    --num-workers 0")
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

