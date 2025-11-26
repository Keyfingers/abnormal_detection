"""
AnoVox数据集加载器测试
"""
import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torchvision import transforms
from torch.utils.data import DataLoader
from src.datasets.anovox_dataset import AnoVoxDataset


def test_dataset_loading():
    """测试数据集加载"""
    print("="*60)
    print("测试AnoVox数据集加载器")
    print("="*60)
    
    # 检查数据集路径
    data_root = os.environ.get('ANOVOX_DATA_ROOT', None)
    if data_root is None:
        print("警告: 未设置ANOVOX_DATA_ROOT环境变量")
        print("请设置环境变量: export ANOVOX_DATA_ROOT=/path/to/AnoVox_Normality_Mono_Town03")
        print("或修改此脚本中的data_root变量")
        return False
    
    if not os.path.exists(data_root):
        print(f"错误: 数据集路径不存在: {data_root}")
        return False
    
    print(f"数据集路径: {data_root}")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((800, 1333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # 创建数据集
        print("\n创建数据集...")
        dataset = AnoVoxDataset(
            root_dir=data_root,
            transform=transform
        )
        
        print(f"✓ 数据集创建成功")
        print(f"  样本数量: {len(dataset)}")
        
        # 测试单个样本
        print("\n测试单个样本...")
        sample = dataset[0]
        
        print(f"✓ 样本加载成功")
        print(f"  图像形状: {sample['img'].shape}")
        print(f"  点云形状: {sample['points'].shape}")
        print(f"  投影矩阵形状: {sample['projection_matrix'].shape}")
        print(f"  场景: {sample['meta']['scenario']}")
        print(f"  帧ID: {sample['meta']['frame_id']}")
        
        # 测试DataLoader
        print("\n测试DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 测试时设为0避免多进程问题
            collate_fn=AnoVoxDataset.collate_fn
        )
        
        batch = next(iter(dataloader))
        
        print(f"✓ DataLoader测试成功")
        print(f"  批次图像形状: {batch['img'].shape}")
        print(f"  批次点云数量: {len(batch['points'])}")
        print(f"  批次投影矩阵形状: {batch['projection_matrix'].shape}")
        
        # 验证数据格式
        assert batch['img'].dim() == 4, "图像应该是4D (B, C, H, W)"
        assert isinstance(batch['points'], list), "点云应该是列表"
        assert batch['projection_matrix'].dim() == 3, "投影矩阵应该是3D (B, 3, 4)"
        
        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)

