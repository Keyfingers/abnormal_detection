"""
Geometric3DBranch单元测试
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.geometric_3d import Geometric3DBranch
from src.utils.pointcloud_preprocessing import voxelize_pointcloud, normalize_coordinates


def test_pointcloud_preprocessing():
    """测试点云预处理"""
    print("测试: 点云预处理...")
    
    try:
        # 创建测试点云（1000个点）
        np.random.seed(42)
        points = np.random.rand(1000, 3).astype(np.float32) * 10.0  # 10m x 10m x 10m区域
        
        # 测试体素化
        voxel_coords, voxel_features, point_to_voxel = voxelize_pointcloud(
            points,
            voxel_size=0.05,
            return_indices=True
        )
        
        print(f"✓ 点云预处理成功")
        print(f"  输入点数: {len(points)}")
        print(f"  体素数: {len(voxel_coords)}")
        print(f"  体素坐标形状: {voxel_coords.shape}")
        print(f"  体素特征形状: {voxel_features.shape}")
        
        # 验证输出格式
        assert len(voxel_coords) > 0, "体素数量应该大于0"
        assert voxel_coords.shape[1] == 3, "体素坐标应该是3维"
        assert voxel_features.shape[0] == len(voxel_coords), "体素特征数量应该等于体素数量"
        
        return True
        
    except Exception as e:
        print(f"✗ 点云预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """测试模型初始化"""
    print("测试: 模型初始化...")
    
    # 检查MMDetection3D是否安装
    try:
        import mmdet3d
    except ImportError:
        print("⚠ 跳过测试: MMDetection3D未安装")
        print("   请参考requirements.txt中的说明安装MMDetection3D")
        return False
    
    checkpoint_path = "checkpoints/mmdet3d/mmdet3d_placeholder.pth"
    
    # 如果占位权重不存在，尝试创建
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 权重文件不存在 ({checkpoint_path})")
        print("   尝试创建占位权重文件...")
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            placeholder_weights = {
                'state_dict': {},
                'meta': {
                    'config': 'Placeholder config for testing',
                    'epoch': 0
                },
                'note': 'Placeholder weight file for testing'
            }
            torch.save(placeholder_weights, checkpoint_path)
            print(f"✓ 已创建占位权重文件")
        except Exception as e:
            print(f"✗ 无法创建占位权重文件: {e}")
            print("   请先运行: python scripts/download_mmdet3d_weights.py --create-placeholder")
            return False
    
    try:
        model = Geometric3DBranch(
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=128,
            voxel_size=0.05,
            device="cpu"  # 使用CPU进行测试
        )
        print("✓ 模型初始化成功")
        return True
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("测试: 前向传播...")
    
    # 检查MMDetection3D是否安装
    try:
        import mmdet3d
    except ImportError:
        print("⚠ 跳过测试: MMDetection3D未安装")
        return False
    
    checkpoint_path = "checkpoints/mmdet3d/mmdet3d_placeholder.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 跳过测试: 权重文件不存在 ({checkpoint_path})")
        return False
    
    try:
        model = Geometric3DBranch(
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=128,
            voxel_size=0.05,
            device="cpu"
        )
        
        # 创建测试点云
        np.random.seed(42)
        points = np.random.rand(1000, 3).astype(np.float32) * 10.0
        
        # 前向传播
        with torch.no_grad():
            output = model(points)
        
        print(f"✓ 前向传播成功")
        print(f"  输入点数: {len(points)}")
        print(f"  输出体素特征形状: {output['voxel_features'].shape}")
        print(f"  输出体素坐标形状: {output['voxel_coords'].shape}")
        
        # 验证输出格式
        assert 'voxel_features' in output, "输出应该包含'voxel_features'"
        assert 'voxel_coords' in output, "输出应该包含'voxel_coords'"
        assert output['voxel_features'].shape[1] == 128, f"特征维度应该是128，实际是{output['voxel_features'].shape[1]}"
        
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_dimension():
    """验证输出特征维度"""
    print("测试: 特征维度验证...")
    
    # 检查MMDetection3D是否安装
    try:
        import mmdet3d
    except ImportError:
        print("⚠ 跳过测试: MMDetection3D未安装")
        return False
    
    checkpoint_path = "checkpoints/mmdet3d/mmdet3d_placeholder.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 跳过测试: 权重文件不存在 ({checkpoint_path})")
        return False
    
    try:
        model = Geometric3DBranch(
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=128,
            voxel_size=0.05,
            device="cpu"
        )
        
        # 创建测试点云
        np.random.seed(42)
        points = np.random.rand(500, 3).astype(np.float32) * 10.0
        
        with torch.no_grad():
            output = model(points)
        
        feature_dim = output['voxel_features'].shape[1]
        
        if feature_dim == 128:
            print(f"✓ 特征维度验证通过: {feature_dim}")
            return True
        else:
            print(f"✗ 特征维度不匹配: 期望128，实际{feature_dim}")
            return False
        
    except Exception as e:
        print(f"✗ 特征维度验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frozen_parameters():
    """验证参数是否被冻结"""
    print("测试: 参数冻结验证...")
    
    # 检查MMDetection3D是否安装
    try:
        import mmdet3d
    except ImportError:
        print("⚠ 跳过测试: MMDetection3D未安装")
        return False
    
    checkpoint_path = "checkpoints/mmdet3d/mmdet3d_placeholder.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 跳过测试: 权重文件不存在 ({checkpoint_path})")
        return False
    
    try:
        model = Geometric3DBranch(
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=128,
            voxel_size=0.05,
            device="cpu"
        )
        
        # 检查所有参数是否被冻结
        frozen_count = 0
        total_count = 0
        
        for name, param in model.named_parameters():
            total_count += 1
            if not param.requires_grad:
                frozen_count += 1
        
        if frozen_count == total_count and total_count > 0:
            print(f"✓ 参数冻结验证通过: {frozen_count}/{total_count} 参数已冻结")
            return True
        else:
            print(f"✗ 参数冻结验证失败: {frozen_count}/{total_count} 参数已冻结")
            return False
        
    except Exception as e:
        print(f"✗ 参数冻结验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """测试批次处理"""
    print("测试: 批次处理...")
    
    # 检查MMDetection3D是否安装
    try:
        import mmdet3d
    except ImportError:
        print("⚠ 跳过测试: MMDetection3D未安装")
        return False
    
    checkpoint_path = "checkpoints/mmdet3d/mmdet3d_placeholder.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 跳过测试: 权重文件不存在 ({checkpoint_path})")
        return False
    
    try:
        model = Geometric3DBranch(
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=128,
            voxel_size=0.05,
            device="cpu"
        )
        
        # 创建多个测试点云
        np.random.seed(42)
        point_clouds = [
            np.random.rand(500, 3).astype(np.float32) * 10.0,
            np.random.rand(800, 3).astype(np.float32) * 10.0,
            np.random.rand(300, 3).astype(np.float32) * 10.0,
        ]
        
        with torch.no_grad():
            output = model(point_clouds)
        
        print(f"✓ 批次处理成功")
        print(f"  输入点云数量: {len(point_clouds)}")
        print(f"  输出体素特征形状: {output['voxel_features'].shape}")
        print(f"  输出体素坐标形状: {output['voxel_coords'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 批次处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("Geometric3DBranch单元测试")
    print("="*60)
    print()
    
    tests = [
        ("点云预处理", test_pointcloud_preprocessing),
        ("模型初始化", test_model_initialization),
        ("前向传播", test_forward_pass),
        ("特征维度验证", test_feature_dimension),
        ("参数冻结验证", test_frozen_parameters),
        ("批次处理", test_batch_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        print("-" * 60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

