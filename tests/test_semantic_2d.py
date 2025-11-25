"""
Semantic2DBranch单元测试
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.semantic_2d import Semantic2DBranch
from src.utils.image_preprocessing import preprocess_image, load_image


def test_model_initialization():
    """测试模型初始化"""
    print("测试: 模型初始化...")
    
    config_path = "configs/mask2former_swin_l_cityscapes.yaml"
    checkpoint_path = "checkpoints/mask2former/model_final_064788.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 跳过测试: 权重文件不存在 ({checkpoint_path})")
        print("   请先运行 scripts/download_mask2former_weights.py 下载权重")
        return False
    
    try:
        model = Semantic2DBranch(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=256,
            device="cpu"  # 使用CPU进行测试
        )
        print("✓ 模型初始化成功")
        return True
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return False


def test_forward_pass():
    """测试前向传播"""
    print("测试: 前向传播...")
    
    config_path = "configs/mask2former_swin_l_cityscapes.yaml"
    checkpoint_path = "checkpoints/mask2former/model_final_064788.pkl"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 跳过测试: 权重文件不存在 ({checkpoint_path})")
        return False
    
    try:
        model = Semantic2DBranch(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=256,
            device="cpu"
        )
        
        # 创建测试输入（模拟预处理后的图像）
        batch_size = 1
        test_input = torch.randn(batch_size, 3, 800, 800)
        
        # 前向传播
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ 前向传播成功")
        print(f"  输入形状: {test_input.shape}")
        print(f"  输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_dimension():
    """验证输出特征维度为256"""
    print("测试: 特征维度验证...")
    
    config_path = "configs/mask2former_swin_l_cityscapes.yaml"
    checkpoint_path = "checkpoints/mask2former/model_final_064788.pkl"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 跳过测试: 权重文件不存在 ({checkpoint_path})")
        return False
    
    try:
        model = Semantic2DBranch(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=256,
            device="cpu"
        )
        
        test_input = torch.randn(1, 3, 800, 800)
        
        with torch.no_grad():
            output = model(test_input)
        
        expected_dim = 256
        actual_dim = output.shape[1]
        
        if actual_dim == expected_dim:
            print(f"✓ 特征维度正确: {actual_dim}")
            return True
        else:
            print(f"✗ 特征维度错误: 期望 {expected_dim}, 实际 {actual_dim}")
            return False
            
    except Exception as e:
        print(f"✗ 特征维度验证失败: {e}")
        return False


def test_frozen_parameters():
    """验证参数被正确冻结"""
    print("测试: 参数冻结验证...")
    
    config_path = "configs/mask2former_swin_l_cityscapes.yaml"
    checkpoint_path = "checkpoints/mask2former/model_final_064788.pkl"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 跳过测试: 权重文件不存在 ({checkpoint_path})")
        return False
    
    try:
        model = Semantic2DBranch(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            feature_dim=256,
            device="cpu"
        )
        
        # 检查所有参数的requires_grad状态
        frozen_params = sum(1 for p in model.model.parameters() if not p.requires_grad)
        total_params = sum(1 for _ in model.model.parameters())
        
        if frozen_params == total_params:
            print(f"✓ 所有参数已冻结 ({frozen_params}/{total_params})")
            return True
        else:
            print(f"✗ 参数冻结不完整: {frozen_params}/{total_params}")
            return False
            
    except Exception as e:
        print(f"✗ 参数冻结验证失败: {e}")
        return False


def test_preprocessing():
    """测试图像预处理"""
    print("测试: 图像预处理...")
    
    try:
        # 创建测试图像（随机numpy数组）
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        # 预处理
        preprocessed = preprocess_image(test_image)
        
        # 验证输出形状和类型
        assert isinstance(preprocessed, torch.Tensor), "输出应该是torch.Tensor"
        assert preprocessed.dim() == 4, "输出应该有4个维度 (B, C, H, W)"
        assert preprocessed.shape[0] == 1, "batch维度应该是1"
        assert preprocessed.shape[1] == 3, "通道数应该是3"
        
        print(f"✓ 图像预处理成功")
        print(f"  输入形状: {test_image.shape}")
        print(f"  输出形状: {preprocessed.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 图像预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("Semantic2DBranch单元测试")
    print("="*60)
    print()
    
    results = []
    
    # 运行测试
    results.append(("图像预处理", test_preprocessing()))
    results.append(("模型初始化", test_model_initialization()))
    results.append(("前向传播", test_forward_pass()))
    results.append(("特征维度", test_feature_dimension()))
    results.append(("参数冻结", test_frozen_parameters()))
    
    # 打印测试结果
    print()
    print("="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s} {status}")
    
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

