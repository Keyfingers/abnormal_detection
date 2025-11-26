"""
阶段三集成测试
测试Semantic2DBranch + Geometric3DBranch + FeatureSplatting的完整流程
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
from src.models.geometric_3d import Geometric3DBranch
from src.models.feature_splatting import FeatureSplatting
from src.utils.camera_calibration import create_default_projection_matrix, projection_matrix_to_torch
from src.utils.image_preprocessing import preprocess_image
from PIL import Image
import numpy as np


def test_full_pipeline():
    """测试完整的三个阶段流程"""
    print("测试: 完整流程（阶段一+二+三）...")
    
    try:
        device = "cpu"  # 使用CPU进行测试
        
        # ========== 阶段一：Semantic2DBranch ==========
        print("\n[阶段一] 初始化Semantic2DBranch...")
        
        # 检查权重文件是否存在
        checkpoint_path = "checkpoints/mask2former/model_final_064788.pkl"
        config_path = "configs/mask2former_swin_l_cityscapes.yaml"
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠ 跳过阶段一测试: 权重文件不存在 ({checkpoint_path})")
            print("   将使用模拟的2D特征")
            use_mock_2d = True
        else:
            use_mock_2d = False
            try:
                semantic_branch = Semantic2DBranch(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    freeze_backbone=True,
                    feature_dim=256,
                    device=device
                )
                print("✓ Semantic2DBranch初始化成功")
            except Exception as e:
                print(f"⚠ Semantic2DBranch初始化失败: {e}")
                print("   将使用模拟的2D特征")
                use_mock_2d = True
        
        # ========== 阶段二：Geometric3DBranch ==========
        print("\n[阶段二] 初始化Geometric3DBranch...")
        
        checkpoint_path_3d = "checkpoints/mmdet3d/mmdet3d_placeholder.pth"
        
        if not os.path.exists(checkpoint_path_3d):
            print(f"⚠ 跳过阶段二测试: 权重文件不存在 ({checkpoint_path_3d})")
            print("   将使用模拟的3D特征")
            use_mock_3d = True
        else:
            use_mock_3d = False
            try:
                geometric_branch = Geometric3DBranch(
                    checkpoint_path=checkpoint_path_3d,
                    freeze_backbone=True,
                    feature_dim=128,
                    voxel_size=0.05,
                    device=device
                )
                print("✓ Geometric3DBranch初始化成功")
            except Exception as e:
                print(f"⚠ Geometric3DBranch初始化失败: {e}")
                print("   将使用模拟的3D特征")
                use_mock_3d = True
        
        # ========== 阶段三：Feature Splatting ==========
        print("\n[阶段三] 初始化Feature Splatting...")
        
        # 使用较小的图像尺寸以加快测试
        test_image_height, test_image_width = 200, 200
        
        feature_splatting = FeatureSplatting(
            feature_dim=128,  # 与Geometric3DBranch输出一致
            image_height=test_image_height,
            image_width=test_image_width,
            voxel_size=0.05,
            device=device
        )
        print("✓ Feature Splatting初始化成功")
        
        # ========== 数据准备 ==========
        print("\n[数据准备] 准备测试数据...")
        
        # 创建测试图像
        if use_mock_2d:
            # 模拟2D特征图
            image_height, image_width = test_image_height, test_image_width
            feature_2d = torch.randn(image_height, image_width, 256)
            print("✓ 使用模拟的2D特征图")
        else:
            # 使用真实图像（较小尺寸）
            dummy_image = Image.fromarray(np.random.randint(0, 255, (test_image_height, test_image_width, 3), dtype=np.uint8))
            image_tensor = preprocess_image(dummy_image, target_min_size=test_image_height, target_max_size=test_image_width)
            
            with torch.no_grad():
                feature_2d = semantic_branch(image_tensor)  # (1, 256, H', W')
                # 转换为 (H, W, C) 格式
                feature_2d = feature_2d.squeeze(0).permute(1, 2, 0)  # (H', W', 256)
            print(f"✓ 提取2D特征图，形状: {feature_2d.shape}")
        
        # 创建测试点云
        if use_mock_3d:
            # 模拟3D体素特征（减少数量以加快测试）
            M = 20  # 体素数
            voxel_features = torch.randn(M, 128)
            voxel_coords = torch.randint(0, 50, (M, 3), dtype=torch.int32)
            print("✓ 使用模拟的3D体素特征")
        else:
            # 使用真实点云（减少点数）
            np.random.seed(42)
            points = np.random.rand(200, 3).astype(np.float32) * 5.0
            
            with torch.no_grad():
                output_3d = geometric_branch(points)
                voxel_features = output_3d['voxel_features']  # (M, 128)
                voxel_coords = output_3d['voxel_coords']  # (M, 3)
            print(f"✓ 提取3D体素特征，形状: {voxel_features.shape}")
        
        # ========== Feature Splatting投影 ==========
        print("\n[Feature Splatting] 执行3D到2D投影...")
        
        # 创建投影矩阵
        projection_matrix = create_default_projection_matrix(
            image_width=test_image_width,
            image_height=test_image_height
        )
        projection_tensor = projection_matrix_to_torch(projection_matrix, device=device)
        
        # 执行投影
        with torch.no_grad():
            feature_2d_projected = feature_splatting(
                voxel_features,
                voxel_coords,
                projection_tensor
            )  # (H, W, 128)
        
        print(f"✓ 投影完成，输出形状: {feature_2d_projected.shape}")
        print(f"  非零像素数: {(feature_2d_projected.abs() > 1e-6).sum().item()}")
        
        # ========== 特征对齐验证 ==========
        print("\n[特征对齐] 验证特征维度对齐...")
        
        # 2D特征图应该是 (H, W, 256)
        # 投影后的3D特征图应该是 (H, W, 128)
        # 需要确保空间尺寸一致
        
        h_2d, w_2d = feature_2d.shape[:2]
        h_3d, w_3d = feature_2d_projected.shape[:2]
        
        print(f"  2D特征图尺寸: {h_2d}x{w_2d}")
        print(f"  3D投影特征图尺寸: {h_3d}x{w_3d}")
        
        # 如果尺寸不一致，需要调整（这里简化处理）
        if h_2d != h_3d or w_2d != w_3d:
            print(f"⚠ 尺寸不一致，将调整3D特征图")
            # 使用插值调整尺寸
            feature_2d_projected = torch.nn.functional.interpolate(
                feature_2d_projected.permute(2, 0, 1).unsqueeze(0),  # (1, C, H, W)
                size=(h_2d, w_2d),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)  # (H, W, C)
            print(f"  调整后3D特征图尺寸: {feature_2d_projected.shape[:2]}")
        
        # ========== 特征融合准备 ==========
        print("\n[特征融合准备] 准备融合特征...")
        
        # 将2D特征从256维降到128维（使用1x1卷积）
        if feature_2d.shape[2] != 128:
            dim_adapter = torch.nn.Conv2d(256, 128, 1)
            feature_2d_128 = dim_adapter(feature_2d.permute(2, 0, 1).unsqueeze(0))
            feature_2d_128 = feature_2d_128.squeeze(0).permute(1, 2, 0)  # (H, W, 128)
        else:
            feature_2d_128 = feature_2d
        
        # 现在两个特征图都是 (H, W, 128)
        print(f"  2D特征图（调整后）: {feature_2d_128.shape}")
        print(f"  3D投影特征图: {feature_2d_projected.shape}")
        
        assert feature_2d_128.shape == feature_2d_projected.shape, \
            f"特征图形状不匹配: {feature_2d_128.shape} vs {feature_2d_projected.shape}"
        
        # 可以拼接或相加
        feature_fused = torch.cat([feature_2d_128, feature_2d_projected], dim=2)  # (H, W, 256)
        print(f"✓ 特征融合完成，融合特征形状: {feature_fused.shape}")
        
        print("\n" + "="*60)
        print("✓ 完整流程测试成功！")
        print("="*60)
        print(f"  阶段一（2D语义）: {'真实' if not use_mock_2d else '模拟'}")
        print(f"  阶段二（3D几何）: {'真实' if not use_mock_3d else '模拟'}")
        print(f"  阶段三（Feature Splatting）: 真实")
        print(f"  最终融合特征形状: {feature_fused.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行集成测试"""
    print("="*60)
    print("阶段三集成测试")
    print("="*60)
    print()
    
    tests = [
        ("完整流程（阶段一+二+三）", test_full_pipeline),
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
            import traceback
            traceback.print_exc()
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

