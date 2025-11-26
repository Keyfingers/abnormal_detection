"""
Fusion Head单元测试
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.fusion_head import FusionHead, CrossAttention
from src.losses.anomaly_loss import FocalLoss, DiceLoss, AnomalyDetectionLoss
from src.utils.pseudo_anomaly import generate_random_box_mask, inject_feature_noise, generate_pseudo_anomalies


def test_fusion_head_initialization():
    """测试Fusion Head初始化"""
    print("测试: Fusion Head初始化...")
    
    try:
        model = FusionHead(
            img_feature_dim=256,
            pts_feature_dim=128,
            hidden_dim=64,
            device="cpu"
        )
        
        print(f"✓ Fusion Head初始化成功")
        print(f"  图像特征维度: {model.img_feature_dim}")
        print(f"  点云特征维度: {model.pts_feature_dim}")
        print(f"  隐藏维度: {model.hidden_dim}")
        return True
        
    except Exception as e:
        print(f"✗ Fusion Head初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_head_forward():
    """测试Fusion Head前向传播"""
    print("测试: Fusion Head前向传播...")
    
    try:
        model = FusionHead(
            img_feature_dim=256,
            pts_feature_dim=128,
            hidden_dim=64,
            device="cpu"
        )
        
        # 创建测试特征
        B, H, W = 2, 100, 100
        img_features = torch.randn(B, 256, H, W)
        pts_features = torch.randn(B, 128, H, W)
        
        # 前向传播
        anomaly_map = model(img_features, pts_features)
        
        assert anomaly_map.shape == (B, 1, H, W), f"输出形状错误: {anomaly_map.shape}"
        assert torch.all(anomaly_map >= 0) and torch.all(anomaly_map <= 1), "输出应该在[0,1]范围内"
        
        print(f"✓ Fusion Head前向传播成功")
        print(f"  输入形状: img={img_features.shape}, pts={pts_features.shape}")
        print(f"  输出形状: {anomaly_map.shape}")
        print(f"  输出范围: [{anomaly_map.min():.4f}, {anomaly_map.max():.4f}]")
        return True
        
    except Exception as e:
        print(f"✗ Fusion Head前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_attention():
    """测试交叉注意力机制"""
    print("测试: 交叉注意力机制...")
    
    try:
        attention = CrossAttention(dim=64)
        
        B, C, H, W = 2, 64, 50, 50
        img_feat = torch.randn(B, C, H, W)
        pts_feat = torch.randn(B, C, H, W)
        
        fused_feat = attention(img_feat, pts_feat)
        
        assert fused_feat.shape == (B, C, H, W), f"输出形状错误: {fused_feat.shape}"
        
        print(f"✓ 交叉注意力成功")
        print(f"  输出形状: {fused_feat.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 交叉注意力失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_focal_loss():
    """测试Focal Loss"""
    print("测试: Focal Loss...")
    
    try:
        loss_func = FocalLoss(alpha=0.25, gamma=2.0)
        
        B, H, W = 2, 100, 100
        pred = torch.rand(B, H, W)  # 预测概率
        target = torch.randint(0, 2, (B, H, W)).float()  # 真实标签
        
        loss = loss_func(pred, target)
        
        assert loss.item() >= 0, "损失应该非负"
        assert not torch.isnan(loss), "损失不应该是NaN"
        
        print(f"✓ Focal Loss计算成功")
        print(f"  损失值: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Focal Loss失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dice_loss():
    """测试Dice Loss"""
    print("测试: Dice Loss...")
    
    try:
        loss_func = DiceLoss()
        
        B, H, W = 2, 100, 100
        pred = torch.rand(B, H, W)
        target = torch.randint(0, 2, (B, H, W)).float()
        
        loss = loss_func(pred, target)
        
        assert loss.item() >= 0 and loss.item() <= 1, "Dice Loss应该在[0,1]范围内"
        assert not torch.isnan(loss), "损失不应该是NaN"
        
        print(f"✓ Dice Loss计算成功")
        print(f"  损失值: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Dice Loss失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_loss():
    """测试组合损失函数"""
    print("测试: 组合损失函数...")
    
    try:
        loss_func = AnomalyDetectionLoss(focal_weight=1.0, dice_weight=1.0)
        
        B, H, W = 2, 100, 100
        pred = torch.rand(B, 1, H, W)
        target = torch.randint(0, 2, (B, 1, H, W)).float()
        
        loss_dict = loss_func(pred, target)
        
        assert 'loss' in loss_dict, "应该包含总损失"
        assert 'focal_loss' in loss_dict, "应该包含Focal Loss"
        assert 'dice_loss' in loss_dict, "应该包含Dice Loss"
        assert loss_dict['loss'].item() >= 0, "总损失应该非负"
        
        print(f"✓ 组合损失函数成功")
        print(f"  总损失: {loss_dict['loss'].item():.4f}")
        print(f"  Focal Loss: {loss_dict['focal_loss'].item():.4f}")
        print(f"  Dice Loss: {loss_dict['dice_loss'].item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ 组合损失函数失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pseudo_anomaly_generation():
    """测试伪异常生成"""
    print("测试: 伪异常生成...")
    
    try:
        B, C, H, W = 2, 128, 100, 100
        img_features = torch.randn(B, C, H, W)
        pts_features = torch.randn(B, C, H, W)
        
        # 生成伪异常
        img_corrupted, pts_corrupted, anomaly_mask = generate_pseudo_anomalies(
            img_features,
            pts_features,
            anomaly_prob=1.0,  # 强制生成异常
            num_boxes=1,
            noise_type="gaussian",
            noise_scale=2.0
        )
        
        assert img_corrupted.shape == img_features.shape, "特征形状应该不变"
        assert pts_corrupted.shape == pts_features.shape, "特征形状应该不变"
        assert anomaly_mask.shape == (B, 1, H, W), f"掩码形状错误: {anomaly_mask.shape}"
        assert torch.all(anomaly_mask >= 0) and torch.all(anomaly_mask <= 1), "掩码应该在[0,1]范围内"
        
        # 检查是否有异常区域
        assert anomaly_mask.sum() > 0, "应该生成异常区域"
        
        print(f"✓ 伪异常生成成功")
        print(f"  特征形状: {img_corrupted.shape}")
        print(f"  掩码形状: {anomaly_mask.shape}")
        print(f"  异常像素数: {anomaly_mask.sum().item()}")
        return True
        
    except Exception as e:
        print(f"✗ 伪异常生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_training_step():
    """测试端到端训练步骤"""
    print("测试: 端到端训练步骤...")
    
    try:
        # 创建模型
        fusion_head = FusionHead(
            img_feature_dim=256,
            pts_feature_dim=128,
            hidden_dim=64,
            device="cpu"
        )
        
        # 创建损失函数
        loss_func = AnomalyDetectionLoss()
        
        # 创建优化器（仅优化Fusion Head）
        optimizer = torch.optim.Adam(fusion_head.parameters(), lr=1e-3)
        
        # 模拟训练步骤
        B, H, W = 2, 100, 100
        img_features = torch.randn(B, 256, H, W)
        pts_features = torch.randn(B, 128, H, W)
        
        # 生成伪异常
        img_corrupted, pts_corrupted, anomaly_mask = generate_pseudo_anomalies(
            img_features, pts_features, anomaly_prob=1.0
        )
        
        # 前向传播
        anomaly_map = fusion_head(img_corrupted, pts_corrupted)
        
        # 计算损失
        loss_dict = loss_func(anomaly_map, anomaly_mask)
        loss = loss_dict['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss), "损失不应该是NaN"
        assert loss.item() >= 0, "损失应该非负"
        
        print(f"✓ 端到端训练步骤成功")
        print(f"  损失值: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ 端到端训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("Fusion Head单元测试")
    print("="*60)
    print()
    
    tests = [
        ("Fusion Head初始化", test_fusion_head_initialization),
        ("Fusion Head前向传播", test_fusion_head_forward),
        ("交叉注意力机制", test_cross_attention),
        ("Focal Loss", test_focal_loss),
        ("Dice Loss", test_dice_loss),
        ("组合损失函数", test_combined_loss),
        ("伪异常生成", test_pseudo_anomaly_generation),
        ("端到端训练步骤", test_end_to_end_training_step),
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

