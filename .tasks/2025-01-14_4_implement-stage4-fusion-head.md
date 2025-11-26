# 背景
文件名：2025-01-14_4_implement-stage4-fusion-head.md
创建于：2025-01-14
创建者：root
主分支：main
任务分支：task/implement-stage4-fusion-head_2025-01-14_4
Yolo模式：Ask

# 任务描述
实现阶段四：轻量级融合头（Fusion Head），完成模型架构的最后一块拼图。

# 项目概览
本项目旨在实现一个融合图像和点云的道路异常检测系统，采用五阶段架构：
1. 阶段一：冻结的Mask2Former（2D图像分支）✅
2. 阶段二：冻结的MinkUNet（3D点云分支）✅
3. 阶段三：Feature Splatting投影（核心创新）✅
4. 阶段四：轻量级融合头 - 当前任务 ✅
5. 阶段五：训练与评估

# 核心设计

## 1. 融合头架构

### 适配器（Adapter）
- **维度对齐**：将2D（256维）和3D（128维）特征投影到统一维度（64维）
- **特征交互**：使用1x1卷积或交叉注意力机制（可选）
- **参数量**：<5%总参数量（符合PEFT原则）

### 门控机制（Gating）
- **动态权重**：学习在每个像素点更信任图像还是点云
- **自适应**：例如在黑暗隧道中，自动提高点云特征的权重

### 异常判定（Decision）
- **语义-几何不一致性**：基于2D和3D特征的不一致性判定异常
- **输出**：Sigmoid激活的异常概率图 (H, W, 1)

## 2. 训练策略：自监督合成异常

### 问题
训练数据只有Normal数据，直接使用Focal Loss会导致模型崩溃（永远输出0）

### 解决方案
**在线生成伪异常**（Self-Supervised Synthetic Anomaly）：
- 训练阶段：仅使用Normal数据，在线生成伪异常
  - 随机在特征图上注入噪声（高斯噪声或特征打乱）
  - 生成异常掩码（标签：0=正常，1=异常）
- 验证阶段：使用真实的AnoVox异常数据

## 3. 损失函数

**组合损失**：`Loss = Focal Loss + Dice Loss`

- **Focal Loss**：解决正负样本不平衡（异常区域通常很小）
- **Dice Loss**：优化分割边界（异常检测本质上是分割任务）

# 实现内容

## 已创建文件

1. **src/models/fusion_head.py**
   - `FusionHead`类：融合头主模块
   - `CrossAttention`类：交叉注意力机制（可选）

2. **src/losses/anomaly_loss.py**
   - `FocalLoss`类：Focal Loss实现
   - `DiceLoss`类：Dice Loss实现
   - `AnomalyDetectionLoss`类：组合损失函数

3. **src/utils/pseudo_anomaly.py**
   - `generate_random_box_mask()`：生成随机矩形掩码
   - `inject_feature_noise()`：特征层噪声注入
   - `generate_pseudo_anomalies()`：在线生成伪异常

4. **src/models/anomaly_detector.py**
   - `AnomalyDetector`类：端到端模型（整合所有阶段）

5. **src/training/train_anomaly_detector.py**
   - 完整训练脚本框架
   - 包含伪异常生成训练流程

6. **tests/test_fusion_head.py**
   - 融合头单元测试（8个测试项）

# 任务进度

[2025-01-14]
- ✅ 已完成：融合头架构设计（适配器 + 门控机制 + 异常判定）
- ✅ 已完成：Focal Loss和Dice Loss实现
- ✅ 已完成：伪异常生成函数实现
- ✅ 已完成：端到端模型类创建
- ✅ 已完成：训练脚本框架创建
- ✅ 已完成：单元测试编写
- ✅ 已完成：README文档更新
- ✅ 已完成：代码修复（导入F、参数顺序、梯度裁剪）
- ✅ 已完成：所有单元测试通过（8/8）

# 技术细节

## 融合头架构

```
输入: F_img (B, 256, H, W) + F_pts (B, 128, H, W)
  ↓
维度对齐 (Adapter)
  ↓
特征交互 (1x1 Conv 或 Cross-Attention)
  ↓
瓶颈结构 (降维 -> 特征提取 -> 升维)
  ↓
门控机制 (动态权重)
  ↓
异常判定 (Sigmoid)
  ↓
输出: Anomaly Map (B, 1, H, W)
```

## 训练流程

```
1. 提取特征（Backbone冻结）
   - 2D特征：Semantic2DBranch
   - 3D特征：Geometric3DBranch + FeatureSplatting

2. 在线生成伪异常
   - 随机掩码生成
   - 特征层噪声注入
   - 生成异常标签

3. 融合头前向传播（可训练）
   - 特征融合
   - 异常判定

4. 计算损失
   - Focal Loss + Dice Loss

5. 反向传播（仅更新FusionHead）
```

# 下一步

- [ ] 实现AnoVox数据集DataLoader
- [ ] 开始阶段五：训练与评估
- [ ] 实现评估指标（FPR95、AUPR等）

# 最终审查

## 完成情况

阶段四：轻量级融合头已全部完成 ✅

### 核心模块
1. **FusionHead** (`src/models/fusion_head.py`)
   - 适配器（Adapter）：维度对齐和特征交互
   - 门控机制（Gating）：动态权重学习
   - 异常判定：基于语义-几何不一致性
   - 参数量：<5%总参数量（符合PEFT原则）

2. **损失函数** (`src/losses/anomaly_loss.py`)
   - Focal Loss：解决正负样本不平衡
   - Dice Loss：优化分割边界
   - 组合损失：Focal Loss + Dice Loss

3. **伪异常生成** (`src/utils/pseudo_anomaly.py`)
   - 随机矩形掩码生成
   - 特征层噪声注入（高斯噪声/特征打乱）
   - 在线合成异常策略

4. **端到端模型** (`src/models/anomaly_detector.py`)
   - 整合所有阶段（Semantic2D + Geometric3D + FeatureSplatting + FusionHead）
   - 冻结前三个阶段，仅训练FusionHead
   - 支持特征提取和端到端推理

5. **训练脚本** (`src/training/train_anomaly_detector.py`)
   - 完整训练流程框架
   - 伪异常生成训练策略
   - 验证流程（使用真实异常数据）

### 测试验证
- ✅ 所有单元测试通过（8/8）
- ✅ 代码无linter错误
- ✅ 导入和基本功能正常

### 代码质量
- ✅ 修复了导入问题（添加torch.nn.functional）
- ✅ 修复了参数顺序问题（Python语法）
- ✅ 修复了梯度裁剪问题（参数列表转换）

### 下一步
- [ ] 实现AnoVox数据集DataLoader
- [ ] 开始阶段五：训练与评估
- [ ] 实现评估指标（FPR95、AUPR等）

## 总结

阶段四已完全实现，所有核心功能都已就绪，代码质量良好，测试全部通过。可以进入阶段五（训练与评估）的开发。

