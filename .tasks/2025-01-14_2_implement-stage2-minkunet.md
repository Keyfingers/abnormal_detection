# 背景
文件名：2025-01-14_2_implement-stage2-minkunet.md
创建于：2025-11-25_19:42:47
创建者：root
主分支：main
任务分支：task/implement-stage2-minkunet_2025-01-14_2
Yolo模式：Ask

# 任务描述
实现阶段二：冻结的3D点云分支（使用MMDetection3D），用于提取点云的几何特征。

**技术方案变更**：由于MinkUNet的NuScenes预训练权重不可用，改用MMDetection3D框架，使用SemanticKITTI预训练权重进行微调。

# 项目概览
本项目旨在实现一个融合图像和点云的道路异常检测系统，采用五阶段架构：
1. 阶段一：冻结的Mask2Former（2D图像分支）✅
2. 阶段二：冻结的MinkUNet（3D点云分支）- 当前任务
3. 阶段三：Feature Splatting投影（核心创新）
4. 阶段四：轻量级融合头
5. 阶段五：训练与评估

核心创新点：
- 冻结骨干网络范式（Frozen Backbone Paradigm）
- Feature Splatting特征对齐机制
- 轻量级适配器（Adapter）思想

数据集：AnoVox基准测试数据集（已下载）

⚠️ 警告：永远不要修改此部分 ⚠️
核心RIPER-5协议规则：
- 必须在每个响应开头声明模式 [MODE: MODE_NAME]
- RESEARCH模式：只允许观察、阅读、提问，禁止建议和实施
- INNOVATE模式：只允许讨论方案，禁止具体规划
- PLAN模式：只允许详细规划，禁止实施
- EXECUTE模式：只允许按照计划实施，禁止偏离
- REVIEW模式：验证实施与计划的一致性
- 未经明确许可，不能在模式之间转换
⚠️ 警告：永远不要修改此部分 ⚠️

# 分析
## 当前项目状态
- 项目目录：/root/autodl-tmp/abnormal_detection
- 阶段一已完成：Semantic2DBranch已实现并测试通过
- 已有引用：src/training/train_fusion.py中引用了Geometric3DBranch，但该模块尚未实现
- 数据集：AnoVox数据集已下载到 /root/autodl-tmp/dataset/anovox/
- Git状态：需要创建功能分支

## 技术方案分析（阶段二）
根据base-rule.mdc和用户决策，阶段二需要实现：

### 1. MMDetection3D模型选择（已变更）
- **原方案**：MinkUNet（基于Minkowski Engine）
- **新方案**：MMDetection3D框架的点云分割模型
- **模型选择**：PointNet++、SparseUNet或其他MMDetection3D支持的分割模型
- **预训练**：SemanticKITTI数据集上训练的权重（可微调）
- **架构**：MMDetection3D的点云分割backbone + decoder
- **输入**：点云数据（MMDetection3D标准格式）
- **输出**：解码器最后一层的体素特征F_vox，每个非空体素包含特征向量 v∈R^D

### 2. 冻结策略
- 冻结所有参数：MMDetection3D模型的所有层
- 目的：保留SemanticKITTI大规模数据习得的几何先验
- 作用：作为几何特征提取器，提供"平坦路面"和"立体障碍物"的几何常识

### 3. 技术细节
- 使用MMDetection3D框架
- 点云预处理（MMDetection3D标准格式）
- 加载SemanticKITTI预训练权重
- 提取解码器最后一层的体素特征

## 需要解决的问题
1. MMDetection3D安装和配置
2. SemanticKITTI预训练权重下载
3. Geometric3DBranch类重构（使用MMDetection3D）
4. MMDetection3D配置文件创建
5. 点云数据预处理（MMDetection3D格式）
6. 特征提取接口设计（输出体素特征，保持与现有接口兼容）
7. 与阶段三的接口对接（为Feature Splatting准备）

# 提议的解决方案

## 方案选择（待确认）

### 1. MinkUNet实现框架
- **选项A**：使用MinkowskiEngine官方MinkUNet实现（推荐）
  - 优势：官方支持，稳定可靠，社区资源丰富
  - 需要：安装MinkowskiEngine，获取NuScenes预训练权重
  - 实现：直接使用MinkowskiEngine提供的MinkUNet类

- **选项B**：使用其他开源实现（如OpenPCDet、mmdetection3d）
  - 优势：可能有更完整的工具链
  - 劣势：需要适配，可能不符合项目需求

### 2. 点云预处理
- **体素化**：使用MinkowskiEngine的体素化工具
- **体素尺寸**：5cm × 5cm × 5cm（根据base-rule.mdc）
- **特征提取**：提取解码器最后一层的体素特征

### 3. 接口设计
参考Semantic2DBranch的结构：
- `__init__`: checkpoint_path, freeze_backbone, feature_dim, device
- `forward`: 输入点云，输出体素特征
- 需要实现点云预处理工具（类似image_preprocessing.py）

### 4. 预训练权重
- 需要查找NuScenes上训练的MinkUNet权重
- 可能需要从MinkowskiEngine官方仓库或相关论文作者处获取

# 当前执行步骤："4. 执行阶段 - 已完成所有文件创建"

# 任务进度
[2025-11-25_19:42:47]
- 已创建：任务文件 .tasks/2025-01-14_2_implement-stage2-minkunet.md
- 状态：研究阶段 - 分析技术方案和项目需求

[2025-01-14_20:00:00]
- 已完成：执行阶段 - 创建所有核心文件（使用MinkowskiEngine）
- 已创建文件：
  * requirements.txt - 更新MinkowskiEngine安装说明
  * src/utils/pointcloud_preprocessing.py - 点云预处理工具（体素化、归一化、特征提取）
  * src/models/geometric_3d.py - Geometric3DBranch类实现（MinkowskiEngine版本）
  * scripts/download_minkunet_weights.py - 权重下载脚本（支持占位权重）
  * tests/test_geometric_3d.py - 单元测试脚本（6个测试项）
  * README.md - 更新阶段二的使用说明和安装步骤
- 已创建目录结构：checkpoints/minkunet/
- 状态：等待用户确认和测试

[2025-01-14_21:00:00]
- **技术方案变更**：改用MMDetection3D替代MinkowskiEngine
- 原因：MinkUNet的NuScenes预训练权重不可用，MMDetection3D有SemanticKITTI预训练权重
- 已完成：重构阶段 - 使用MMDetection3D替代MinkowskiEngine
- 已更新文件：
  * requirements.txt - 移除MinkowskiEngine，添加MMDetection3D安装说明
  * configs/mmdet3d_semantickitti.py - 创建MMDetection3D配置文件
  * src/models/geometric_3d.py - 重构为使用MMDetection3D
  * scripts/download_mmdet3d_weights.py - 新建MMDetection3D权重下载脚本
  * tests/test_geometric_3d.py - 更新测试脚本适配MMDetection3D
  * README.md - 更新安装和使用说明（MMDetection3D）
- 已创建目录结构：checkpoints/mmdet3d/
- 状态：重构完成，等待用户确认和测试

[2025-01-14_22:00:00]
- 已更新：添加真实的SemanticKITTI预训练权重链接
- 权重URL：https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w32_8xb2-15e_semantickitti/minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth
- 模型：MinkUNet w32 (SemanticKITTI预训练)
- 已更新文件：
  * scripts/download_mmdet3d_weights.py - 添加默认权重URL，默认模型改为minkunet
  * README.md - 更新权重下载说明，添加真实权重链接
- 状态：权重下载脚本已配置完成，可以直接使用

# 最终审查
（待完成）

