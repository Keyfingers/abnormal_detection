# 背景
文件名：2025-01-14_1_implement-stage1-mask2former.md
创建于：2025-01-14_17:04:03
创建者：root
主分支：main
任务分支：task/implement-stage1-mask2former_2025-01-14_1
Yolo模式：Ask

# 任务描述
从零开始搭建融合图像与点云的道路异常检测项目框架，按照base-rule.mdc中的完整方案实施。当前阶段：实现阶段一（冻结的Mask2Former 2D图像分支）。

# 项目概览
本项目旨在实现一个融合图像和点云的道路异常检测系统，采用五阶段架构：
1. 阶段一：冻结的Mask2Former（2D图像分支）- 当前任务
2. 阶段二：冻结的MinkUNet（3D点云分支）
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
- 已有文件：src/training/train_fusion.py（部分实现）
- 数据集：AnoVox数据集已下载到 /root/autodl-tmp/dataset/anovox/
- Git状态：已创建功能分支 task/implement-stage1-mask2former_2025-01-14_1

## 技术方案分析（阶段一）
根据base-rule.mdc，阶段一需要实现：

### 1. Mask2Former模型选择
- 模型：Mask2Former（基于Transformer的全景分割模型）
- 预训练：Cityscapes数据集上训练的权重
- 架构：包含Backbone（如Swin-Transformer）和Pixel Decoder
- 输出：Pixel Decoder输出的高分辨率特征图 F_img ∈ R^(H×W×C)

### 2. 冻结策略
- 冻结所有参数：Backbone + Pixel Decoder
- 目的：保留大规模数据习得的"正常性"先验
- 作用：作为高维语义特征提取器，提供稳定的"语义锚点"

### 3. 技术细节
- 使用detectron2框架（Mask2Former的官方实现）
- 提取Pixel Decoder输出的特征图
- 特征维度需要与后续融合模块对齐

## 需要解决的问题
1. Mask2Former预训练权重下载（Cityscapes）
2. 项目目录结构设计
3. 依赖包管理（requirements.txt）
4. Mask2Former模型封装（冻结参数）
5. 特征提取接口设计
6. 配置文件管理

# 提议的解决方案

## 方案选择（已确认）
1. **框架选择**：Detectron2（官方实现）
2. **特征维度**：256维
3. **Backbone选择**：Swin-Large
4. **权重存储**：项目目录下的特定文件夹（建议 `checkpoints/mask2former/`）

## 方案一：Detectron2实现（已选定）

### 核心优势
- 官方支持，与论文实现一致
- 可直接访问Pixel Decoder中间特征
- Cityscapes预训练权重可直接使用
- 社区资源丰富，问题易解决

### 实现要点
1. **模型加载**：使用detectron2的配置系统和权重加载机制
2. **特征提取**：通过forward hook或修改模型结构，在Pixel Decoder输出处截取特征
3. **维度对齐**：确保Pixel Decoder输出特征维度为256维（可能需要添加1x1卷积进行维度调整）
4. **冻结机制**：在初始化时设置所有参数的`requires_grad=False`

### 架构设计
- **模块封装**：`Semantic2DBranch`类封装Mask2Former，提供统一接口
- **配置管理**：使用YAML配置文件管理模型参数
- **权重管理**：预训练权重存储在`checkpoints/mask2former/`目录
- **特征接口**：提供`forward()`方法，输入图像，输出256维特征图

### 需要实现的核心组件
1. `src/models/semantic_2d.py`：Semantic2DBranch类实现
2. `configs/mask2former_swin_l_cityscapes.yaml`：Detectron2配置文件
3. `scripts/download_mask2former_weights.py`：权重下载脚本
4. `src/utils/image_preprocessing.py`：图像预处理工具
5. `tests/test_semantic_2d.py`：单元测试脚本

# 当前执行步骤："3. 执行阶段 - 已完成所有文件创建"

# 任务进度
[2025-01-14_17:04:03]
- 已创建：功能分支 task/implement-stage1-mask2former_2025-01-14_1
- 已创建：任务文件 .tasks/2025-01-14_1_implement-stage1-mask2former.md
- 状态：研究阶段 - 分析技术方案和项目需求

[2025-01-14_17:10:00]
- 已完成：创新阶段 - 方案讨论和选择
- 决策确认：
  * 框架：Detectron2
  * 特征维度：256维
  * Backbone：Swin-Large
  * 权重存储：checkpoints/mask2former/
- 状态：等待进入PLAN模式

[2025-01-14_17:20:00]
- 已完成：规划阶段 - 制定详细实施计划
- 状态：进入执行阶段

[2025-01-14_17:25:00]
- 已完成：执行阶段 - 创建所有核心文件
- 已创建文件：
  * requirements.txt - 依赖包清单
  * configs/mask2former_swin_l_cityscapes.yaml - Detectron2配置文件
  * src/models/semantic_2d.py - Semantic2DBranch类实现
  * src/utils/image_preprocessing.py - 图像预处理工具
  * scripts/download_mask2former_weights.py - 权重下载脚本
  * tests/test_semantic_2d.py - 单元测试脚本
  * README.md - 项目说明文档
- 已创建目录结构：checkpoints/, configs/, src/models/, src/utils/, scripts/, tests/
- 状态：等待用户确认和测试

[2025-01-14_17:30:00]
- 已更新：权重下载链接
- 权重URL：https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_064788.pkl
- 权重文件名：model_final_064788.pkl
- 已更新文件：
  * scripts/download_mask2former_weights.py - 使用正确的下载URL
  * configs/mask2former_swin_l_cityscapes.yaml - 更新权重路径
  * src/models/semantic_2d.py - 更新错误提示信息
  * tests/test_semantic_2d.py - 更新测试中的权重路径
  * README.md - 更新文档中的权重路径
- 状态：所有文件已更新，可以使用正确的权重链接下载

[2025-01-14_19:30:00]
- 已修复：Mask2Former导入和配置问题
- 问题1：Mask2Former未安装 - 已修复，添加了本地路径导入支持
- 问题2：配置文件缺少基础配置 - 已修复，移除了_BASE_引用
- 问题3：backbone名称错误 - 已修复，使用"D2SwinTransformer"
- 问题4：配置项不完整 - 已修复，添加了完整的SWIN和SEM_SEG_HEAD配置
- 问题5：NUM_OBJECT_QUERIES不匹配 - 已修复，从100改为200以匹配权重文件
- 测试结果：5/5 测试通过 ✓
  * 图像预处理 ✓
  * 模型初始化 ✓
  * 前向传播 ✓
  * 特征维度验证 ✓ (256维)
  * 参数冻结验证 ✓ (625个参数全部冻结)
- 状态：阶段一实现完成，所有功能正常

# 最终审查
（待完成）

