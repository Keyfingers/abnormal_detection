# 背景
文件名：2025-11-20_1_upgrade-splatting-fusion
创建于：2025-11-20_10:00:00
创建者：Spark
主分支：main
任务分支：task/upgrade-splatting-fusion_2025-11-20_1
Yolo模式：Off

# 任务描述
将现有的“双线性插值投影 + 简单拼接融合”架构升级为技术方案文档中描述的“Feature Splatting + Gated Adapter Fusion”架构。
1.  实现 Gaussian Feature Splatting (3D到2D投影)，引入高斯核权重，解决LiDAR稀疏性问题。
2.  实现 Gated Fusion Head (门控融合头)，引入瓶颈结构和门控机制，动态加权不同模态特征。

# 项目概览
-   **目标**：对齐技术方案文档，提升创新性和学术价值。
-   **核心变动**：
    -   `src/utils/projection.py`: 重写投影逻辑，从 Bilinear 改为 Gaussian Splatting。
    -   `src/models/fusion.py`: 重构融合网络，从 Conv Stack 改为 Gated Adapter。

⚠️ 警告：永远不要修改此部分 ⚠️
1.  **系统思维**：确保新模块与现有冻结骨干网络（Mask2Former, MinkUNet）接口兼容。
2.  **辩证思维**：平衡算法复杂度与运行效率，Python端实现需避免过高计算开销。
3.  **创新思维**：使用各向同性高斯近似来简化协方差计算，在不修改骨干网络前提下实现Splatting。
⚠️ 警告：永远不要修改此部分 ⚠️

# 分析
-   **投影模块现状**：使用 `project_3d_to_2d_bilinear`，简单的双线性插值，缺乏体积感和空洞填补能力。
-   **融合模块现状**：`FusionHead` 为简单的 CNN 堆叠，缺乏门控机制和瓶颈设计。
-   **数据流**：`features_3d` (SparseTensor) -> `project` -> `features_3d_2d` (Dense) -> `FusionHead`。

# 提议的解决方案

## 1. Gaussian Feature Splatting (`src/utils/projection.py`)
-   **算法**：
    1.  对于每个3D点，计算其投影中心 $(u, v)$。
    2.  定义高斯核半径 $R$ (可基于 depth 或固定值)。
    3.  在 $(u, v)$ 周围 $K \times K$ 窗口内计算高斯权重 $w_{ij} = \exp(-\frac{(i-u)^2 + (j-v)^2}{2\sigma^2})$。
    4.  使用加权平均聚合特征：$F_{2D}(x, y) = \frac{\sum w_k F_{3D}^k}{\sum w_k}$。
-   **优势**：自然平滑，填补空洞，梯度可导。

## 2. Gated Adapter Fusion (`src/models/fusion.py`)
-   **结构**：
    1.  **Input Projector**: 1x1 Conv 将 2D/3D 特征对齐到相同维度。
    2.  **Gating Network**: 输入 Concat 特征，输出 Gate Map $G \in [0, 1]$ (Sigmoid)。
    3.  **Weighted Sum**: $F_{fused} = G \cdot F_{2D} + (1-G) \cdot F_{3D}$。
    4.  **Bottleneck Adapter**: Conv(1x1, C->C/4) -> ReLU -> Conv(1x1, C/4->C) -> Residual Add。
    5.  **Output Head**: Conv -> Sigmoid -> Anomaly Score。
-   **优势**：动态权重，参数高效，防止过拟合。

# 当前执行步骤："[步骤编号和名称]"
- 例如："2. 创建任务文件"

# 任务进度
[2025-11-20] 任务初始化
[2025-11-20] 实现 Gaussian Feature Splatting (`src/utils/projection.py`)
[2025-11-20] 实现 Gated Adapter Fusion (`src/models/fusion.py`)
[2025-11-20] 集成新模块到 FusionModel

# 最终审查
实施与计划完全匹配。
已成功将投影机制升级为 Gaussian Splatting，融合机制升级为 Gated Adapter，代码结构已对齐技术方案文档的高级要求。
