# 融合图像与点云的道路异常检测

本项目实现了基于冻结Mask2Former和MinkUNet的多模态道路异常检测系统。

## 项目概述

本项目采用五阶段架构：
1. **阶段一**：冻结的Mask2Former（2D图像分支）✅
2. **阶段二**：冻结的MinkUNet（3D点云分支）✅
3. **阶段三**：Feature Splatting投影（核心创新）✅
4. **阶段四**：轻量级融合头 ✅
5. **阶段五**：训练与评估

## 核心特性

- **冻结骨干网络范式**：保留预训练模型的通用表征能力
- **Feature Splatting特征对齐**：可微的3D到2D特征投影
- **轻量级适配器设计**：参数高效微调（PEFT）

## 环境配置

### 系统要求

- Python >= 3.8
- CUDA >= 11.1 (推荐) 或 CPU
- PyTorch >= 1.10.0

### 安装步骤

1. **克隆项目**
```bash
git clone <repository_url>
cd abnormal_detection
```

2. **创建虚拟环境（推荐）**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装PyTorch**
```bash
# 根据您的CUDA版本选择
# CUDA 11.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113

# CUDA 11.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu111

# CPU only
pip install torch torchvision
```

4. **安装其他依赖**
```bash
pip install -r requirements.txt
```

5. **安装Detectron2**
```bash
# CUDA 11.3
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# CUDA 11.1
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# CPU only
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

更多安装选项请参考：[Detectron2安装指南](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)

6. **安装Mask2Former**
```bash
# 方法1: 直接从GitHub安装（推荐）
pip install git+https://github.com/facebookresearch/Mask2Former.git

# 方法2: 克隆后安装
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former
pip install -e .
```

**重要**: Mask2Former需要单独安装，它不在detectron2的核心包中。

7. **安装MMDetection3D**
```bash
# Step 1: 安装mmcv-full（MMDetection3D的依赖）
# CUDA 11.3
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# CUDA 11.1
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

# CPU only
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html

# Step 2: 安装MMDetection3D
pip install mmdet3d

# 或从源码安装
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && pip install -v -e .
```

更多安装选项请参考：[MMDetection3D安装指南](https://mmdetection3d.readthedocs.io/en/latest/get_started.html)

**注意**: MMDetection3D需要先安装mmcv-full，请根据您的CUDA版本选择正确的wheel。

## 快速开始

### 1. 下载预训练权重

**阶段一：Mask2Former权重**
```bash
python scripts/download_mask2former_weights.py
```

权重文件将下载到 `checkpoints/mask2former/model_final_064788.pkl`

**注意**：如果自动下载失败，请手动下载：
1. 访问 [Mask2Former GitHub](https://github.com/facebookresearch/Mask2Former)
2. 查看 Model Zoo 或 Releases 页面
3. 下载 `mask2former_swin_large_IN21k_384_bs16_50ep_800k_cityscapes` 预训练权重
4. 保存到 `checkpoints/mask2former/model_final_064788.pkl`

**阶段二：MMDetection3D权重**
```bash
# 下载MinkUNet SemanticKITTI预训练权重（默认）
python scripts/download_mmdet3d_weights.py

# 或指定模型名称
python scripts/download_mmdet3d_weights.py --model minkunet

# 或使用自定义URL下载
python scripts/download_mmdet3d_weights.py --url <weight_url>

# 创建占位权重文件（仅用于测试，不推荐）
python scripts/download_mmdet3d_weights.py --create-placeholder
```

权重文件将保存到 `checkpoints/mmdet3d/`

**默认权重**：MinkUNet SemanticKITTI预训练权重
- URL: https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w32_8xb2-15e_semantickitti/minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth
- 模型：MinkUNet w32
- 数据集：SemanticKITTI
- 训练配置：8xb2-15e (8 GPUs, batch size 2, 15 epochs)

**注意**：MMDetection3D提供了SemanticKITTI预训练权重，可以从[MMDetection3D Model Zoo](https://github.com/open-mmlab/mmdetection3d)获取更多模型权重。

### 2. 使用Semantic2DBranch

```python
import torch
from src.models.semantic_2d import Semantic2DBranch
from src.utils.image_preprocessing import preprocess_image, load_image

# 初始化模型
model = Semantic2DBranch(
    config_path="configs/mask2former_swin_l_cityscapes.yaml",
    checkpoint_path="checkpoints/mask2former/model_final_064788.pkl",
    freeze_backbone=True,
    feature_dim=256,
    device="cuda"  # 或 "cpu"
)

# 加载和预处理图像
image = load_image("path/to/image.jpg")
image_tensor = preprocess_image(image)

# 提取特征
with torch.no_grad():
    features = model(image_tensor)  # (1, 256, H', W')

print(f"特征形状: {features.shape}")
```

### 3. 使用Geometric3DBranch

```python
import numpy as np
import torch
from src.models.geometric_3d import Geometric3DBranch

# 初始化模型
model = Geometric3DBranch(
    checkpoint_path="checkpoints/mmdet3d/mmdet3d_placeholder.pth",
    config_path=None,  # 可选：MMDetection3D配置文件路径
    freeze_backbone=True,
    feature_dim=128,
    voxel_size=0.05,  # 5cm体素
    device="cuda"  # 或 "cpu"
)

# 创建测试点云（N, 3）或（N, 4）
points = np.random.rand(1000, 3).astype(np.float32) * 10.0

# 提取特征
with torch.no_grad():
    output = model(points)

print(f"体素特征形状: {output['voxel_features'].shape}")
print(f"体素坐标形状: {output['voxel_coords'].shape}")
```

### 4. 使用Feature Splatting（阶段三）

```python
import torch
import numpy as np
from src.models.geometric_3d import Geometric3DBranch
from src.models.feature_splatting import FeatureSplatting
from src.utils.camera_calibration import create_default_projection_matrix, projection_matrix_to_torch

# 初始化3D分支
geometric_branch = Geometric3DBranch(
    checkpoint_path="checkpoints/mmdet3d/mmdet3d_placeholder.pth",
    freeze_backbone=True,
    feature_dim=128,
    voxel_size=0.05,
    device="cuda"
)

# 初始化Feature Splatting
feature_splatting = FeatureSplatting(
    feature_dim=128,
    image_height=800,
    image_width=1333,
    voxel_size=0.05,
    device="cuda"
)

# 准备点云数据
points = np.random.rand(1000, 3).astype(np.float32) * 10.0

# 提取3D体素特征
with torch.no_grad():
    output_3d = geometric_branch(points)
    voxel_features = output_3d['voxel_features']  # (M, 128)
    voxel_coords = output_3d['voxel_coords']  # (M, 3)

# 创建投影矩阵（需要根据实际相机标定调整）
projection_matrix = create_default_projection_matrix(
    image_width=1333,
    image_height=800
)
projection_tensor = projection_matrix_to_torch(projection_matrix, device="cuda")

# 执行Feature Splatting投影
with torch.no_grad():
    feature_map_2d = feature_splatting(
        voxel_features,
        voxel_coords,
        projection_tensor
    )  # (H, W, 128)

print(f"2D特征图形状: {feature_map_2d.shape}")
```

### 5. 使用Fusion Head（阶段四）

```python
import torch
import numpy as np
from src.models.fusion_head import FusionHead
from src.losses.anomaly_loss import AnomalyDetectionLoss
from src.utils.pseudo_anomaly import generate_pseudo_anomalies

# 初始化融合头
fusion_head = FusionHead(
    img_feature_dim=256,
    pts_feature_dim=128,
    hidden_dim=64,
    use_gating=True,
    device="cuda"
)

# 准备特征（来自阶段一、二、三）
img_features = torch.randn(1, 256, 800, 1333).cuda()  # 2D特征
pts_features = torch.randn(1, 128, 800, 1333).cuda()   # 3D投影特征

# 训练时：生成伪异常
img_corrupted, pts_corrupted, anomaly_mask = generate_pseudo_anomalies(
    img_features, pts_features, anomaly_prob=0.5
)

# 融合和判定
anomaly_map = fusion_head(img_corrupted, pts_corrupted)  # (1, 1, 800, 1333)

# 计算损失
loss_func = AnomalyDetectionLoss()
loss_dict = loss_func(anomaly_map, anomaly_mask)
print(f"总损失: {loss_dict['loss']:.4f}")
```

### 6. 使用端到端模型

```python
from src.models.anomaly_detector import AnomalyDetector
from src.utils.camera_calibration import create_default_projection_matrix, projection_matrix_to_torch

# 初始化端到端模型
model = AnomalyDetector(
    mask2former_config_path="configs/mask2former_swin_l_cityscapes.yaml",
    mask2former_checkpoint_path="checkpoints/mask2former/model_final_064788.pkl",
    minkunet_checkpoint_path="checkpoints/mmdet3d/mmdet3d_placeholder.pth",
    device="cuda"
)

# 准备数据
images = torch.randn(1, 3, 800, 1333).cuda()
points = np.random.rand(1000, 3).astype(np.float32) * 10.0
projection_matrix = projection_matrix_to_torch(
    create_default_projection_matrix(1333, 800), "cuda"
)

# 前向传播
output = model(images, points, projection_matrix)
anomaly_map = output['anomaly_map']  # (1, 1, H, W)
print(f"异常概率图形状: {anomaly_map.shape}")
```

### 7. 使用AnoVox数据集DataLoader

```python
from torchvision import transforms
from torch.utils.data import DataLoader
from src.datasets.anovox_dataset import AnoVoxDataset

# 图像预处理（适配Mask2Former）
transform = transforms.Compose([
    transforms.Resize((800, 1333)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集
dataset = AnoVoxDataset(
    root_dir="/path/to/AnoVox_Normality_Mono_Town03",
    transform=transform
)

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=AnoVoxDataset.collate_fn  # 关键！处理点云列表
)

# 使用DataLoader
for batch in dataloader:
    images = batch['img']  # (B, C, H, W)
    points = batch['points']  # List[Tensor]，每个元素是一个点云
    projection_matrices = batch['projection_matrix']  # (B, 3, 4)
    # ... 训练代码
```

### 8. 开始训练

```bash
# 使用命令行参数
python src/training/train_anomaly_detector.py \
    --data_root /path/to/AnoVox_Normality_Mono_Town03 \
    --batch_size 2 \
    --num_epochs 50 \
    --learning_rate 1e-3 \
    --device cuda

# 或使用Python脚本
python -c "
from src.training.train_anomaly_detector import train
from src.models.anomaly_detector import AnomalyDetector
from torchvision import transforms
from torch.utils.data import DataLoader
from src.datasets.anovox_dataset import AnoVoxDataset

# 创建数据集和DataLoader
transform = transforms.Compose([
    transforms.Resize((800, 1333)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = AnoVoxDataset(root_dir='/path/to/AnoVox_Normality_Mono_Town03', transform=transform)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=AnoVoxDataset.collate_fn)

# 初始化模型
model = AnomalyDetector(
    mask2former_config_path='configs/mask2former_swin_l_cityscapes.yaml',
    mask2former_checkpoint_path='checkpoints/mask2former/model_final_064788.pkl',
    minkunet_checkpoint_path='checkpoints/mmdet3d/mmdet3d_placeholder.pth',
    device='cuda'
)

# 开始训练
train(model, train_loader, val_loader=None, num_epochs=50)
"
```

### 9. 运行测试

```bash
# 测试阶段一
python tests/test_semantic_2d.py

# 测试阶段二
python tests/test_geometric_3d.py

# 测试阶段三
python tests/test_feature_splatting.py

# 测试阶段四
python tests/test_fusion_head.py

# 测试AnoVox数据集加载器
export ANOVOX_DATA_ROOT=/path/to/AnoVox_Normality_Mono_Town03
python tests/test_anovox_dataset.py

# 集成测试（阶段一+二+三）
python tests/test_integration_stage3.py
```

## 项目结构

```
abnormal_detection/
├── checkpoints/              # 预训练模型权重
│   ├── mask2former/
│   └── mmdet3d/
├── configs/                  # 配置文件
│   ├── mask2former_swin_l_cityscapes.yaml
│   └── mmdet3d_semantickitti.py
├── src/                      # 源代码
│   ├── models/               # 模型定义
│   │   ├── semantic_2d.py    # Semantic2DBranch
│   │   ├── geometric_3d.py   # Geometric3DBranch
│   │   ├── feature_splatting.py  # Feature Splatting
│   │   ├── fusion_head.py    # Fusion Head
│   │   └── anomaly_detector.py  # 端到端模型
│   ├── datasets/             # 数据集加载器
│   │   └── anovox_dataset.py # AnoVox数据集
│   ├── losses/               # 损失函数
│   │   └── anomaly_loss.py  # Focal Loss + Dice Loss
│   ├── utils/                # 工具函数
│   │   ├── image_preprocessing.py
│   │   ├── pointcloud_preprocessing.py
│   │   ├── camera_calibration.py
│   │   └── pseudo_anomaly.py
│   └── training/             # 训练脚本
│       └── train_anomaly_detector.py
├── scripts/                  # 工具脚本
│   ├── download_mask2former_weights.py
│   └── download_mmdet3d_weights.py
├── tests/                    # 测试代码
│   ├── test_semantic_2d.py
│   └── test_geometric_3d.py
├── requirements.txt          # 依赖包清单
└── README.md                # 项目说明
```

## 配置说明

### Mask2Former配置

配置文件位于 `configs/mask2former_swin_l_cityscapes.yaml`，主要参数：

- **Backbone**: Swin-Large Transformer
- **特征维度**: 256维
- **输入尺寸**: 最小800px，最大1333px
- **类别数**: 19（Cityscapes语义分割类别）

## 常见问题

### Q1: Detectron2安装失败

**A**: 请确保：
1. PyTorch已正确安装
2. CUDA版本匹配（如果使用GPU）
3. 使用正确的wheel URL（根据CUDA版本）

### Q1.5: Mask2Former未安装错误

**A**: 如果遇到 `"No object named 'MaskFormer' found in 'META_ARCH' registry!"` 错误：
1. 确保已安装Mask2Former: `pip install git+https://github.com/facebookresearch/Mask2Former.git`
2. 如果已安装，尝试重新导入: `import mask2former`
3. 检查Python路径中是否包含Mask2Former目录

### Q2: 权重文件下载失败

**A**: 
1. 检查网络连接
2. 尝试手动下载（见快速开始部分）
3. 检查磁盘空间是否充足

### Q3: CUDA内存不足

**A**: 
1. 减小batch size
2. 使用CPU模式（device="cpu"）
3. 减小输入图像尺寸

### Q4: 特征维度不匹配

**A**: 
- 模型会自动添加维度适配器
- 确保 `feature_dim` 参数设置为256（2D）或128（3D）

### Q5: MMDetection3D安装失败

**A**: 
1. 确保先安装mmcv-full（MMDetection3D的依赖）
2. 根据CUDA版本选择正确的mmcv-full wheel
3. 如果安装失败，尝试从源码安装
4. 参考[MMDetection3D安装指南](https://mmdetection3d.readthedocs.io/en/latest/get_started.html)

### Q6: MMDetection3D预训练权重下载失败

**A**: 
1. 默认使用MinkUNet SemanticKITTI预训练权重，URL已内置在脚本中
2. 如果自动下载失败，可以手动下载：
   ```bash
   wget https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w32_8xb2-15e_semantickitti/minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth -O checkpoints/mmdet3d/minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth
   ```
3. 更多模型权重可以从[MMDetection3D Model Zoo](https://github.com/open-mmlab/mmdetection3d)获取
4. 查看configs/目录下的模型配置文件，其中的checkpoint字段包含权重URL
5. 可以使用占位权重文件进行测试：`python scripts/download_mmdet3d_weights.py --create-placeholder`（不推荐，仅用于代码测试）

## 开发计划

- [x] 阶段一：冻结的Mask2Former实现
- [x] 阶段二：冻结的MinkUNet实现
- [x] 阶段三：Feature Splatting投影
- [x] 阶段四：轻量级融合头
- [x] 阶段五：AnoVox数据集DataLoader ✅
- [ ] 阶段五：训练与评估（数据集已就绪，可以开始训练）

## 阶段四：轻量级融合头技术细节

### 核心设计

融合头采用"适配器+门控+判定"的三层架构：

1. **适配器（Adapter）**：
   - 维度对齐：将2D（256维）和3D（128维）特征投影到统一维度（64维）
   - 特征交互：使用1x1卷积或交叉注意力机制
   - 参数量：<5%总参数量（符合PEFT原则）

2. **门控机制（Gating）**：
   - 学习动态权重，决定在每个像素点更信任图像还是点云
   - 例如：在黑暗隧道中，自动提高点云特征的权重

3. **异常判定（Decision）**：
   - 基于"语义-几何不一致性"输出异常概率图
   - 输出：Sigmoid激活的异常概率图 (H, W, 1)

### 训练策略：自监督合成异常

**问题**：训练数据只有Normal数据，直接使用Focal Loss会导致模型崩溃（永远输出0）

**解决方案**：在线生成伪异常（Self-Supervised Synthetic Anomaly）

- **训练阶段**：仅使用Normal数据，在线生成伪异常
  - 随机在特征图上注入噪声（高斯噪声或特征打乱）
  - 生成异常掩码（标签：0=正常，1=异常）
  - Focal Loss现在可以正常工作

- **验证阶段**：使用真实的AnoVox异常数据
  - 计算FPR95、AUPR等指标
  - 验证模型对真实异常的检测能力

### 损失函数

组合损失：`Loss = Focal Loss + Dice Loss`

- **Focal Loss**：解决正负样本不平衡（异常区域通常很小）
- **Dice Loss**：优化分割边界（异常检测本质上是分割任务）

### 使用示例

参考"快速开始"部分的"使用Fusion Head（阶段四）"和"使用端到端模型"章节。

## 阶段五：AnoVox数据集DataLoader技术细节

### 数据集结构

AnoVox数据集采用以下目录结构：

```
AnoVox_Normality_Mono_Town03/
├── Scenario_000/
│   ├── RGB-CAM(0, 0, 1.8)(0, 0, 0)/
│   │   └── *.png  (RGB图像)
│   ├── LIDAR(0, 0, 1.8)(0, 0, 0)/
│   │   └── *.npy  (LiDAR点云)
│   └── sensor_setup.json  (传感器配置)
└── Scenario_001/
    └── ...
```

### 核心功能

1. **自动配对**：自动匹配RGB图像和LiDAR点云（基于文件名）
2. **投影矩阵计算**：
   - 从`sensor_setup.json`解析相机内参
   - 从文件夹名解析传感器外参（位置和旋转）
   - 自动构建3D到2D投影矩阵
3. **点云处理**：支持`.npy`格式点云，自动处理不同维度（x,y,z或x,y,z,intensity）
4. **批次处理**：自定义`collate_fn`处理点云列表（MinkowskiEngine要求）

### 传感器参数解析

文件夹名格式：`RGB-CAM(x, y, z)(roll, pitch, yaw)`
- 第一个括号：传感器位置（米）
- 第二个括号：传感器旋转（度）

如果RGB和LiDAR位置相同，外参矩阵简化为单位矩阵（坐标轴对齐）。

### 使用示例

参考"快速开始"部分的"使用AnoVox数据集DataLoader"章节。

## 阶段三：Feature Splatting技术细节

### 核心创新

Feature Splatting是本项目的核心创新模块，实现了3D体素特征到2D特征图的可微投影：

1. **高斯建模**：将每个体素建模为3D高斯分布（椭球）
2. **投影变换**：通过相机投影矩阵将3D高斯投影到2D平面
3. **特征聚合**：使用2D高斯权重进行特征聚合，生成稠密的2D特征图
4. **可微性**：整个过程对位置和协方差可导，支持端到端训练

### 技术特点

- **稀疏到稠密**：解决点云稀疏性问题，生成连续的特征图
- **可微投影**：支持反向传播，可以学习优化投影参数
- **几何感知**：通过协方差矩阵建模空间不确定性
- **高效实现**：使用GPU加速的光栅化过程

### 使用示例

参考"快速开始"部分的"使用Feature Splatting（阶段三）"章节。

## 许可证

[添加许可证信息]

## 致谢

- [Mask2Former](https://github.com/facebookresearch/Mask2Former) - Facebook Research
- [Detectron2](https://github.com/facebookresearch/detectron2) - Facebook Research
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) - OpenMMLab
- [AnoVox Dataset](https://github.com/AnoVox) - 异常检测基准数据集

