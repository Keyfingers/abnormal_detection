# 融合图像与点云的道路异常检测

本项目实现了基于冻结Mask2Former和MinkUNet的多模态道路异常检测系统。

## 项目概述

本项目采用五阶段架构：
1. **阶段一**：冻结的Mask2Former（2D图像分支）✅
2. **阶段二**：冻结的MinkUNet（3D点云分支）✅
3. **阶段三**：Feature Splatting投影（核心创新）
4. **阶段四**：轻量级融合头
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

### 4. 运行测试

```bash
# 测试阶段一
python tests/test_semantic_2d.py

# 测试阶段二
python tests/test_geometric_3d.py
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
│   │   └── geometric_3d.py   # Geometric3DBranch
│   ├── utils/                # 工具函数
│   │   ├── image_preprocessing.py
│   │   └── pointcloud_preprocessing.py
│   └── training/             # 训练脚本
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
- [ ] 阶段三：Feature Splatting投影
- [ ] 阶段四：轻量级融合头
- [ ] 阶段五：训练与评估

## 许可证

[添加许可证信息]

## 致谢

- [Mask2Former](https://github.com/facebookresearch/Mask2Former) - Facebook Research
- [Detectron2](https://github.com/facebookresearch/detectron2) - Facebook Research
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) - OpenMMLab
- [AnoVox Dataset](https://github.com/AnoVox) - 异常检测基准数据集

