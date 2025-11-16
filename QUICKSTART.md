# 快速开始指南

## 项目概述

本项目实现了融合图像和点云的道路异常检测系统，包含以下五个阶段：

1. **阶段一**：环境搭建
2. **阶段二**：2D语义分支训练（Mask2Former + RbA）
3. **阶段三**：3D几何分支训练（MinkUNet自编码器）
4. **阶段四**：融合模块训练
5. **阶段五**：评估和对比

## 快速开始

### 1. 环境安装

```bash
# 安装PyTorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 安装Detectron2
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# 安装Minkowski Engine
cd MinkowskiEngine-master
pip install -e . --no-deps
cd ..

# 安装其他依赖
pip install -r requirements.txt

# 编译MSDeformAttn CUDA kernel
cd RbA/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../..
```

### 2. 数据准备

#### nuScenes数据集
```bash
# 下载nuScenes数据集（v1.0）
# 解压到 autodl-tmp/dataset/nuscenes/ 目录
# 目录结构应该是：
# autodl-tmp/dataset/nuscenes/
#   ├── samples/
#   ├── sweeps/
#   ├── maps/
#   └── v1.0-*/
```

#### AnoVox数据集
```bash
# 下载AnoVox数据集
# 解压到 autodl-tmp/dataset/anovox/ 目录
# 目录结构应该是：
# autodl-tmp/dataset/anovox/
#   ├── train/
#   │   ├── images/
#   │   ├── pointclouds/
#   │   └── anomaly_masks/
#   └── test/
#       ├── images/
#       ├── pointclouds/
#       └── anomaly_masks/
```

### 3. 训练流程

#### 阶段二：训练2D语义分支

```bash
python src/training/train_2d_branch.py \
    --config-file RbA/configs/cityscapes/semantic-segmentation/swin/single_decoder_layer/maskformer2_swin_base_IN21k_384_bs16_90k_1dl.yaml \
    --data-root data/nuscenes \
    --num-gpus 4 \
    --output-dir outputs/semantic_2d
```

**注意**：2D分支的训练实际上使用RbA项目的训练脚本。上述命令是一个封装，实际执行时会调用RbA的`train_net.py`。

#### 阶段三：训练3D几何分支

```bash
python src/training/train_3d_branch.py \
    --data-root data/nuscenes \
    --output-dir outputs/geometric_3d \
    --batch-size 4 \
    --num-epochs 100 \
    --lr 0.001 \
    --voxel-size 0.05 \
    --feature-dim 128
```

#### 阶段四：训练融合模块

```bash
python src/training/train_fusion.py \
    --data-root data/anovox \
    --semantic-ckpt outputs/semantic_2d/model_final.pth \
    --geometric-ckpt outputs/geometric_3d/model_final.pth \
    --semantic-config RbA/configs/cityscapes/semantic-segmentation/swin/single_decoder_layer/maskformer2_swin_base_IN21k_384_bs16_90k_1dl.yaml \
    --output-dir outputs/fusion \
    --batch-size 4 \
    --num-epochs 50 \
    --lr 0.0001
```

### 4. 评估

#### 评估基线1（2D-only）

```bash
python scripts/evaluate_baselines.py \
    --model-type 2d \
    --checkpoint outputs/semantic_2d/model_final.pth \
    --config RbA/configs/cityscapes/semantic-segmentation/swin/single_decoder_layer/maskformer2_swin_base_IN21k_384_bs16_90k_1dl.yaml \
    --data-root data/anovox/test
```

#### 评估基线2（3D-only）

```bash
python scripts/evaluate_baselines.py \
    --model-type 3d \
    --checkpoint outputs/geometric_3d/model_final.pth \
    --data-root data/anovox/test
```

#### 评估融合模型

```bash
python scripts/evaluate_fusion.py \
    --checkpoint outputs/fusion/model_final.pth \
    --semantic-ckpt outputs/semantic_2d/model_final.pth \
    --geometric-ckpt outputs/geometric_3d/model_final.pth \
    --semantic-config RbA/configs/cityscapes/semantic-segmentation/swin/single_decoder_layer/maskformer2_swin_base_IN21k_384_bs16_90k_1dl.yaml \
    --data-root data/anovox/test
```

## 项目结构说明

```
abnormal_detection/
├── src/
│   ├── models/           # 模型定义
│   │   ├── semantic_2d.py    # 2D语义分支
│   │   ├── geometric_3d.py   # 3D几何分支
│   │   └── fusion.py         # 融合模块
│   ├── data/            # 数据加载器
│   │   ├── nuscenes_dataset.py
│   │   └── anovox_dataset.py
│   ├── utils/           # 工具函数
│   │   ├── projection.py     # 3D-2D投影
│   │   └── metrics.py         # 评估指标
│   └── training/        # 训练脚本
│       ├── train_2d_branch.py
│       ├── train_3d_branch.py
│       └── train_fusion.py
├── scripts/            # 评估脚本
│   ├── evaluate_baselines.py
│   └── evaluate_fusion.py
├── configs/            # 配置文件
├── RbA/               # Mask2Former + RbA（已存在）
├── MinkowskiEngine-master/  # Minkowski Engine（已存在）
└── README.md
```

## 注意事项

1. **数据格式**：数据加载器假设了特定的数据格式。如果您的数据格式不同，需要修改相应的数据加载器。

2. **相机标定**：3D-2D投影需要准确的相机标定参数。请确保数据集中包含正确的内参和外参。

3. **点云格式**：当前实现假设点云是numpy数组格式。如果使用其他格式（如.bin文件），需要修改数据加载器。

4. **内存使用**：点云数据可能占用大量内存。如果遇到内存问题，可以减小batch size或使用数据采样。

5. **GPU要求**：建议使用至少8GB显存的GPU进行训练。

## 常见问题

### Q: 如何修改特征维度？
A: 在训练脚本中使用`--feature-2d-dim`和`--feature-3d-dim`参数。

### Q: 如何调整融合网络的层数？
A: 修改`FusionHead`的`num_layers`参数，或直接在代码中修改。

### Q: 训练时出现CUDA内存不足？
A: 减小batch size，或使用梯度累积。

### Q: 如何可视化结果？
A: 可以在评估脚本中添加可视化代码，保存异常分数图为图像文件。

## 下一步

1. 根据实际数据格式调整数据加载器
2. 调整超参数以获得最佳性能
3. 添加可视化功能
4. 实现更复杂的融合策略（如注意力机制）

