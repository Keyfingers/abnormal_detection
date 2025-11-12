# 融合图像和点云的道路异常检测

这是一个基于Mask2Former和MinkUNet的多模态融合道路异常检测项目，用于自动驾驶场景。

## 项目结构

```
abnormal_detection/
├── RbA/                          # Mask2Former + RbA评分（2D语义分支）
├── MinkowskiEngine-master/       # Minkowski Engine库（3D稀疏卷积）
├── src/
│   ├── models/
│   │   ├── semantic_2d.py        # 2D语义分支模型封装
│   │   ├── geometric_3d.py       # 3D几何分支模型（MinkUNet）
│   │   └── fusion.py             # 融合模块
│   ├── data/
│   │   ├── nuscenes_dataset.py   # nuScenes数据集加载器
│   │   └── anovox_dataset.py      # AnoVox数据集加载器
│   ├── utils/
│   │   ├── projection.py         # 3D-2D投影工具
│   │   └── metrics.py            # 评估指标
│   └── training/
│       ├── train_2d_branch.py    # 训练2D语义分支
│       ├── train_3d_branch.py    # 训练3D几何分支
│       └── train_fusion.py       # 训练融合模块
├── scripts/
│   ├── evaluate_baselines.py     # 评估基线模型
│   └── evaluate_fusion.py        # 评估融合模型
├── configs/                      # 配置文件
├── requirements.txt              # Python依赖
└── README.md                     # 项目说明

```

## 安装

### 1. 环境要求

- Python >= 3.8
- CUDA >= 10.2
- PyTorch >= 1.9.0
- GCC >= 7.4.0

### 2. 安装依赖

```bash
# 安装PyTorch（根据CUDA版本选择）
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 安装Detectron2（用于Mask2Former）
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

## 数据集准备

### nuScenes数据集
1. 下载nuScenes完整数据集（v1.0）
2. 解压到 `data/nuscenes/` 目录
3. 数据集应包含：
   - `samples/` - 图像和点云数据
   - `sweeps/` - 传感器扫描数据
   - `maps/` - 地图数据
   - `v1.0-*/` - 标注文件

### AnoVox数据集
1. 下载AnoVox数据集
2. 解压到 `data/anovox/` 目录
3. 数据集应包含训练集和测试集

## 使用流程

### 阶段一：环境搭建
按照上述安装步骤完成环境配置。

### 阶段二：训练2D语义分支
```bash
python src/training/train_2d_branch.py \
    --config configs/semantic_2d.yaml \
    --data_root data/nuscenes \
    --output_dir outputs/semantic_2d
```

### 阶段三：训练3D几何分支
```bash
python src/training/train_3d_branch.py \
    --config configs/geometric_3d.yaml \
    --data_root data/nuscenes \
    --output_dir outputs/geometric_3d
```

### 阶段四：训练融合模块
```bash
python src/training/train_fusion.py \
    --config configs/fusion.yaml \
    --semantic_ckpt outputs/semantic_2d/model_final.pth \
    --geometric_ckpt outputs/geometric_3d/model_final.pth \
    --data_root data/anovox \
    --output_dir outputs/fusion
```

### 阶段五：评估
```bash
# 评估基线1（2D-only）
python scripts/evaluate_baselines.py \
    --model_type 2d \
    --checkpoint outputs/semantic_2d/model_final.pth \
    --data_root data/anovox/test

# 评估基线2（3D-only）
python scripts/evaluate_baselines.py \
    --model_type 3d \
    --checkpoint outputs/geometric_3d/model_final.pth \
    --data_root data/anovox/test

# 评估融合模型
python scripts/evaluate_fusion.py \
    --checkpoint outputs/fusion/model_final.pth \
    --data_root data/anovox/test
```

## 引用

如果使用本项目，请引用相关论文：

- RbA: Segmenting Unknown Regions Rejected by All (ICCV 2023)
- 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks (CVPR 2019)

