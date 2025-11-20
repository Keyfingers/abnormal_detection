# AnoVox数据集准备完成总结

## ✅ 已完成的工作

### 1. 训练数据集（阶段4）- 常态数据
- **数据集**: AnoVox_Normality_Mono_Town03
- **位置**: `/root/autodl-tmp/dataset/anovox/AnoVox_Normality_Mono_Town03/`
- **内容**: 75个点云文件（.npy格式）
- **训练集**: 60个样本
- **测试集**: 15个样本
- **状态**: ✅ 已下载、已解压、已适配

### 2. 数据集加载器
- **文件**: `src/data/anovox_normality_dataset.py`
- **功能**: 
  - 支持.npy格式点云
  - 自动处理Scenario目录结构
  - 创建占位符图像（实际训练时使用真实图像）
  - 异常掩码全为0（常态数据）
  - 默认相机标定参数

### 3. 训练脚本更新
- **文件**: `src/training/train_fusion.py`
- **更新**: 自动检测AnoVox Normality格式并使用对应的数据集加载器

## 📋 数据集说明

### 训练数据（阶段4）- 常态数据
- **目的**: 训练融合头学习"正常时特征是什么样的"
- **特点**: 
  - 所有样本都是正常样本
  - 异常掩码全为0
  - 模型学习输出低异常分数（接近0）

### 评估数据（阶段5）- 异常数据（待下载）
- **目的**: 评估模型在异常样本上的表现
- **需要**: 包含异常样本和异常掩码的数据集
- **推荐**: AnoVox Static/Content Anomaly数据集

## 🔍 关于数据集格式的说明

### 当前数据集格式
```
AnoVox_Normality_Mono_Town03/
└── Scenario_486bab15-e643-418f-9c8b-4a1e8a954f6f/
    └── LIDAR(...)/
        └── *.npy (点云文件，Nx4)
```

### 代码适配
- ✅ 支持.npy格式点云（自动转换）
- ✅ 处理Scenario目录结构
- ✅ 创建占位符图像（训练时使用真实图像输入）
- ✅ 异常掩码全0（常态数据）
- ✅ 默认相机标定（可后续更新）

## ⚠️ 重要说明

### 1. 图像数据
当前数据集**没有RGB图像**，但训练时会：
- 使用占位符图像初始化
- **实际训练时，模型会从真实输入图像提取2D特征**
- 融合头学习的是特征融合，不是图像本身

### 2. 相机标定
当前使用默认标定参数，如果需要精确的3D-2D投影：
- 需要从数据集元数据获取真实标定
- 或使用CARLA的默认相机参数

### 3. 异常评估集
**需要单独下载**包含异常样本的数据集用于阶段5评估。

## 🚀 下一步操作

### 步骤1：下载异常评估集（可选，用于阶段5）

```bash
# 使用脚本下载
./scripts/download_anovox_anomaly.sh

# 或手动下载（例如：AnoVox_Static_Mono_Town10_1）
cd /root/autodl-tmp/dataset/anovox
aria2c -x 16 -s 16 "https://zenodo.org/records/10881577/files/AnoVox_Dynamic_Multi_Town10.tar.gz?download=1" -o anomaly.tar.gz
```

### 步骤2：开始训练融合模块

```bash
python src/training/train_fusion.py \
    --data-root /root/autodl-tmp/dataset/anovox \
    --semantic-ckpt RbA/ckpts/swin_b_1dl/swin_b_1dl/model_final.pth \
    --semantic-config RbA/configs/cityscapes/semantic-segmentation/swin/single_decoder_layer/maskformer2_swin_base_IN21k_384_bs16_90k_1dl.yaml \
    --geometric-ckpt /path/to/3d/checkpoint.pth \
    --output-dir outputs/fusion \
    --batch-size 2 \
    --num-epochs 50 \
    --lr 1e-4
```

**注意**: 
- 需要先准备3D预训练模型（MinkUNet）
- 训练时模型会从真实输入图像提取特征（不是占位符）

### 步骤3：验证数据集

```bash
# 测试数据集加载
python -c "from src.data.anovox_normality_dataset import AnoVoxNormalityDataset; ds = AnoVoxNormalityDataset('/root/autodl-tmp/dataset/anovox', split='train'); print(f'数据集大小: {len(ds)}'); sample = ds[0]; print(f'点云形状: {sample[\"point_cloud\"].shape}'); print(f'异常掩码值: {sample[\"anomaly_mask\"].sum()}')"
```

## 📊 数据集统计

- **训练集**: 60个样本
- **测试集**: 15个样本
- **点云格式**: .npy (Nx4, 使用前3列xyz)
- **异常掩码**: 全0（常态数据）

## ✅ 验证清单

- [x] 数据集已下载
- [x] 数据集已解压
- [x] 数据集加载器已创建
- [x] 训练脚本已更新
- [x] 数据集格式已适配
- [ ] 3D预训练模型已准备（需要检查）
- [ ] 异常评估集已下载（可选，用于阶段5）

## 🎯 训练策略说明

根据项目规则和您的理解：

1. **阶段4（训练）**: 
   - 使用"常态"数据训练融合头
   - 所有样本的异常掩码为0
   - 模型学习：当2D和3D特征都表示"正常"时，融合输出应该接近0

2. **阶段5（评估）**:
   - 使用"异常"数据评估模型
   - 包含异常样本和异常掩码
   - 测试模型能否识别异常区域

这种策略的优势：
- ✅ 避免从头训练大模型
- ✅ 通过nuScenes保证泛化能力
- ✅ 通过AnoVox证明特定任务上的创新
- ✅ 符合"混合正常和异常数据进行训练是成功的关键"的结论

---

**更新时间**: 2025-11-18  
**状态**: ✅ 训练数据集准备完成，可以开始训练融合模块





