# 2D训练问题修复总结

## 问题诊断

### 发现的问题
1. **loss_ce几乎为0**：分类损失从早期的`3.951e-05`降到`1.823e-06`，现在基本为0或极小值（`1e-7`到`1e-9`）
2. **total_loss停滞在0.09**：主要由`loss_mask`和`loss_dice`组成，已收敛但无意义
3. **根本原因**：占位符标签全为0（背景类），模型学习到将所有像素预测为背景

### 训练日志分析
- 早期训练（iter 619-999）：total_loss从0.1851降到0.09376
- 当前状态（iter 13000+）：total_loss在0.09左右波动
- loss_ce几乎为0，说明模型没有学到有效的语义分割

## 解决方案

### 1. 停止无意义的训练 ✅
- 已停止当前训练进程
- 占位符标签（全0）无法提供有效的学习信号

### 2. 下载预训练模型 🔄
- 正在使用aria2c多线程下载（16个连接）
- 模型：`swin_b_1dl.zip` (336MB)
- 下载位置：`RbA/ckpts/swin_b_1dl/`
- 当前状态：下载中（约1%，预计16分钟）

### 3. 使用预训练模型提取特征（下一步）
根据项目规则，阶段二应该：
- **使用预训练的Mask2Former模型**（在Cityscapes上训练）
- **在nuScenes上提取特征图**，而不是训练
- 提取Pixel Decoder的2D特征图用于融合

## 下一步操作

### 步骤1：等待下载完成
```bash
# 监控下载进度
tail -f /root/autodl-tmp/abnormal_detection/RbA/ckpts/swin_b_1dl/download.log

# 或使用监控脚本
./check_download.sh
```

### 步骤2：验证模型文件
下载完成后，应该有以下文件：
- `RbA/ckpts/swin_b_1dl/model_final.pth` - 预训练模型权重
- `RbA/ckpts/swin_b_1dl/config.yaml` - 配置文件（已存在）

### 步骤3：修改代码使用预训练模型
需要修改：
1. `src/models/semantic_2d.py` - 确保正确加载预训练权重
2. 创建特征提取脚本，而不是训练脚本
3. 在nuScenes数据上提取2D特征图

### 步骤4：提取特征图
使用预训练模型在nuScenes数据上：
- 提取Pixel Decoder的2D特征图（用于融合）
- 计算RbA异常评分（用于基线）

## 关键代码修改点

### 1. 加载预训练模型
```python
# 在Semantic2DBranch.__init__中
checkpoint_path = "RbA/ckpts/swin_b_1dl/model_final.pth"
if checkpoint_path and os.path.exists(checkpoint_path):
    checkpointer = DetectionCheckpointer(self.model)
    checkpointer.load(checkpoint_path)
```

### 2. 提取特征图
```python
# 使用forward_features方法
mask_features, _, multi_scale_features = pixel_decoder.forward_features(features)
# mask_features shape: (B, mask_dim, H, W) - 这就是用于融合的F_2D
```

## 注意事项

1. **不需要训练**：预训练模型已经在Cityscapes上训练好，直接使用即可
2. **特征提取**：重点是提取Pixel Decoder输出的特征图，而不是最终的语义分割结果
3. **RbA评分**：使用预训练模型的logits计算异常评分，作为2D-Only基线

## 当前状态

- ✅ 训练进程已停止
- ✅ 预训练模型下载完成（336MB，已解压）
- ✅ 模型文件位置：`RbA/ckpts/swin_b_1dl/swin_b_1dl/model_final.pth`
- ✅ 特征提取脚本已创建：`scripts/extract_2d_features.py`
- ⏳ 准备运行特征提取

## 特征提取脚本使用说明

### 基本用法
```bash
python scripts/extract_2d_features.py \
    --config-file configs/nuscenes_semantic_2d_pretrained.yaml \
    --checkpoint-path RbA/ckpts/swin_b_1dl/swin_b_1dl/model_final.pth \
    --data-root /path/to/nuscenes \
    --split train \
    --output-dir ./outputs/2d_features/train \
    --batch-size 1 \
    --image-size 1024 2048
```

### 参数说明
- `--config-file`: Mask2Former配置文件路径
- `--checkpoint-path`: 预训练模型权重路径（默认已设置）
- `--data-root`: nuScenes数据根目录
- `--split`: 数据集分割（train/val）
- `--output-dir`: 输出目录
- `--batch-size`: 批次大小（默认1，根据GPU内存调整）
- `--image-size`: 图像尺寸 (H W)
- `--no-features`: 不保存特征图（只保存RbA评分）
- `--no-rba`: 不保存RbA评分（只保存特征图）

### 输出文件结构
```
outputs/2d_features/
├── features_2d/          # 2D特征图（用于融合）
│   ├── {sample_token}.npy
│   └── ...
├── rba_scores/          # RbA异常评分（用于基线）
│   ├── {sample_token}.npy
│   └── ...
└── sample_tokens.txt     # 样本token列表
```

---
**更新时间**: 2025-11-17 20:10
**状态**: 准备运行特征提取




