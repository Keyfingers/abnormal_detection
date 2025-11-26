# 阶段四 Fusion Head 数值稳定性和初始化修正

根据代码审查反馈，进行了以下关键修正：

## 1. Focal Loss数值稳定性修正 ✅

### 问题
`F.binary_cross_entropy` 在 `pred` 接近 0 或 1 时容易产生 NaN，因为 `log(0)` 或 `log(1)` 会导致数值不稳定。

### 修正
在计算Focal Loss前，对预测值进行clamp：
```python
# 修正前
ce_loss = F.binary_cross_entropy(pred, target, reduction='none')

# 修正后
pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
```

**文件**：`src/losses/anomaly_loss.py` - `FocalLoss.forward()`

## 2. Fusion Head初始化修正 ✅

### 问题
使用 `kaiming_normal_` 初始化所有层，但对于Sigmoid之前的层（Gating和Decision Head），初始化过大会导致Sigmoid饱和，梯度消失。

### 修正
- **Gating层**：使用较小的初始化（std=0.01），偏置为0（初始权重0.5，公平）
- **Decision Head**：使用较小的初始化（std=0.01），最后一层偏置设为-2.0（初始输出低概率，因为大部分是背景）

```python
# 修正前
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# 修正后
if 'gate' in name or 'decision_head' in name:
    nn.init.normal_(m.weight, mean=0.0, std=0.01)  # 较小的初始化
    if m.bias is not None:
        if 'gate' in name:
            nn.init.constant_(m.bias, 0.0)  # 初始权重0.5
        elif 'decision_head' in name:
            # 最后一层偏置设为负值
            nn.init.constant_(m.bias, -2.0)  # 初始输出低概率
```

**文件**：`src/models/fusion_head.py` - `_initialize_weights()`

**原理**：
- Gating初始化为0.5（公平），让模型学习何时信任图像/点云
- Decision Head初始化为低概率（因为大部分是背景），避免Sigmoid饱和

## 3. 伪异常生成Mask尺寸修正 ✅

### 问题
`min_size` 和 `max_size` 参数是绝对像素值，但在特征图上使用时，如果这些值是基于原图像素的，会导致异常区域过大（覆盖半个屏幕）。

### 修正
改为相对于特征图尺寸的比例参数：
```python
# 修正前
def generate_random_box_mask(
    shape: Tuple[int, ...],
    min_size: int = 20,  # 绝对像素值
    max_size: int = 100,
    ...
)

# 修正后
def generate_random_box_mask(
    shape: Tuple[int, ...],
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    min_size_ratio: float = 0.05,  # 相对于特征图尺寸的比例
    max_size_ratio: float = 0.2,
    ...
)
```

**文件**：`src/utils/pseudo_anomaly.py` - `generate_random_box_mask()`

**默认值**：
- `min_size_ratio=0.05`：特征图的5%（例如200x200特征图，最小10像素）
- `max_size_ratio=0.2`：特征图的20%（例如200x200特征图，最大40像素）

## 4. Fusion Head输出数值稳定性 ✅

### 修正
在Fusion Head输出后添加clamp，确保输出在合理范围内：
```python
anomaly_map = self.decision_head(gated_feat)
anomaly_map = torch.clamp(anomaly_map, min=1e-6, max=1.0 - 1e-6)
```

**文件**：`src/models/fusion_head.py` - `forward()`

## 5. 训练脚本更新 ✅

### 修正
更新训练脚本中的伪异常生成调用，使用比例参数：
```python
img_features_corrupted, pts_features_corrupted, anomaly_mask = generate_pseudo_anomalies(
    img_features,
    pts_features_proj,
    anomaly_prob=anomaly_prob,
    num_boxes=1,
    noise_type=noise_type,
    noise_scale=noise_scale,
    min_size_ratio=0.05,  # 特征图的5%
    max_size_ratio=0.2    # 特征图的20%
)
```

**文件**：`src/training/train_anomaly_detector.py` - `train_one_epoch()`

## 总结

✅ **已完成**：
1. Focal Loss数值稳定性（clamp预测值）
2. Fusion Head初始化（Gating和Decision Head使用较小初始化）
3. 伪异常生成Mask尺寸（使用相对比例而非绝对像素）
4. Fusion Head输出数值稳定性（clamp输出）

**当前状态**：所有数值稳定性和初始化问题已修正，代码可以安全地进行大规模训练。

