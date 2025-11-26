# 背景
文件名：2025-01-14_5_implement-anovox-dataloader.md
创建于：2025-01-14
创建者：root
主分支：main
任务分支：task/implement-anovox-dataloader_2025-01-14_5
Yolo模式：Ask

# 任务描述
实现AnoVox数据集DataLoader，完成阶段五的数据加载部分。

# 项目概览
本项目旨在实现一个融合图像和点云的道路异常检测系统，采用五阶段架构：
1. 阶段一：冻结的Mask2Former（2D图像分支）✅
2. 阶段二：冻结的MinkUNet（3D点云分支）✅
3. 阶段三：Feature Splatting投影（核心创新）✅
4. 阶段四：轻量级融合头 ✅
5. 阶段五：AnoVox数据集DataLoader ✅

# 数据集结构

AnoVox_Normality_Mono_Town03/
├── Scenario_251237df-b040-4cc4-b3e5-b2c852929280/
│   ├── RGB-CAM(0, 0, 1.8)(0, 0, 0)_1932154471836802243/
│   │   └── RGB-CAM(...)_帧ID.png
│   ├── LIDAR(0, 0, 1.8)(0, 0, 0)_6343367489698313030/
│   │   └── LIDAR(...)_帧ID.npy
│   └── sensor_setup.json
└── ...

# 实现内容

## 已创建/修改文件

1. **src/datasets/anovox_dataset.py**
   - `AnoVoxDataset`类：数据集加载器
   - `parse_sensor_params_from_dirname()`：从文件夹名解析传感器参数
   - `extract_intrinsic_from_config()`：从sensor_setup.json提取相机内参
   - `build_extrinsic_matrix()`：构建外参矩阵
   - `collate_fn()`：自定义批次处理函数

2. **src/datasets/__init__.py**
   - 模块初始化文件

3. **src/training/train_anomaly_detector.py**
   - 更新训练脚本，集成DataLoader
   - 添加命令行参数支持
   - 修复投影矩阵批次处理

4. **tests/test_anovox_dataset.py**
   - 数据集加载器测试脚本

5. **README.md**
   - 更新文档，添加DataLoader使用说明

# 任务进度

[2025-01-14]
- ✅ 已完成：AnoVoxDataset类实现
- ✅ 已完成：传感器参数解析（从文件夹名）
- ✅ 已完成：相机内参提取（从sensor_setup.json）
- ✅ 已完成：投影矩阵计算
- ✅ 已完成：文件名匹配逻辑（适配实际格式）
- ✅ 已完成：训练脚本集成
- ✅ 已完成：测试脚本编写
- ✅ 已完成：README文档更新
- ✅ 已完成：测试验证（2600帧数据对成功加载）
- ✅ 已完成：坐标系转换修复（LiDAR到相机坐标轴转换）
- ✅ 已完成：点云通道数修复（确保4维以匹配MinkUNet）
- ✅ 已完成：路径匹配健壮性改进（添加警告信息）

# 技术细节

## 文件名格式适配

实际文件名格式：
- RGB: `RGB-CAM(0, 0, 1.8)(0, 0, 0)_1932154471836802243_4884.png`
- LIDAR: `LIDAR(0, 0, 1.8)(0, 0, 0)_6343367489698313030_4884.npy`

匹配策略：
- 从RGB文件名提取帧ID（最后一个下划线后的数字）
- 在LIDAR目录中查找匹配的文件（使用glob模式匹配）

## sensor_setup.json解析

JSON格式：
```json
{
  "RGB-CAM(...)": {
    "sensor_type": "RGB-CAM",
    "args": {
      "image_height": 512,
      "image_width": 768,
      "camera_fov": 90.0
    }
  }
}
```

内参计算：
- 从args中提取image_height, image_width, camera_fov
- 计算焦距：f = (w/2) / tan(FOV/2)
- 构建内参矩阵K

## 投影矩阵计算

1. **内参矩阵K**：从sensor_setup.json提取
2. **外参矩阵T**：从文件夹名解析传感器位置和旋转
   - **关键修复**：添加了LiDAR到相机的坐标轴转换矩阵
   - LiDAR坐标系：X-前, Y-右, Z-上
   - Camera坐标系：X-右, Y-下, Z-前
   - 转换矩阵：`R_lidar2cam = [[0,1,0], [0,0,-1], [1,0,0]]`
3. **投影矩阵P**：P = K @ T[:3, :]

## 关键修复

### 1. 坐标系转换（Critical Fix）
**问题**：即使物理位置重合，LiDAR和相机的坐标系定义不同，不能直接用单位矩阵。

**修复**：添加标准坐标轴转换矩阵：
```python
rotation_lidar2cam = np.array([
    [0, 1, 0],   # x_cam = y_lidar
    [0, 0, -1],  # y_cam = -z_lidar
    [1, 0, 0]    # z_cam = x_lidar
])
```

### 2. 点云通道数（Critical Fix）
**问题**：MinkUNet预训练权重期望4维(x, y, z, intensity)，如果只有3维会导致权重形状不匹配。

**修复**：如果点云只有3维，自动补全intensity通道（设为1）：
```python
if points.shape[1] == 3:
    intensity = np.ones((points.shape[0], 1), dtype=points.dtype)
    points = np.hstack((points, intensity))
```

### 3. 路径匹配健壮性
**改进**：添加警告信息，方便调试数据完整性。

# 测试结果

## 初始测试
```
数据集路径: /root/autodl-tmp/dataset/anovox/AnoVox_Normality_Mono_Town03
扫描到 14 个场景...
共加载 2600 帧数据对。
✓ 数据集创建成功
✓ 样本加载成功
✓ DataLoader测试成功
✓ 所有测试通过！
```

## 修复后验证
```
点云形状: torch.Size([89147, 4])
  ✓ 点云通道数正确（包含intensity）

投影矩阵形状: torch.Size([3, 4])
投影矩阵前3列（旋转部分）:
tensor([[ 384.,  384.,    0.],
        [ 256.,    0., -384.],
        [   1.,    0.,    0.]])
  ✓ 坐标轴转换已应用（非单位矩阵）
```

**关键验证点**：
- ✅ 点云形状：(N, 4) - intensity通道已补全
- ✅ 投影矩阵：坐标轴转换已正确应用
- ✅ 数据完整性：2600帧数据对全部匹配成功

# 下一步

- [ ] 开始实际训练（使用命令行或Python脚本）
- [ ] 实现评估指标（FPR95、AUPR等）
- [ ] 实现验证集DataLoader（Anomaly数据）

# 最终审查

## 完成情况

阶段五：AnoVox数据集DataLoader已全部完成 ✅

### 核心功能
1. **自动配对**：成功匹配RGB图像和LiDAR点云（2600帧）
2. **投影矩阵计算**：正确解析sensor_setup.json和文件夹名
3. **批次处理**：自定义collate_fn处理点云列表
4. **训练集成**：完整集成到训练脚本

### 代码质量
- ✅ 无linter错误
- ✅ 测试全部通过
- ✅ 适配实际数据集格式

### 数据集信息
- 场景数量：14个
- 数据对数量：2600帧
- 图像尺寸：512x768（原始），800x1333（预处理后）
- 点云格式：.npy (N, 3) 或 (N, 4)

## 总结

AnoVox数据集DataLoader已完全实现并测试通过，可以开始训练模型。

