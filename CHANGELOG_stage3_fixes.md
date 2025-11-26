# 阶段三 Feature Splatting 代码修正日志

根据代码审查反馈，进行了以下关键修正：

## 1. 核心数学修正 (Critical Fixes)

### A. 雅可比矩阵推导修正 ✅

**问题**：原代码使用 `pixel_coords (u, v)` 来替代物理坐标 `(X, Y, Z)`，导致单位混乱和计算错误。

**修正**：
- 修改 `project_3d_to_2d()` 函数，返回 `cam_coords`（相机坐标系坐标）
- 修改 `compute_2d_covariance()` 函数，使用真实的相机坐标 `(X, Y, Z)` 计算雅可比矩阵
- 正确的雅可比矩阵：
  ```
  J = [fx/Z,    0,      -fx*X/Z^2]
      [0,        fy/Z,   -fy*Y/Z^2]
  ```

**文件**：`src/models/feature_splatting.py`
- `project_3d_to_2d()`: 现在返回 `(pixel_coords, depths, cam_coords)`
- `compute_2d_covariance()`: 参数从 `pixel_coords` 改为 `cam_coords`

### B. 3D协方差初始化改进 ✅

**问题**：原代码假设各向同性高斯，但道路场景是扁平的。

**修正**：
- 添加 `road_aware` 参数（默认 `True`）
- 道路感知初始化：
  - X轴（前）：`(voxel_size/2)^2`
  - Y轴（左）：`(voxel_size/2)^2`
  - Z轴（上，高度轴）：`(voxel_size/4)^2` （方差更小，贴合路面）

**文件**：`src/models/feature_splatting.py`
- `compute_3d_covariance()`: 添加 `road_aware` 参数

## 2. 数值稳定性改进 ✅

### A. 深度过滤改进

**问题**：原代码使用 `clamp`，但相机背后的点应该直接过滤。

**修正**：
- 在 `project_3d_to_2d()` 中，只对有效点（深度 > 0.1米）计算像素坐标
- 在 `forward()` 中，添加深度过滤：`depths > 0.1`（避免数值问题）

### B. 协方差逆矩阵数值稳定性

**问题**：协方差矩阵可能奇异，导致计算不稳定。

**修正**：
- 在计算逆矩阵前，添加小的正则项：`cov_2d + eye(2) * 1e-5`
- 使用 `try-except` 处理奇异情况，回退到伪逆

**文件**：`src/models/feature_splatting.py`
- `rasterize_splats()`: 改进协方差逆矩阵计算

## 3. 性能优化 ⚠️

### A. Splat半径限制

**修正**：
- 添加最大Splat半径限制：`max_splat_radius = min(self.splat_radius * 3, 10.0)`
- 避免过大的计算量

### B. 向量化改进（部分）

**修正**：
- 改进像素索引的向量化处理
- 注意：完整的向量化Splatting需要CUDA实现，当前版本作为MVP版本

**建议**：如果性能成为瓶颈，可以考虑：
1. 使用 `scatter_add` 进行更高效的累积
2. 实现CUDA版本的Gaussian Rasterizer
3. 或者先使用简单的Scatter Mean（硬投影）作为baseline

## 4. 代码细节修正 ✅

### A. 投影矩阵类型确保

**修正**：确保投影矩阵是 `float` 类型
```python
projection_matrix = projection_matrix.float().to(voxel_features.device)
```

### B. 测试代码更新

**修正**：更新测试代码以适配新的函数签名
- `test_3d_to_2d_projection()`: 适配返回3个值
- `test_2d_covariance_computation()`: 使用 `cam_coords` 而不是 `pixel_coords`

**文件**：`tests/test_feature_splatting.py`

## 5. 待完成项目

### A. 坐标系验证 ⏳

**建议**：添加坐标系验证测试
- 验证LiDAR点 `(10, 0, -1.73)`（车前方10米地面）投影到图像下半部分中间
- 确认MinkUNet输出的坐标系定义
- 验证KITTI/AnoVox的坐标系转换

### B. 完整性能优化 ⏳

**建议**：
- 实现完全向量化的Splatting（使用CUDA或高级PyTorch操作）
- 或者提供简化版本（Scatter Mean）作为快速baseline

## 总结

✅ **已完成**：
1. 雅可比矩阵修正（使用真实相机坐标）
2. 3D协方差道路感知初始化
3. 深度过滤和数值稳定性改进
4. 基本性能优化（半径限制）

⏳ **待完成**：
1. 坐标系验证测试
2. 完整性能优化（向量化Splatting）

**当前状态**：代码框架正确，数学推导已修正，可以开始端到端训练。性能优化可以在后续迭代中完成。

