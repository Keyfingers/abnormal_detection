# AnoVox数据集准备指南

## 数据集要求

根据代码要求，AnoVox数据集需要包含以下结构：

```
anovox/
├── train/
│   ├── images/          # 训练图像（.jpg或.png）
│   ├── pointclouds/     # 训练点云（.bin格式，Nx4，前3列为xyz）
│   ├── anomaly_masks/   # 训练异常掩码（.png，二值化）
│   └── calibrations/    # 相机标定文件（.txt，包含内参和外参）
└── test/
    ├── images/          # 测试图像
    ├── pointclouds/     # 测试点云
    ├── anomaly_masks/   # 测试异常掩码
    └── calibrations/    # 相机标定文件
```

## 选择哪个版本？

### 推荐选择标准：

1. **最新版本**：通常包含最新的修复和更新
2. **完整版本**：包含训练集和测试集
3. **包含标定数据**：必须包含相机内参和外参（用于3D-2D投影）

### Zenodo上的AnoVox版本：

访问：https://zenodo.org/communities/anovox/records?q=&l=list&p=1&s=10&sort=newest

常见版本：
- **AnoVox v1.0**：初始发布版本
- **AnoVox v1.1**：更新版本（如果有）
- **AnoVox Full Dataset**：完整数据集
- **AnoVox Train/Test Split**：已分割的训练/测试集

## 下载步骤

### 方法1：使用脚本（推荐）

```bash
chmod +x scripts/download_anovox.sh
./scripts/download_anovox.sh
```

脚本会提示您输入Zenodo记录ID。

### 方法2：手动下载

1. 访问Zenodo页面：https://zenodo.org/communities/anovox/records
2. 选择合适的数据集版本（推荐最新完整版本）
3. 下载数据集文件（通常是zip或tar.gz）
4. 解压到 `/root/autodl-tmp/dataset/anovox/`

### 方法3：使用wget/curl

```bash
# 替换 RECORD_ID 为实际的Zenodo记录ID
RECORD_ID="your_record_id"
ZENODO_URL="https://zenodo.org/record/${RECORD_ID}/files"

# 下载
wget "${ZENODO_URL}/anovox.zip" -O /root/autodl-tmp/dataset/anovox.zip

# 解压
cd /root/autodl-tmp/dataset
unzip anovox.zip -d anovox/
```

## 验证数据集

下载完成后，运行验证脚本：

```bash
python scripts/verify_anovox_dataset.py --data-root /root/autodl-tmp/dataset/anovox
```

## 相机标定文件格式

标定文件（`calibrations/{sample_id}.txt`）应包含：

```
# 相机内参 (3x3)
intrinsic:
fx 0 cx
0 fy cy
0 0 1

# 相机外参 (4x4)
extrinsic:
r11 r12 r13 tx
r21 r22 r23 ty
r31 r32 r33 tz
0 0 0 1
```

或者JSON格式：

```json
{
  "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "extrinsic": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]]
}
```

## 注意事项

1. **点云格式**：代码期望`.bin`格式，每个点4个float32（x, y, z, intensity），只使用前3列
2. **图像格式**：支持`.jpg`或`.png`
3. **掩码格式**：`.png`格式，灰度图，>128为异常区域
4. **标定数据**：必须包含，否则无法进行3D-2D投影

## 如果数据集格式不同

如果下载的AnoVox数据集格式与代码期望不同，可能需要：

1. 重命名目录结构
2. 转换点云格式（如果不同）
3. 调整标定文件格式（如果不同）

可以修改 `src/data/anovox_dataset.py` 来适配实际的数据格式。






