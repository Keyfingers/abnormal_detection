# AnoVox数据集下载指南

## 概述

本目录包含用于下载AnoVox数据集的脚本，支持断点续传和多节点下载。

## 下载脚本

### 1. `download_anovox_complete.sh` - 交互式下载脚本

**功能**：
- 交互式选择要下载的数据集
- 支持断点续传
- 支持多节点下载（使用aria2c）
- 自动解压下载的文件

**使用方法**：
```bash
cd /root/autodl-tmp/abnormal_detection/scripts
./download_anovox_complete.sh
```

**交互选项**：
- 选择数据集编号（多个用空格分隔）
- 脚本会自动下载并解压选定的数据集

### 2. `download_anovox_auto.sh` - 自动下载脚本（推荐）

**功能**：
- 非交互式，直接下载推荐的数据集
- 支持断点续传
- 支持多节点下载（使用aria2c）
- 自动解压下载的文件

**使用方法**：
```bash
cd /root/autodl-tmp/abnormal_detection/scripts

# 下载所有推荐数据集（默认）
./download_anovox_auto.sh

# 或指定模式
./download_anovox_auto.sh normality  # 仅下载常态训练集
./download_anovox_auto.sh static      # 仅下载静态异常评估集
./download_anovox_auto.sh all        # 下载所有推荐数据集
```

## 数据集说明

### 常态训练集（用于阶段4微调）

1. **AnoVox_Normality_Mono_Town03** (ID: 10887912)
   - 用途：训练融合头学习"正常时特征是什么样的"
   - 特点：所有样本都是正常样本，异常掩码全为0

### 内容/静态异常评估集（用于阶段5评估）

1. **AnoVox_Static_Mono_Town10_1** (ID: 10881577)
   - 用途：评估模型在异常样本上的表现
   - 特点：包含异常样本和异常掩码

2. **AnoVox_Static_Surround_Town10** (ID: 10897966)
   - 用途：评估模型在异常样本上的表现
   - 特点：包含异常样本和异常掩码

## 下载路径

所有数据集将下载到：
```
/root/autodl-tmp/dataset/anovox/
├── downloads/          # 下载的压缩文件
├── download_logs/      # 下载日志
├── AnoVox_Normality_Mono_Town03/     # 解压后的常态训练集
├── AnoVox_Static_Mono_Town10_1/      # 解压后的静态异常评估集
└── AnoVox_Static_Surround_Town10/    # 解压后的静态异常评估集
```

## 下载工具

脚本会自动检测并使用以下下载工具（按优先级）：

1. **aria2c**（推荐）
   - 支持多节点下载（16个连接）
   - 支持断点续传
   - 下载速度更快

2. **wget**（备选）
   - 支持断点续传
   - 单连接下载

如果系统中没有安装aria2c，脚本会尝试自动安装。

## 断点续传

脚本支持断点续传功能：
- 如果下载中断，重新运行脚本会自动从断点继续下载
- 已下载的文件会被跳过（除非手动删除）

## 注意事项

1. **磁盘空间**：确保有足够的磁盘空间（建议至少50GB）
2. **网络连接**：下载大文件需要稳定的网络连接
3. **下载时间**：根据网络速度，完整下载可能需要数小时
4. **文件完整性**：下载完成后，脚本会自动解压文件

## 故障排除

### 问题1：无法获取下载链接

**原因**：Zenodo API可能暂时不可用或记录ID错误

**解决方案**：
- 检查网络连接
- 验证Zenodo记录ID是否正确
- 稍后重试

### 问题2：下载速度慢

**原因**：网络带宽限制或Zenodo服务器负载高

**解决方案**：
- 确保使用aria2c（多节点下载）
- 在网络条件较好的时段下载
- 检查防火墙设置

### 问题3：下载中断

**原因**：网络不稳定或连接超时

**解决方案**：
- 重新运行脚本（支持断点续传）
- 检查网络连接稳定性
- 查看下载日志：`/root/autodl-tmp/dataset/anovox/download_logs/`

## 验证下载

下载完成后，可以运行验证脚本检查数据集：

```bash
python scripts/verify_anovox_dataset.py --data-root /root/autodl-tmp/dataset/anovox
```

## 下一步

下载完成后，可以开始：

1. **阶段4：训练融合模块**
   ```bash
   python src/training/train_fusion.py \
       --data-root /root/autodl-tmp/dataset/anovox \
       --semantic-ckpt <path_to_2d_checkpoint> \
       --geometric-ckpt <path_to_3d_checkpoint> \
       --output-dir outputs/fusion
   ```

2. **阶段5：评估模型**
   ```bash
   python scripts/evaluate_fusion.py \
       --checkpoint outputs/fusion/model_final.pth \
       --data-root /root/autodl-tmp/dataset/anovox
   ```

## 相关文档

- [QUICKSTART.md](../QUICKSTART.md) - 项目快速开始指南
- [ANOVOX_DATASET_STATUS.md](../ANOVOX_DATASET_STATUS.md) - 数据集状态报告
- [ANOVOX_SETUP_COMPLETE.md](../ANOVOX_SETUP_COMPLETE.md) - 数据集准备完成总结



