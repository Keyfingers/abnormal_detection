#!/usr/bin/env python3
"""
验证AnoVox数据集是否完整且格式正确
"""
import os
import sys
import argparse
from pathlib import Path

def verify_anovox_dataset(data_root: str):
    """验证AnoVox数据集"""
    data_root = Path(data_root)
    
    print("=" * 60)
    print("AnoVox数据集验证")
    print("=" * 60)
    print(f"数据根目录: {data_root}")
    print()
    
    errors = []
    warnings = []
    
    # 检查基本目录结构
    required_splits = ['train', 'test']
    required_subdirs = ['images', 'pointclouds', 'anomaly_masks', 'calibrations']
    
    for split in required_splits:
        split_dir = data_root / split
        if not split_dir.exists():
            errors.append(f"缺少目录: {split_dir}")
            continue
        
        print(f"检查 {split} 目录...")
        
        # 检查子目录
        for subdir in required_subdirs:
            subdir_path = split_dir / subdir
            if not subdir_path.exists():
                errors.append(f"缺少目录: {subdir_path}")
            else:
                file_count = len(list(subdir_path.glob('*')))
                print(f"  {subdir}: {file_count} 个文件")
                if file_count == 0:
                    warnings.append(f"目录为空: {subdir_path}")
        
        # 检查文件一致性
        images_dir = split_dir / 'images'
        pointclouds_dir = split_dir / 'pointclouds'
        masks_dir = split_dir / 'anomaly_masks'
        calibrations_dir = split_dir / 'calibrations'
        
        if all(d.exists() for d in [images_dir, pointclouds_dir, masks_dir, calibrations_dir]):
            # 获取样本ID
            image_files = set(f.stem for f in images_dir.glob('*.jpg')) | set(f.stem for f in images_dir.glob('*.png'))
            pc_files = set(f.stem for f in pointclouds_dir.glob('*.bin'))
            mask_files = set(f.stem for f in masks_dir.glob('*.png'))
            calib_files = set(f.stem for f in calibrations_dir.glob('*.txt')) | set(f.stem for f in calibrations_dir.glob('*.json'))
            
            print(f"  样本统计:")
            print(f"    图像: {len(image_files)}")
            print(f"    点云: {len(pc_files)}")
            print(f"    掩码: {len(mask_files)}")
            print(f"    标定: {len(calib_files)}")
            
            # 检查文件一致性
            all_ids = image_files | pc_files | mask_files | calib_files
            missing_images = all_ids - image_files
            missing_pcs = all_ids - pc_files
            missing_masks = all_ids - mask_files
            missing_calibs = all_ids - calib_files
            
            if missing_images:
                warnings.append(f"{split}: {len(missing_images)} 个样本缺少图像")
            if missing_pcs:
                warnings.append(f"{split}: {len(missing_pcs)} 个样本缺少点云")
            if missing_masks:
                warnings.append(f"{split}: {len(missing_masks)} 个样本缺少掩码")
            if missing_calibs:
                warnings.append(f"{split}: {len(missing_calibs)} 个样本缺少标定文件")
    
    print()
    print("=" * 60)
    print("验证结果")
    print("=" * 60)
    
    if errors:
        print("❌ 错误:")
        for error in errors:
            print(f"  - {error}")
        print()
    
    if warnings:
        print("⚠️  警告:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    if not errors and not warnings:
        print("✅ 数据集验证通过！")
        return True
    elif not errors:
        print("⚠️  数据集基本完整，但有一些警告")
        return True
    else:
        print("❌ 数据集不完整，请检查上述错误")
        return False


def main():
    parser = argparse.ArgumentParser(description='验证AnoVox数据集')
    parser.add_argument('--data-root', type=str, required=True,
                        help='AnoVox数据根目录')
    
    args = parser.parse_args()
    
    success = verify_anovox_dataset(args.data_root)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()





