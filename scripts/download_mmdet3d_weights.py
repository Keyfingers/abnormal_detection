"""
下载MMDetection3D SemanticKITTI预训练权重
"""
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, save_path: str, chunk_size: int = 8192):
    """
    下载文件并显示进度
    
    Args:
        url: 下载链接
        save_path: 保存路径
        chunk_size: 每次下载的块大小
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ 下载完成: {save_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ 下载失败: {e}")
        return False


def download_mmdet3d_weights(
    save_dir: str = "checkpoints/mmdet3d",
    model_name: str = "pointnet2_ssg",
    weight_url: str = None
) -> str:
    """
    下载MMDetection3D SemanticKITTI预训练权重
    
    MMDetection3D提供了多种点云分割模型的预训练权重。
    可以从OpenMMLab Model Zoo下载：
    https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointnet2
    
    Args:
        save_dir: 保存目录
        model_name: 模型名称（如pointnet2_ssg, spvcnn等）
        weight_url: 权重文件下载URL，如果为None则使用默认URL或提示手动下载
        
    Returns:
        权重文件路径，如果下载失败则返回None
    """
    # MMDetection3D Model Zoo的权重URL
    # MinkUNet SemanticKITTI预训练权重
    default_minkunet_url = "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/minkunet_w32_8xb2-15e_semantickitti/minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth"
    
    # 默认权重URL（根据模型名称选择）
    if weight_url is None:
        if model_name.lower() in ['minkunet', 'minkunet_w32']:
            weight_url = default_minkunet_url
            print(f"使用默认MinkUNet SemanticKITTI预训练权重")
        else:
            print("\n" + "="*60)
            print("MMDetection3D SemanticKITTI预训练权重下载")
            print("="*60)
            print("\nMMDetection3D提供了多种点云分割模型的预训练权重。")
            print("请从以下资源获取权重：")
            print("\n1. MMDetection3D Model Zoo:")
            print("   https://github.com/open-mmlab/mmdetection3d")
            print("   查看 configs/ 目录下的模型配置文件")
            print("\n2. 常用模型和权重链接：")
            print("   - MinkUNet SemanticKITTI:")
            print(f"     {default_minkunet_url}")
            print("   - PointNet++: https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointnet2")
            print("   - SPVCNN: https://github.com/open-mmlab/mmdetection3d/tree/main/configs/spvcnn")
            print("\n3. 下载方式：")
            print("   - 使用 --model minkunet 自动下载MinkUNet权重")
            print("   - 从配置文件中的checkpoint字段获取权重URL")
            print("   - 或使用 --url 参数指定权重URL")
            print("\n4. 如果找不到预训练权重，可以：")
            print("   - 使用 --create-placeholder 创建占位权重文件（仅用于测试）")
            print("   - 在SemanticKITTI数据集上从头训练")
            print("="*60)
            return None
    
    # 保存路径
    filename = os.path.basename(weight_url) if weight_url else f"{model_name}_semantickitti.pth"
    if not filename.endswith(('.pth', '.pkl', '.ckpt')):
        filename = filename + ".pth"
    
    save_path = os.path.join(save_dir, filename)
    
    # 检查文件是否已存在
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        if file_size > 0:
            print(f"✓ 权重文件已存在: {save_path} ({file_size / 1024 / 1024:.2f} MB)")
            return save_path
        else:
            print(f"⚠ 权重文件存在但为空，将重新下载")
            os.remove(save_path)
    
    print(f"开始下载MMDetection3D权重...")
    print(f"模型: {model_name}")
    print(f"URL: {weight_url}")
    print(f"保存路径: {save_path}")
    
    # 下载文件
    success = download_file(weight_url, save_path)
    
    if not success:
        print("\n" + "="*60)
        print("自动下载失败。请手动下载权重文件：")
        print(f"URL: {weight_url}")
        print(f"保存路径: {save_path}")
        print("\n手动下载步骤：")
        print(f"1. 访问: {weight_url}")
        print(f"2. 下载文件到: {save_path}")
        print("\n或使用以下命令：")
        print(f"  wget {weight_url} -O {save_path}")
        print("="*60)
        return None
    
    return save_path


def create_placeholder_weights(save_dir: str = "checkpoints/mmdet3d"):
    """
    创建占位权重文件（用于测试）
    
    注意：这只是为了测试代码结构，实际使用时需要真实的预训练权重。
    
    Args:
        save_dir: 保存目录
    """
    import torch
    
    save_path = os.path.join(save_dir, "mmdet3d_placeholder.pth")
    
    # 创建一个简单的占位权重字典
    # 实际使用时，这个结构应该匹配真实的MMDetection3D权重
    placeholder_weights = {
        'state_dict': {},
        'meta': {
            'config': 'Placeholder config for testing',
            'epoch': 0,
            'iter': 0
        },
        'note': 'This is a placeholder weight file for testing. Please replace with real SemanticKITTI pretrained weights from MMDetection3D Model Zoo.'
    }
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(placeholder_weights, save_path)
    
    print(f"✓ 已创建占位权重文件: {save_path}")
    print("⚠ 警告：这是占位文件，仅用于测试代码结构。")
    print("   实际使用时，请替换为真实的SemanticKITTI预训练权重。")
    print("\n获取真实权重的方法：")
    print("1. 访问 https://github.com/open-mmlab/mmdetection3d")
    print("2. 查看 configs/ 目录下的模型配置文件")
    print("3. 从配置文件中的checkpoint字段获取权重URL")
    
    return save_path


def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    save_dir = project_root / "checkpoints" / "mmdet3d"
    
    print("="*60)
    print("MMDetection3D权重下载工具")
    print("="*60)
    print("\n默认模型: MinkUNet (SemanticKITTI预训练)")
    print("默认权重URL:")
    print("https://download.openmmlab.com/mmdetection3d/v1.1.0_models/minkunet/")
    print("minkunet_w32_8xb2-15e_semantickitti/")
    print("minkunet_w32_8xb2-15e_semantickitti_20230309_160710-7fa0a6f1.pth")
    print("="*60)
    
    # 尝试从环境变量或命令行参数获取权重URL
    import argparse
    parser = argparse.ArgumentParser(description='Download MMDetection3D weights')
    parser.add_argument('--url', type=str, default=None,
                       help='Weight file download URL')
    parser.add_argument('--model', type=str, default='minkunet',
                       help='Model name (e.g., minkunet, pointnet2_ssg, spvcnn)')
    parser.add_argument('--create-placeholder', action='store_true',
                       help='Create a placeholder weight file for testing')
    args = parser.parse_args()
    
    if args.create_placeholder:
        weight_path = create_placeholder_weights(str(save_dir))
        if weight_path:
            print(f"\n✓ 占位权重文件已创建: {weight_path}")
        return
    
    weight_path = download_mmdet3d_weights(
        str(save_dir),
        model_name=args.model,
        weight_url=args.url
    )
    
    if weight_path:
        print(f"\n✓ 权重文件已准备就绪: {weight_path}")
    else:
        print(f"\n✗ 权重文件下载失败")
        print("\n提示：如果找不到预训练权重，可以使用 --create-placeholder 创建占位文件进行测试")
        print("示例：python scripts/download_mmdet3d_weights.py --create-placeholder")
        sys.exit(1)


if __name__ == "__main__":
    main()

