"""
下载Mask2Former Swin-Large Cityscapes预训练权重
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


def download_mask2former_weights(
    save_dir: str = "checkpoints/mask2former",
    weight_url: str = None
) -> str:
    """
    下载Mask2Former Swin-Large Cityscapes预训练权重
    
    Args:
        save_dir: 保存目录
        weight_url: 权重文件下载URL，如果为None则使用默认URL
        
    Returns:
        权重文件路径
    """
    # 默认权重URL（用户提供的官方链接）
    if weight_url is None:
        weight_url = "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_064788.pkl"
    
    # 保存路径（注意：原始文件是.pkl格式，但detectron2通常使用.pth）
    # 我们保持原始文件名，detectron2可以自动处理
    filename = os.path.basename(weight_url)
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
    
    print(f"开始下载Mask2Former权重...")
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
        print("="*60)
        return None
    
    return save_path


def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    save_dir = project_root / "checkpoints" / "mask2former"
    
    print("="*60)
    print("Mask2Former权重下载工具")
    print("="*60)
    
    weight_path = download_mask2former_weights(str(save_dir))
    
    if weight_path:
        print(f"\n✓ 权重文件已准备就绪: {weight_path}")
    else:
        print(f"\n✗ 权重文件下载失败，请手动下载")
        sys.exit(1)


if __name__ == "__main__":
    main()

