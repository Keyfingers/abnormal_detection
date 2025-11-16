"""
可视化工具：用于定性分析
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os


def visualize_anomaly_detection(
    image: np.ndarray,
    anomaly_score: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Anomaly Detection",
):
    """
    可视化异常检测结果
    
    修复说明：
    - 规则要求：必须包含关键案例的定性分析（2D失败/3D失败/融合成功）
    - 这个函数用于生成可视化图像，便于分析模型性能
    
    Args:
        image: 原始图像 (H, W, 3) 或 (3, H, W)
        anomaly_score: 异常分数图 (H, W)，值在[0,1]
        ground_truth: 真实异常掩码 (H, W)，可选
        save_path: 保存路径，可选
        title: 图像标题
    """
    # 确保图像格式正确
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # 归一化图像到[0,1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # 创建图形
    fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))
    if ground_truth is None:
        axes = [axes, None]
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 异常分数图（热力图）
    im = axes[1].imshow(anomaly_score, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Anomaly Score')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 真实掩码（如果有）
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def visualize_comparison(
    image: np.ndarray,
    score_2d: np.ndarray,
    score_3d: np.ndarray,
    score_fusion: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Method Comparison",
):
    """
    可视化三种方法的对比结果
    
    修复说明：
    - 规则要求：必须包含关键案例的定性分析
    - 这个函数用于对比2D-only、3D-only和融合方法的结果
    - 便于识别2D失败/3D失败/融合成功的案例
    
    Args:
        image: 原始图像 (H, W, 3) 或 (3, H, W)
        score_2d: 2D基线异常分数 (H, W)
        score_3d: 3D基线异常分数 (H, W)
        score_fusion: 融合方法异常分数 (H, W)
        ground_truth: 真实异常掩码 (H, W)，可选
        save_path: 保存路径，可选
        title: 图像标题
    """
    # 确保图像格式正确
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # 归一化图像
    if image.max() > 1.0:
        image = image / 255.0
    
    # 创建图形
    num_cols = 5 if ground_truth is not None else 4
    fig, axes = plt.subplots(1, num_cols, figsize=(20, 4))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # 2D基线
    im1 = axes[1].imshow(score_2d, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('2D Baseline (RbA)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # 3D基线
    im2 = axes[2].imshow(score_3d, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('3D Baseline (Reconstruction)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    # 融合方法
    im3 = axes[3].imshow(score_fusion, cmap='hot', vmin=0, vmax=1)
    axes[3].set_title('Fusion Method', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3])
    
    # 真实掩码（如果有）
    if ground_truth is not None:
        axes[4].imshow(ground_truth, cmap='gray')
        axes[4].set_title('Ground Truth', fontsize=12)
        axes[4].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison visualization to {save_path}")
    
    plt.close()


def save_qualitative_results(
    images: list,
    scores_2d: list,
    scores_3d: list,
    scores_fusion: list,
    ground_truths: Optional[list] = None,
    sample_ids: Optional[list] = None,
    output_dir: str = "outputs/qualitative",
    case_type: str = "all",
):
    """
    保存定性分析结果
    
    修复说明：
    - 规则要求：必须包含关键案例（2D失败/3D失败/融合成功）的定性分析
    - 这个函数用于批量保存可视化结果
    - case_type可以是："2d_failure", "3d_failure", "fusion_success", "all"
    
    Args:
        images: 图像列表
        scores_2d: 2D基线分数列表
        scores_3d: 3D基线分数列表
        scores_fusion: 融合方法分数列表
        ground_truths: 真实掩码列表，可选
        sample_ids: 样本ID列表，可选
        output_dir: 输出目录
        case_type: 案例类型
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (img, s2d, s3d, sf) in enumerate(zip(images, scores_2d, scores_3d, scores_fusion)):
        sample_id = sample_ids[idx] if sample_ids else f"sample_{idx}"
        gt = ground_truths[idx] if ground_truths else None
        
        # 转换为numpy数组
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(s2d, torch.Tensor):
            s2d = s2d.cpu().numpy()
        if isinstance(s3d, torch.Tensor):
            s3d = s3d.cpu().numpy()
        if isinstance(sf, torch.Tensor):
            sf = sf.cpu().numpy()
        if gt is not None and isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        
        # 保存对比图
        save_path = os.path.join(output_dir, f"{case_type}_{sample_id}_comparison.png")
        visualize_comparison(
            image=img,
            score_2d=s2d,
            score_3d=s3d,
            score_fusion=sf,
            ground_truth=gt,
            save_path=save_path,
            title=f"{case_type.upper()}: {sample_id}",
        )

