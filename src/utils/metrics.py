"""
评估指标：AUROC, AP, FPR@95等
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_auroc(pred_scores: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    计算AUROC
    
    Args:
        pred_scores: 预测异常分数 (N,)
        gt_labels: 真实标签 (N,)，0=正常，1=异常
    
    Returns:
        AUROC值
    """
    if len(np.unique(gt_labels)) < 2:
        return 0.0
    return roc_auc_score(gt_labels, pred_scores)


def compute_ap(pred_scores: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    计算Average Precision (AP)
    
    Args:
        pred_scores: 预测异常分数 (N,)
        gt_labels: 真实标签 (N,)，0=正常，1=异常
    
    Returns:
        AP值
    """
    if len(np.unique(gt_labels)) < 2:
        return 0.0
    return average_precision_score(gt_labels, pred_scores)


def compute_fpr_at_95_tpr(pred_scores: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    计算FPR@95（在TPR=95%时的假阳性率）
    
    Args:
        pred_scores: 预测异常分数 (N,)
        gt_labels: 真实标签 (N,)，0=正常，1=异常
    
    Returns:
        FPR@95值
    """
    if len(np.unique(gt_labels)) < 2:
        return 1.0
    
    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
    
    # 找到TPR >= 0.95的第一个阈值
    tpr_95_idx = np.where(tpr >= 0.95)[0]
    if len(tpr_95_idx) == 0:
        return 1.0
    
    return fpr[tpr_95_idx[0]]


def compute_metrics(pred_scores: np.ndarray, gt_labels: np.ndarray) -> dict:
    """
    计算所有评估指标
    
    Args:
        pred_scores: 预测异常分数 (N,)
        gt_labels: 真实标签 (N,)，0=正常，1=异常
    
    Returns:
        Dict包含所有指标
    """
    # 转换为numpy数组
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.cpu().numpy()
    
    # 展平
    pred_scores = pred_scores.flatten()
    gt_labels = gt_labels.flatten()
    
    metrics = {
        'auroc': compute_auroc(pred_scores, gt_labels),
        'ap': compute_ap(pred_scores, gt_labels),
        'fpr_at_95_tpr': compute_fpr_at_95_tpr(pred_scores, gt_labels),
    }
    
    return metrics

