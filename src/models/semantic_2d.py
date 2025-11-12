"""
2D语义分支：基于Mask2Former的语义分割和RbA异常评分
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

# 添加RbA路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../RbA'))
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from mask2former import add_maskformer2_config
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead


class Semantic2DBranch(nn.Module):
    """
    2D语义分支模型封装
    
    功能：
    1. 使用Mask2Former进行语义分割
    2. 提取Pixel Decoder的2D特征图（用于融合）
    3. 计算RbA异常评分（用于基线）
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            config_path: Mask2Former配置文件路径
            checkpoint_path: 预训练模型权重路径
            freeze_backbone: 是否冻结主干网络
        """
        super().__init__()
        
        # 加载配置
        self.cfg = get_cfg()
        add_maskformer2_config(self.cfg)
        self.cfg.merge_from_file(config_path)
        self.cfg.freeze()
        
        # 构建模型
        self.model = build_model(self.cfg)
        
        # 加载权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        # 冻结主干网络（如果指定）
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        self.model.eval()
    
    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = True,
        return_rba_score: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 输入图像 (B, 3, H, W)
            return_features: 是否返回2D特征图
            return_rba_score: 是否返回RbA异常评分
        
        Returns:
            Dict包含：
            - 'sem_seg': 语义分割logits (B, K, H, W)
            - 'features_2d': 2D特征图 (B, C_2D, H, W) [可选]
            - 'rba_score': RbA异常评分 (B, H, W) [可选]
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            # 准备输入（Detectron2格式）
            inputs = [{"image": img} for img in images]
            
            # 前向传播
            outputs = self.model(inputs)
            
            results = {}
            
            # 提取语义分割logits
            sem_seg_logits = outputs[0]['sem_seg']  # (K, H, W)
            if len(sem_seg_logits.shape) == 3:
                sem_seg_logits = sem_seg_logits.unsqueeze(0)  # (1, K, H, W)
            results['sem_seg'] = sem_seg_logits
            
            # 提取2D特征图（用于融合）
            if return_features:
                features_2d = self._extract_pixel_decoder_features(images)
                results['features_2d'] = features_2d
            
            # 计算RbA异常评分（用于基线）
            if return_rba_score:
                rba_score = self._compute_rba_score(sem_seg_logits)
                results['rba_score'] = rba_score
            
            return results
    
    def _extract_pixel_decoder_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取Pixel Decoder的2D特征图
        
        Args:
            images: 输入图像 (B, 3, H, W)
        
        Returns:
            2D特征图 (B, C_2D, H, W)
        """
        # 准备输入
        inputs = [{"image": img} for img in images]
        
        # 获取backbone特征
        features = self.model.backbone(images)
        
        # 通过Pixel Decoder获取特征
        sem_seg_head = self.model.sem_seg_head
        if hasattr(sem_seg_head, 'pixel_decoder'):
            pixel_decoder = sem_seg_head.pixel_decoder
            # 调用forward_features获取特征
            mask_features, _, _ = pixel_decoder.forward_features(features)
            # mask_features shape: (B, mask_dim, H, W)
            return mask_features
        else:
            # 如果没有pixel_decoder，使用backbone的最后一层特征
            return list(features.values())[-1]
    
    def _compute_rba_score(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算RbA异常评分
        
        RbA评分公式：-sum(tanh(logits))
        
        Args:
            logits: 语义分割logits (B, K, H, W)
        
        Returns:
            RbA异常评分 (B, H, W)
        """
        # 应用tanh并求和
        rba_score = -torch.tanh(logits).sum(dim=1)  # (B, H, W)
        return rba_score
    
    def get_semantic_features(self, images: torch.Tensor) -> torch.Tensor:
        """获取2D语义特征（用于融合）"""
        return self.forward(images, return_features=True, return_rba_score=False)['features_2d']
    
    def get_rba_score(self, images: torch.Tensor) -> torch.Tensor:
        """获取RbA异常评分（用于基线）"""
        return self.forward(images, return_features=False, return_rba_score=True)['rba_score']

