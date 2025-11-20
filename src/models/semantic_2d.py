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
rbA_path = os.path.join(os.path.dirname(__file__), '../../RbA')
sys.path.insert(0, rbA_path)
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from mask2former import add_maskformer2_config
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead

# 导入RbA的setup函数（用于正确加载配置）
try:
    # 需要先设置detectron2的默认设置
    from detectron2.utils.logger import setup_logger
    from detectron2.utils import comm
    from detectron2.engine import default_setup
    from detectron2.projects.deeplab import add_deeplab_config
    
    # 初始化comm（单GPU模式）
    if not comm.is_main_process():
        comm.init_process_group("gloo", init_method="tcp://localhost:23456", rank=0, world_size=1)
    
    def rbA_setup(args):
        """RbA风格的配置加载"""
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        if hasattr(args, 'opts') and args.opts:
            cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
        return cfg
    
    try:
        from easydict import EasyDict as edict
    except ImportError:
        # 如果没有easydict，使用普通字典
        class edict(dict):
            def __getattr__(self, name):
                return self[name]
            def __setattr__(self, name, value):
                self[name] = value
    USE_RBA_SETUP = True
except Exception as e:
    print(f"警告: 无法使用RbA setup函数 ({e})，将使用手动配置加载")
    USE_RBA_SETUP = False


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
        
        # 使用RbA的setup函数加载配置（推荐方法）
        if USE_RBA_SETUP:
            print("使用RbA的setup函数加载配置")
            args = edict({
                'config_file': config_path,
                'eval-only': True,
                'opts': ['OUTPUT_DIR', 'output/']
            })
            self.cfg = rbA_setup(args)
        else:
            # 回退到手动加载配置
            print("使用手动配置加载")
            self.cfg = get_cfg()
            add_maskformer2_config(self.cfg)
            try:
                self.cfg.merge_from_file(config_path)
            except Exception as e:
                print(f"警告: 配置加载失败 ({e})")
                raise
        
        self.cfg.freeze()
        
        # 构建模型
        print(f"构建模型，META_ARCHITECTURE={self.cfg.MODEL.META_ARCHITECTURE}")
        if USE_RBA_SETUP:
            from train_net import Trainer
            self.model = Trainer.build_model(self.cfg)
        else:
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
        
        # 设置为评估模式，避免计算损失
        self.model.eval()
        # 确保criterion不会在推理时被调用
        if hasattr(self.model, 'criterion') and self.model.criterion is not None:
            # 临时禁用criterion，避免在推理时计算损失
            self._original_criterion = self.model.criterion
            self.model.criterion = None
    
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
        # 确保在eval模式下运行，避免计算损失
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            # 准备输入（Detectron2格式）
            inputs = [{"image": img} for img in images]
            
            # 前向传播（在eval模式下，不会计算损失）
            outputs = self.model(inputs)
        
        # 恢复训练状态（如果需要）
        if was_training:
            self.model.train()
            
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
        
        修复说明：
        - RbA (Rejected by All) 的核心思想：异常区域被所有已知类别拒绝
        - 对于已知类，logits应该很高；对于未知/异常类，所有logits都很低
        - 需要明确已知类集合，但这里使用通用方法：低logits表示异常
        
        RbA评分公式（改进版）：
        - 方法1：-sum(tanh(logits)) - 所有logits都很低时，分数高（异常）
        - 方法2：-max(logits) - 最大logit很低时，分数高（异常）
        - 方法3：使用softmax后的熵 - 熵高表示不确定性高（异常）
        
        Args:
            logits: 语义分割logits (B, K, H, W)，K是类别数
        
        Returns:
            RbA异常评分 (B, H, W)，值越大表示越异常
        """
        # 修复：使用多种方法组合计算RbA评分
        # 方法1：所有logits都很低时，表示被所有类别拒绝（异常）
        score_method1 = -torch.tanh(logits).sum(dim=1)  # (B, H, W)
        
        # 方法2：最大logit很低时，表示没有类别接受（异常）
        max_logits = torch.max(logits, dim=1)[0]  # (B, H, W)
        score_method2 = -max_logits
        
        # 方法3：使用softmax熵 - 熵高表示不确定性高（异常）
        probs = torch.softmax(logits, dim=1)  # (B, K, H, W)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # (B, H, W)
        score_method3 = entropy
        
        # 组合多种方法（归一化后加权平均）
        # 归一化到[0,1]范围
        score1_norm = (score_method1 - score_method1.min()) / (score_method1.max() - score_method1.min() + 1e-8)
        score2_norm = (score_method2 - score_method2.min()) / (score_method2.max() - score_method2.min() + 1e-8)
        score3_norm = (score_method3 - score_method3.min()) / (score_method3.max() - score_method3.min() + 1e-8)
        
        # 加权组合（可以根据实际效果调整权重）
        rba_score = 0.4 * score1_norm + 0.3 * score2_norm + 0.3 * score3_norm
        
        return rba_score
    
    def get_semantic_features(self, images: torch.Tensor) -> torch.Tensor:
        """获取2D语义特征（用于融合）"""
        return self.forward(images, return_features=True, return_rba_score=False)['features_2d']
    
    def get_rba_score(self, images: torch.Tensor) -> torch.Tensor:
        """获取RbA异常评分（用于基线）"""
        return self.forward(images, return_features=False, return_rba_score=True)['rba_score']

