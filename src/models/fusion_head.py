"""
轻量级融合头模块
实现2D和3D特征的融合，输出异常概率图

核心设计：
1. 适配器（Adapter）：解决2D和3D特征的语义不对齐问题
2. 门控机制（Gating）：动态决定信任图像还是点云
3. 异常判定：基于"语义-几何不一致性"输出异常概率图
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FusionHead(nn.Module):
    """
    轻量级融合头
    
    架构设计（瓶颈-残差结构）：
    - 输入：F_img (H, W, C_img) + F_pts (H, W, C_pts)
    - 特征交互层：交叉注意力或1x1卷积
    - 瓶颈结构：降维 -> 特征交互 -> 升维
    - 门控机制：学习动态权重
    - 输出：异常概率图 (H, W, 1)
    
    参数量：<5% 总参数量（符合PEFT原则）
    """
    
    def __init__(
        self,
        img_feature_dim: int = 256,
        pts_feature_dim: int = 128,
        hidden_dim: int = 64,
        use_cross_attention: bool = False,
        use_gating: bool = True,
        device: str = "cuda"
    ):
        """
        初始化融合头
        
        Args:
            img_feature_dim: 图像特征维度，默认256（Mask2Former输出）
            pts_feature_dim: 点云特征维度，默认128（MinkUNet输出）
            hidden_dim: 隐藏层维度（瓶颈维度），默认64
            use_cross_attention: 是否使用交叉注意力，默认False（使用1x1卷积）
            use_gating: 是否使用门控机制，默认True
            device: 计算设备
        """
        super(FusionHead, self).__init__()
        
        self.img_feature_dim = img_feature_dim
        self.pts_feature_dim = pts_feature_dim
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.use_gating = use_gating
        self.device = device
        
        # 维度对齐层（适配器）
        # 将2D和3D特征对齐到相同维度
        self.img_proj = nn.Conv2d(img_feature_dim, hidden_dim, 1)
        self.pts_proj = nn.Conv2d(pts_feature_dim, hidden_dim, 1)
        
        # 特征交互层
        if use_cross_attention:
            # 交叉注意力机制（图像查询点云）
            self.cross_attention = CrossAttention(hidden_dim)
        else:
            # 简单的1x1卷积交互
            self.interaction = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        
        # 瓶颈结构：降维 -> 特征提取 -> 升维
        self.bottleneck = nn.Sequential(
            # 降维
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            
            # 特征提取（3x3卷积）
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            
            # 升维
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 门控机制（学习动态权重）
        if use_gating:
            self.gate = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
                nn.Sigmoid()  # 输出0-1之间的权重
            )
        
        # 异常判定层（输出异常概率）
        self.decision_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),  # 输出单通道异常概率图
            nn.Sigmoid()  # 输出0-1之间的概率
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # 对于Sigmoid之前的层，使用较小的初始化
                if 'gate' in name or 'decision_head' in name:
                    # Gating和Decision Head：初始化为较小的值，避免Sigmoid饱和
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        # 对于Gating：偏置设为0（初始权重0.5，公平）
                        # 对于Decision Head：偏置设为负值（初始输出低概率，因为大部分是背景）
                        if 'gate' in name:
                            nn.init.constant_(m.bias, 0.0)
                        elif 'decision_head' in name and len(self.decision_head) > 0:
                            # Decision Head的最后一层：偏置设为负值
                            if m == list(self.decision_head.modules())[-2]:  # 倒数第二层（Sigmoid之前）
                                nn.init.constant_(m.bias, -2.0)  # 初始输出低概率
                            else:
                                nn.init.constant_(m.bias, 0.0)
                        else:
                            nn.init.constant_(m.bias, 0.0)
                else:
                    # 其他层：使用Kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        img_features: torch.Tensor,
        pts_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：融合2D和3D特征，输出异常概率图
        
        Args:
            img_features: 图像特征 (B, C_img, H, W) 或 (H, W, C_img)
            pts_features: 点云投影特征 (B, C_pts, H, W) 或 (H, W, C_pts)
            
        Returns:
            anomaly_map: 异常概率图 (B, 1, H, W) 或 (H, W, 1)
        """
        # 处理输入格式（支持CHW和HWC）
        if img_features.dim() == 3:
            # (H, W, C) -> (1, C, H, W)
            img_features = img_features.permute(2, 0, 1).unsqueeze(0)
            pts_features = pts_features.permute(2, 0, 1).unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C_img, H, W = img_features.shape
        _, C_pts, _, _ = pts_features.shape
        
        # 1. 维度对齐（适配器）
        img_feat = self.img_proj(img_features)  # (B, hidden_dim, H, W)
        pts_feat = self.pts_proj(pts_features)  # (B, hidden_dim, H, W)
        
        # 2. 特征交互
        if self.use_cross_attention:
            # 交叉注意力：图像特征查询点云特征
            fused_feat = self.cross_attention(img_feat, pts_feat)
        else:
            # 简单拼接 + 1x1卷积
            concat_feat = torch.cat([img_feat, pts_feat], dim=1)  # (B, 2*hidden_dim, H, W)
            fused_feat = self.interaction(concat_feat)  # (B, hidden_dim, H, W)
        
        # 3. 瓶颈结构（特征提取）
        bottleneck_feat = self.bottleneck(fused_feat)  # (B, hidden_dim, H, W)
        
        # 4. 门控机制（学习动态权重）
        if self.use_gating:
            # 计算门控权重
            gate_input = torch.cat([img_feat, pts_feat], dim=1)  # (B, 2*hidden_dim, H, W)
            gate_weight = self.gate(gate_input)  # (B, hidden_dim, H, W)
            
            # 应用门控：融合原始特征和瓶颈特征
            gated_feat = gate_weight * fused_feat + (1 - gate_weight) * bottleneck_feat
        else:
            gated_feat = bottleneck_feat
        
        # 5. 异常判定（输出异常概率图）
        anomaly_map = self.decision_head(gated_feat)  # (B, 1, H, W)
        
        # 数值稳定性：clamp输出，避免极端值
        anomaly_map = torch.clamp(anomaly_map, min=1e-6, max=1.0 - 1e-6)
        
        # 恢复输出格式
        if squeeze_output:
            anomaly_map = anomaly_map.squeeze(0).permute(1, 2, 0)  # (H, W, 1)
        
        return anomaly_map


class CrossAttention(nn.Module):
    """
    交叉注意力模块
    
    让图像特征查询点云特征，学习跨模态的语义对齐
    """
    
    def __init__(self, dim: int):
        super(CrossAttention, self).__init__()
        self.dim = dim
        
        # Query来自图像特征，Key和Value来自点云特征
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.output = nn.Conv2d(dim, dim, 1)
        
        self.scale = dim ** -0.5
    
    def forward(self, img_feat: torch.Tensor, pts_feat: torch.Tensor) -> torch.Tensor:
        """
        交叉注意力：图像特征查询点云特征
        
        Args:
            img_feat: 图像特征 (B, C, H, W)
            pts_feat: 点云特征 (B, C, H, W)
            
        Returns:
            fused_feat: 融合后的特征 (B, C, H, W)
        """
        B, C, H, W = img_feat.shape
        
        # 生成Query, Key, Value
        Q = self.query(img_feat)  # (B, C, H, W)
        K = self.key(pts_feat)    # (B, C, H, W)
        V = self.value(pts_feat)  # (B, C, H, W)
        
        # 展平空间维度
        Q = Q.flatten(2).transpose(1, 2)  # (B, HW, C)
        K = K.flatten(2).transpose(1, 2)  # (B, HW, C)
        V = V.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        # 计算注意力分数
        attn = (Q @ K.transpose(1, 2)) * self.scale  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = attn @ V  # (B, HW, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        
        # 输出投影
        fused_feat = self.output(out) + img_feat  # 残差连接
        
        return fused_feat

