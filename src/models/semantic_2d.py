"""
2D语义分支模块
实现基于冻结Mask2Former的语义特征提取
"""
import os
import torch
import torch.nn as nn
from typing import Optional
import sys

# 尝试导入detectron2，如果未安装则提供友好错误信息
try:
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.utils.logger import setup_logger
    DETECTRON2_AVAILABLE = True
    
    # 尝试导入Mask2Former（如果已安装）
    MASK2FORMER_AVAILABLE = False
    
    # 首先尝试从本地路径导入（如果Mask2Former在项目目录下）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mask2former_path = os.path.join(project_root, "Mask2Former")
    
    if os.path.exists(mask2former_path) and mask2former_path not in sys.path:
        sys.path.insert(0, mask2former_path)
    
    try:
        import mask2former  # Mask2Former需要单独安装
        MASK2FORMER_AVAILABLE = True
    except ImportError:
        # 如果从本地路径导入失败，尝试从系统路径导入
        try:
            import mask2former
            MASK2FORMER_AVAILABLE = True
        except ImportError:
            MASK2FORMER_AVAILABLE = False
            if not os.path.exists(mask2former_path):
                print(f"警告: mask2former未安装。Mask2Former目录不存在: {mask2former_path}")
                print("请按照以下步骤安装：")
                print("1. git clone https://github.com/facebookresearch/Mask2Former.git")
                print("2. 将Mask2Former目录放在项目根目录下")
                print("3. cd Mask2Former && pip install -e .")
                print("4. 如果需要GPU，编译CUDA kernel: cd mask2former/modeling/pixel_decoder/ops && sh make.sh")
        
except ImportError:
    DETECTRON2_AVAILABLE = False
    MASK2FORMER_AVAILABLE = False
    print("警告: detectron2未安装。请参考requirements.txt中的说明安装detectron2。")


class Semantic2DBranch(nn.Module):
    """
    2D语义分支，基于冻结的Mask2Former模型
    
    该类封装了Mask2Former模型，用于提取图像的语义特征。
    所有模型参数被冻结，仅作为特征提取器使用。
    
    Attributes:
        config_path: Detectron2配置文件路径
        checkpoint_path: 预训练权重文件路径
        freeze_backbone: 是否冻结backbone参数
        feature_dim: 输出特征维度，默认256
        device: 计算设备（'cuda'或'cpu'）
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        freeze_backbone: bool = True,
        feature_dim: int = 256,
        device: str = "cuda"
    ):
        """
        初始化Semantic2DBranch
        
        Args:
            config_path: Detectron2配置文件路径
            checkpoint_path: 预训练权重文件路径
            freeze_backbone: 是否冻结所有参数，默认True
            feature_dim: 输出特征维度，默认256
            device: 计算设备，默认'cuda'
            
        Raises:
            FileNotFoundError: 如果配置文件或权重文件不存在
            ImportError: 如果detectron2未安装
        """
        super(Semantic2DBranch, self).__init__()
        
        if not DETECTRON2_AVAILABLE:
            raise ImportError(
                "detectron2未安装。请运行以下命令安装：\n"
                "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html\n"
                "（请根据您的CUDA版本调整URL）"
            )
        
        if not MASK2FORMER_AVAILABLE:
            raise ImportError(
                "mask2former未安装。Mask2Former需要单独安装。\n"
                "请运行以下命令安装：\n"
                "pip install git+https://github.com/facebookresearch/Mask2Former.git\n"
                "或者克隆仓库后安装：\n"
                "git clone https://github.com/facebookresearch/Mask2Former.git\n"
                "cd Mask2Former && pip install -e ."
            )
        
        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"权重文件不存在: {checkpoint_path}\n"
                f"请运行以下命令下载权重文件：\n"
                f"  python scripts/download_mask2former_weights.py\n"
                f"或手动下载：\n"
                f"  https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_064788.pkl"
            )
        
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.freeze_backbone = freeze_backbone
        self.feature_dim = feature_dim
        self.device = device
        
        # 设置detectron2日志
        setup_logger()
        
        # 加载配置
        self.cfg = get_cfg()
        
        # 添加Mask2Former的配置（如果可用）
        if MASK2FORMER_AVAILABLE:
            try:
                from mask2former.config import add_maskformer2_config
                add_maskformer2_config(self.cfg)
            except (ImportError, AttributeError) as e:
                print(f"警告: 无法加载Mask2Former配置: {e}")
                print("将使用基本配置，某些功能可能不可用")
        
        # 合并配置文件（只包含基本配置项）
        self.cfg.merge_from_file(config_path)
        
        # 设置权重路径和设备
        self.cfg.MODEL.WEIGHTS = checkpoint_path
        self.cfg.MODEL.DEVICE = device
        
        # 注意：Swin Transformer的详细配置（如EMBED_DIM, DEPTHS等）会从权重文件中加载
        # 这些配置在训练时已经保存在权重文件中，不需要在配置文件中指定
        
        # 构建模型
        self.model = build_model(self.cfg)
        self.model.eval()
        
        # 加载权重
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(checkpoint_path)
        
        # 冻结参数
        if freeze_backbone:
            self._freeze_parameters()
        
        # 注册hook以提取Pixel Decoder特征
        self.pixel_decoder_features = None
        self._register_hooks()
        
        # 构建维度适配器（如果需要）
        self.dim_adapter = None
        self._build_dimension_adapter()
        
        # 移动到指定设备
        self.model = self.model.to(device)
    
    def _freeze_parameters(self):
        """冻结所有模型参数"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        """注册forward hook以提取Pixel Decoder输出特征"""
        def hook_fn(module, input, output):
            """Hook函数，捕获Pixel Decoder的输出"""
            # output可能是tuple或tensor，需要处理
            if isinstance(output, tuple):
                self.pixel_decoder_features = output[0]
            else:
                self.pixel_decoder_features = output
        
        # 查找Pixel Decoder模块
        # 在Mask2Former中，Pixel Decoder通常在sem_seg_head中
        if hasattr(self.model, 'sem_seg_head'):
            pixel_decoder = getattr(self.model.sem_seg_head, 'pixel_decoder', None)
            if pixel_decoder is not None:
                pixel_decoder.register_forward_hook(hook_fn)
            else:
                # 如果找不到pixel_decoder，尝试从sem_seg_head的最后一层获取特征
                # 这需要根据实际的模型结构调整
                print("警告: 无法找到pixel_decoder，将使用backbone的res5特征")
                self.pixel_decoder_features = None
    
    def _build_dimension_adapter(self):
        """
        构建维度适配器，将特征维度调整到指定维度
        
        注意：由于我们使用hook提取特征，实际的维度适配将在forward中动态处理
        """
        # 这里先不创建，等forward中确定实际维度后再创建
        pass
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播，提取语义特征
        
        Args:
            images: 输入图像张量 (B, C, H, W)，已预处理
            
        Returns:
            torch.Tensor: 特征图 (B, feature_dim, H', W')
        """
        # 确保输入在正确的设备上
        if images.device != self.device:
            images = images.to(self.device)
        
        # 重置特征缓存
        self.pixel_decoder_features = None
        
        # 前向传播
        with torch.no_grad():
            # 准备输入格式（detectron2需要的格式）
            inputs = [{"image": img} for img in images]
            
            # 通过backbone提取特征
            features = self.model.backbone(images)
            
            # 如果hook没有捕获到特征，使用backbone的res5特征
            if self.pixel_decoder_features is None:
                # 使用res5特征（最高层特征）
                pixel_features = features['res5']
            else:
                pixel_features = self.pixel_decoder_features
            
            # 获取特征维度
            _, c, h, w = pixel_features.shape
            
            # 如果维度不匹配，创建或使用维度适配器
            if c != self.feature_dim:
                if self.dim_adapter is None or self.dim_adapter.in_channels != c:
                    self.dim_adapter = nn.Conv2d(
                        in_channels=c,
                        out_channels=self.feature_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    ).to(self.device)
                    # 维度适配器也需要冻结（如果freeze_backbone为True）
                    if self.freeze_backbone:
                        for param in self.dim_adapter.parameters():
                            param.requires_grad = False
                
                pixel_features = self.dim_adapter(pixel_features)
            
            return pixel_features
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取特征的别名方法
        
        Args:
            images: 输入图像张量 (B, C, H, W)
            
        Returns:
            torch.Tensor: 特征图 (B, feature_dim, H', W')
        """
        return self.forward(images)

