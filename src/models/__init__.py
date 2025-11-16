# 修复：延迟导入，避免detectron2依赖问题
# 只有在实际使用时才导入，允许单独使用3D分支而不需要detectron2

def _lazy_import_semantic_2d():
    """延迟导入2D语义分支（需要detectron2）"""
    try:
        from .semantic_2d import Semantic2DBranch
        return Semantic2DBranch
    except ImportError as e:
        raise ImportError(
            f"Semantic2DBranch requires detectron2. Install it with: "
            f"pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
        ) from e

# 直接导入3D和融合模块（不依赖detectron2）
from .geometric_3d import Geometric3DBranch
from .fusion import FusionHead, FusionModel

# 延迟导入2D模块
def __getattr__(name):
    if name == 'Semantic2DBranch':
        return _lazy_import_semantic_2d()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'Semantic2DBranch',
    'Geometric3DBranch',
    'FusionHead',
    'FusionModel',
]

