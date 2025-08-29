import os
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning("safetensors not available, falling back to torch.load")

logger = logging.getLogger(__name__)


class ModelLoader:
    """通用模型加载器，支持多种模型格式"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            device: 设备类型 ('cpu', 'cuda', 'mps' 等)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_weights(self, 
                    model_path: Union[str, Path], 
                    model: Optional[torch.nn.Module] = None,
                    strict: bool = True,
                    **kwargs) -> Union[Dict[str, torch.Tensor], torch.nn.Module]:
        """
        加载模型权重
        
        Args:x
            model_path: 模型文件路径
            model: 要加载权重的模型实例（如果为None，只返回权重字典）
            strict: 是否严格匹配权重
            **kwargs: 传递给torch.load的额外参数
            
        Returns:
            如果提供了model，返回加载了权重的模型；否则返回权重字典
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 根据文件扩展名选择加载方法
        if model_path.suffix.lower() == '.safetensors':
            weights = self._load_safetensors(model_path)
        elif model_path.suffix.lower() in ['.pth', '.pt', '.ckpt']:
            weights = self._load_pytorch(model_path, **kwargs)
        else:
            # 尝试自动检测格式
            weights = self._auto_detect_and_load(model_path, **kwargs)
            
        if model is not None:
            # 加载权重到模型
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=strict)
            if missing_keys:
                logger.warning(f"缺失的键: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"意外的键: {unexpected_keys}")
            return model
        else:
            return weights
    
    def _load_safetensors(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """加载safetensors格式的模型"""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("需要安装safetensors库来加载.safetensors文件")
            
        weights = {}
        # 对于safetensors，先加载到CPU，然后移动到目标设备
        device_for_loading = "cpu" if self.device == "cpu" else "cpu"
        with safe_open(model_path, framework="pt", device=device_for_loading) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        
        # 如果需要，将权重移动到目标设备
        if self.device != "cpu":
            weights = {k: v.to(self.device) for k, v in weights.items()}
        
        return weights
    
    def _load_pytorch(self, model_path: Path, **kwargs) -> Dict[str, torch.Tensor]:
        """加载PyTorch格式的模型"""
        try:
            # 首先尝试weights_only=True（PyTorch 2.6+的默认行为）
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True, **kwargs)
        except Exception as e:
            logger.warning(f"使用weights_only=True加载失败: {e}")
            # 如果失败，尝试weights_only=False
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False, **kwargs)
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model' in checkpoint:
                return checkpoint['model']
            else:
                return checkpoint
        else:
            raise ValueError(f"不支持的checkpoint格式: {type(checkpoint)}")
    
    def _auto_detect_and_load(self, model_path: Path, **kwargs) -> Dict[str, torch.Tensor]:
        """自动检测文件格式并加载"""
        # 读取文件头部来判断格式
        with open(model_path, 'rb') as f:
            header = f.read(8)
            
        if header.startswith(b'PK'):
            # ZIP格式，可能是safetensors
            try:
                return self._load_safetensors(model_path)
            except:
                pass
                
        # 尝试作为PyTorch文件加载
        try:
            return self._load_pytorch(model_path, **kwargs)
        except Exception as e:
            raise ValueError(f"无法识别文件格式: {model_path}, 错误: {e}")
    
    def load_model_with_config(self, 
                              model_class: type,
                              model_path: Union[str, Path],
                              config: Optional[Dict[str, Any]] = None,
                              **kwargs) -> torch.nn.Module:
        """
        加载模型并应用配置
        
        Args:
            model_class: 模型类
            model_path: 模型文件路径
            config: 模型配置参数
            **kwargs: 传递给模型类的额外参数
            
        Returns:
            加载了权重的模型实例
        """
        config = config or {}
        
        # 创建模型实例
        model = model_class(**config, **kwargs)
        
        # 加载权重
        return self.load_weights(model_path, model, **kwargs)


# 便捷函数
def load_model_weights(model_path: Union[str, Path], 
                      device: Optional[str] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
    """便捷函数：只加载模型权重"""
    loader = ModelLoader(device=device)
    return loader.load_weights(model_path, **kwargs)


def load_model(model_class: type,
               model_path: Union[str, Path],
               config: Optional[Dict[str, Any]] = None,
               device: Optional[str] = None,
               **kwargs) -> torch.nn.Module:
    """便捷函数：加载完整的模型"""
    loader = ModelLoader(device=device)
    return loader.load_model_with_config(model_class, model_path, config, **kwargs) 


