#!/usr/bin/env python3
"""
轻量级检查点回调 - 专门为DMD训练优化
"""

import os
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from common.logging import logger

class LightweightCheckpoint(ModelCheckpoint):
    """轻量级检查点回调"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_weights_only = True
    
    @rank_zero_only
    def _save_checkpoint(self, trainer, filepath):
        """重写保存方法，只保存self.model的权重"""
        try:
            # 获取Lumina2ModelDMD中的self.model
            # 由于DMDTrainer继承了Lumina2ModelDMD，所以trainer.lightning_module就是Lumina2ModelDMD本身
            lumina_model = trainer.lightning_module
            model = lumina_model.model  # 这是实际的DiT模型
            
            # 只保存模型权重
            state_dict = {}
            
            # 如果是DeepSpeed模型，需要特殊处理
            if hasattr(model, '_deepspeed_engine'):
                from deepspeed import zero
                with zero.GatheredParameters(model.parameters()):
                    state_dict = model.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 保存为safetensors格式（更小）
            from safetensors.torch import save_file
            save_path = filepath.replace('.ckpt', '.safetensors')
            
            # 添加元数据 - 确保所有值都是字符串类型
            metadata = {
                "epoch": str(trainer.current_epoch),
                "global_step": str(trainer.global_step),
                "model_type": "lumina2_dit_model",
                "version": "1.0",
                "model_name": "NextDiT_2B_GQA_patch2_Adaln_Refiner"
            }
            
            save_file(state_dict, save_path, metadata=metadata)
            
            # 同时保存一个小的.ckpt文件用于Lightning恢复
            torch.save({
                "state_dict": state_dict,
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "model_type": "lumina2_dit_model",
                "model_name": "NextDiT_2B_GQA_patch2_Adaln_Refiner"
            }, filepath)
            
            logger.info(f"保存模型权重: {save_path} ({len(state_dict)} 个参数)")
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            # 回退到默认保存方法
            super()._save_checkpoint(trainer, filepath) 