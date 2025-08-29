#!/usr/bin/env python3
"""
自定义检查点回调 - 只保存模型权重，不保存优化器状态
"""

import os
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from common.logging import logger

class LightweightModelCheckpoint(ModelCheckpoint):
    """轻量级模型检查点回调"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_weights_only = True
    
    @rank_zero_only
    def _save_checkpoint(self, trainer, filepath):
        """重写保存方法，只保存模型权重"""
        try:
            # 获取模型
            model = trainer.lightning_module.model
            
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
            
            # 添加元数据
            metadata = {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "model_type": "lumina2_dmd",
                "version": "1.0"
            }
            
            save_file(state_dict, save_path, metadata=metadata)
            
            # 同时保存一个小的.ckpt文件用于Lightning恢复
            torch.save({
                "state_dict": state_dict,
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "model_type": "lumina2_dmd"
            }, filepath)
            
            logger.info(f"保存轻量级检查点: {save_path} ({len(state_dict)} 个参数)")
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            # 回退到默认保存方法
            super()._save_checkpoint(trainer, filepath) 