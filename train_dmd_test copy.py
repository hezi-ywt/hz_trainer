#!/usr/bin/env python3
"""
DMD (Diffusion Model Distillation) 测试训练脚本
"""

import os
import sys
import time
import argparse
import random
from lightning.pytorch.strategies import DeepSpeedStrategy
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lumina2_model_dmd import Lumina2ModelDMD
from dataset.load_dmd_dataset import DMDDataLoader
from common.logging import logger

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DMDTrainer(Lumina2ModelDMD):
    """DMD训练器"""
    
    def __init__(self, config ,device="cuda"):
        super().__init__(config,device)
        self.config = config
        self.save_hyperparameters()
                
        # 训练参数
        self.learning_rate = config.trainer.learning_rate
        self.weight_decay = config.trainer.get("weight_decay", 0.01)
        
    # def forward(self, batch):
    #     return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        loss = result["loss"]
        self.log('train_loss', loss, prog_bar=True)
        self.log('task_loss', result["task_loss"], prog_bar=False)
        
        # 添加梯度裁剪
        if hasattr(self.config.trainer, 'grad_clip') and self.config.trainer.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=self.config.trainer.grad_clip
            )
            self.log("grad_norm", grad_norm, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            result = self.forward(batch)
        loss = result["loss"]
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_task_loss', result["task_loss"], prog_bar=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.trainer.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

def main():
    parser = argparse.ArgumentParser(description="DMD训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 创建输出目录
    os.makedirs(config.trainer.checkpoint_dir, exist_ok=True)
    os.makedirs(config.trainer.log_dir, exist_ok=True)
    
    # 创建数据加载器
    train_dataloader = DMDDataLoader.create_steps_dataloader(
        data_dir=config.data.train_data_dir,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        max_samples=config.data.get("max_train_samples", None)
    )
    
    # 创建训练器
    trainer = DMDTrainer(config)
    
    # 创建回调
    from lightweight_checkpoint import LightweightCheckpoint
    callbacks = [
        LightweightCheckpoint(
            dirpath=config.trainer.checkpoint_dir,
            filename='dmd_model_{epoch:02d}_{step:06d}_{val_loss:.4f}',
            # every_n_train_steps=100,
            every_n_epochs=1,
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    # 创建Lightning训练器
    from lightning.fabric.strategies import DeepSpeedStrategy
    lightning_trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        callbacks=callbacks,
        log_every_n_steps=5,
        strategy=DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,  # Enable CPU Offloading
        cpu_checkpointing=True,  # (Optional) offload activations to CPU
    ),

        gradient_clip_val=config.trainer.get("grad_clip", 1.0),
        accumulate_grad_batches=config.trainer.get("accumulate_grad_batches", 4),  # 梯度累积
        # 显存优化设置
        sync_batchnorm=False,

    )
    

    # 开始训练
    logger.info("开始DMD训练...")
    
    try:
        if args.resume:
            lightning_trainer.fit(trainer, train_dataloader, ckpt_path=args.resume)
        else:
            lightning_trainer.fit(trainer, train_dataloader)
        
        logger.info("训练完成!")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 