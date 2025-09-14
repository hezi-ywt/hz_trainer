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
import logging
from lightning.fabric.strategies import DeepSpeedStrategy
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from wandb.integration.lightning.fabric import WandbLogger
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
    seed_everything(seed)




class DMDTrainer(Lumina2ModelDMD):
    """DMD训练器"""
    
    def __init__(self, config ,device="cuda"):
        super().__init__(config,device)
        self.config = config
        self.save_hyperparameters()
                
        # 训练参数
        self.learning_rate = config.trainer.learning_rate
        self.weight_decay = config.trainer.get("weight_decay", 0.01)
        
    
    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        loss = result["loss"]
       
        metrics ={
            'train_loss': loss,
            'task_loss': result["task_loss"],
            'task_loss_cfg_distill': result["task_loss_cfg_distill"],
            't': result["t"]
        }
        self.fabric.log_dict(metrics, step=self.global_step)
        # 添加梯度裁剪
        if hasattr(self.config.trainer, 'grad_clip') and self.config.trainer.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=self.config.trainer.grad_clip
            )
            self.fabric.log("grad_norm", grad_norm, prog_bar=True)

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
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.trainer.max_epochs
        )
        return optimizer,scheduler
        
        
        
from fabric_trainer import FabricTrainer


def main():
    parser = argparse.ArgumentParser(description="DMD训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    


    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 设置随机种子
    if config.trainer.get("seed") is not None:
        set_seed(config.trainer.seed)
    
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
    
  
    # 配置DeepSpeed策略
    deepspeed_config_path = os.path.join(os.path.dirname(__file__), "deepspeed_config.json")
    deepspeed_strategy = DeepSpeedStrategy(
        config=deepspeed_config_path
    )
    
    # Configure Fabric
    trainer = FabricTrainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",  # 保持这个设置，DeepSpeed会处理精度转换
        strategy=DeepSpeedStrategy(stage=2),  # 使用配置好的DeepSpeed策略
        loggers=[CSVLogger(config.trainer.log_dir), TensorBoardLogger(config.trainer.log_dir)],
        max_epochs=config.trainer.get("max_epochs", 1000),
        max_steps=config.trainer.get("max_steps", None),
        grad_accum_steps=config.trainer.get("accumulate_grad_batches", 1),
        limit_train_batches=config.trainer.get("limit_train_batches", float("inf")),
        limit_val_batches=config.trainer.get("limit_val_batches",  float("inf")),
        checkpoint_dir=config.trainer.get("checkpoint_dir", "./checkpoints"),
        checkpoint_frequency=config.trainer.get("checkpoint_frequency", 1),
        save_every_n_steps=config.trainer.get("save_every_n_steps", None),
        save_every_n_epochs=config.trainer.get("save_every_n_epochs", 1),
        seed=config.trainer.get("seed", None),
    )

    print(config.trainer.get("save_every_n_steps", None))
    # Instantiate objects
    model = DMDTrainer(config)
    
    torch.set_float32_matmul_precision("high")
    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()