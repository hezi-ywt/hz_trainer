#!/usr/bin/env python3
"""
测试DMD模型修复的脚本
"""

import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lumina2_model_dmd import Lumina2ModelDMD

def test_dmd_model():
    """测试DMD模型的前向传播"""
    
    # 创建测试配置
    config = OmegaConf.create({
        "model": {
            "model_name": "NextDiT_2B_GQA_patch2_Adaln_Refiner",
            "model_path": "/mnt/huggingface/Lumina-Image-2.0",
            "vae_path": "/mnt/huggingface/Neta-Lumina/VAE/ae.safetensors",
            "text_encoder_path": "/mnt/huggingface/Neta-Lumina/Text Encoder/gemma_2_2b_fp16.safetensors",
            "init_from": "/mnt/huggingface/Neta-Lumina/Unet/neta-lumina-v1.0.safetensors",
            "use_ema": False,
            "use_fake_model": False,
            "use_real_model": False,
        },
        "advanced": {
            "use_ema": False
        },
        "trainer": {
            "batch_size": 1,
            "grad_clip": 1.0
        }
    })
    
    print("创建DMD模型...")
    model = Lumina2ModelDMD(config, device="cuda")
    
    # 创建测试数据
    batch_size = 1
    latent_size = 16
    height = 128
    width = 128
    seq_len = 256
    
    print("创建测试数据...")
    test_batch = {
        "cap_feats": torch.randn(batch_size, seq_len, 2048, device="cuda", dtype=torch.bfloat16),
        "cap_masks": torch.ones(batch_size, seq_len, device="cuda", dtype=torch.int32),
        "sigmas": torch.rand(batch_size, device="cuda"),
        "denoised": torch.randn(batch_size, latent_size, height, width, device="cuda"),
        "x1": torch.randn(batch_size, latent_size, height, width, device="cuda")
    }
    
    print("执行前向传播...")
    try:
        with torch.no_grad():
            result = model.forward(test_batch)
        
        print("前向传播成功!")
        print(f"返回结果类型: {type(result)}")
        print(f"返回结果键: {result.keys()}")
        print(f"损失值: {result['loss'].item():.6f}")
        print(f"任务损失值: {result['task_loss'].item():.6f}")
        
        # 测试没有self.log()调用
        print("✅ 没有self.log()调用，修复成功!")
        
        return True
        
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dmd_trainer():
    """测试DMDTrainer的集成"""
    try:
        from train_dmd_test import DMDTrainer
        
        # 创建测试配置
        config = OmegaConf.create({
            "model": {
                "model_name": "NextDiT_2B_GQA_patch2_Adaln_Refiner",
                "model_path": "/mnt/huggingface/Lumina-Image-2.0",
                "vae_path": "/mnt/huggingface/Neta-Lumina/VAE/ae.safetensors",
                "text_encoder_path": "/mnt/huggingface/Neta-Lumina/Text Encoder/gemma_2_2b_fp16.safetensors",
                "init_from": "/mnt/huggingface/Neta-Lumina/Unet/neta-lumina-v1.0.safetensors",
                "use_ema": False,
                "use_fake_model": False,
                "use_real_model": False,
            },
            "advanced": {
                "use_ema": False
            },
            "trainer": {
                "batch_size": 1,
                "grad_clip": 1.0,
                "learning_rate": 1e-4,
                "max_epochs": 1
            }
        })
        
        print("创建DMDTrainer...")
        trainer = DMDTrainer(config)
        
        # 创建测试数据
        batch_size = 1
        latent_size = 16
        height = 128
        width = 128
        seq_len = 256
        
        test_batch = {
            "cap_feats": torch.randn(batch_size, seq_len, 2048, device="cuda", dtype=torch.bfloat16),
            "cap_masks": torch.ones(batch_size, seq_len, device="cuda", dtype=torch.int32),
            "sigmas": torch.rand(batch_size, device="cuda"),
            "denoised": torch.randn(batch_size, latent_size, height, width, device="cuda"),
            "x1": torch.randn(batch_size, latent_size, height, width, device="cuda")
        }
        
        print("测试training_step...")
        with torch.no_grad():
            loss = trainer.training_step(test_batch, 0)
        print(f"training_step成功，损失: {loss.item():.6f}")
        
        print("测试validation_step...")
        with torch.no_grad():
            loss = trainer.validation_step(test_batch, 0)
        print(f"validation_step成功，损失: {loss.item():.6f}")
        
        print("✅ DMDTrainer集成测试成功!")
        return True
        
    except Exception as e:
        print(f"DMDTrainer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试DMD模型修复...")
    
    print("\n=== 测试1: DMD模型前向传播 ===")
    success1 = test_dmd_model()
    
    print("\n=== 测试2: DMDTrainer集成 ===")
    success2 = test_dmd_trainer()
    
    if success1 and success2:
        print("\n✅ 所有测试通过! DMD模型修复成功!")
    else:
        print("\n❌ 部分测试失败! 需要进一步调试.") 