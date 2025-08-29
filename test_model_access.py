#!/usr/bin/env python3
"""
测试模型访问
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from train_dmd_test import DMDTrainer

def test_model_access():
    """测试模型访问"""
    print("测试模型访问...")
    
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
            "learning_rate": 1e-5,
            "max_epochs": 1
        }
    })
    
    try:
        # 创建训练器
        trainer = DMDTrainer(config)
        print("✅ DMDTrainer创建成功")
        
        # 测试模型访问
        print(f"trainer类型: {type(trainer)}")
        print(f"trainer.model类型: {type(trainer.model)}")
        print(f"trainer.model参数数量: {trainer.model.parameter_count():,}")
        
        # 测试state_dict访问
        state_dict = trainer.model.state_dict()
        print(f"state_dict键数量: {len(state_dict)}")
        
        print("✅ 模型访问测试成功")
        
    except Exception as e:
        print(f"❌ 模型访问测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_access() 