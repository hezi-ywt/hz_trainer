#!/usr/bin/env python3
"""
测试完整的数据加载和模型前向传播流程
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.load_dmd_dataset import DMDDataLoader
from lumina2_model_dmd import Lumina2ModelDMD

def test_full_pipeline():
    """测试完整流程"""
    
    print("开始测试完整流程...")
    
    # 创建配置
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
    
    try:
        # 创建数据加载器
        dataloader = DMDDataLoader.create_steps_dataloader(
            data_dir="/mnt/hz_trainer/batch_output_20250827_112221",
            batch_size=1,
            shuffle=False,
            num_workers=0,
            max_samples=1
        )
        
        print("数据加载器创建成功")
        
        # 创建模型
        model = Lumina2ModelDMD(config, device="cuda")
        print("模型创建成功")
        
        # 获取一个batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        
        # 检查数据形状
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 测试前向传播
        with torch.no_grad():
            loss = model(batch)
            print(f"前向传播成功，损失: {loss.item()}")
        
        print("完整流程测试成功!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_pipeline() 