#!/usr/bin/env python3
"""
测试数据加载功能
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.load_dmd_dataset import DMDDataLoader
from common.logging import logger

def test_data_loading():
    """测试数据加载功能"""
    
    # 创建测试配置
    config = OmegaConf.create({
        "data": {
            "train_data_dir": "/mnt/hz_trainer/batch_output_20250827_112221",
            "max_train_samples": 5  # 只测试5个样本
        },
        "trainer": {
            "batch_size": 2,
            "num_workers": 0
        }
    })
    
    print("开始测试数据加载...")
    
    try:
        # 创建数据加载器
        dataloader = DMDDataLoader.create_steps_dataloader(
            data_dir=config.data.train_data_dir,
            batch_size=config.trainer.batch_size,
            shuffle=False,
            num_workers=config.trainer.num_workers,
            max_samples=config.data.max_train_samples
        )
        
        print(f"数据加载器创建成功，共有 {len(dataloader)} 个批次")
        
        # 测试加载第一个批次
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n批次 {batch_idx}:")
            print(f"  批次大小: {len(batch['sigmas'])}")
            print(f"  数据键: {list(batch.keys())}")
            
            # 检查关键数据
            if 'sigmas' in batch:
                print(f"  sigmas: {len(batch['sigmas'])} 个")
                if batch['sigmas']:
                    print(f"    第一个sigma: {batch['sigmas'][0]}")
            
            if 'cap_feats' in batch:
                print(f"  cap_feats: {len(batch['cap_feats'])} 个")
                if batch['cap_feats']:
                    feat = batch['cap_feats'][0]
                    if hasattr(feat, 'shape'):
                        print(f"    特征形状: {feat.shape}")
                    else:
                        print(f"    特征类型: {type(feat)}")
            
            if 'denoised' in batch:
                print(f"  denoised: {len(batch['denoised'])} 个")
                if batch['denoised']:
                    denoised = batch['denoised'][0]
                    if hasattr(denoised, 'shape'):
                        print(f"    去噪图像形状: {denoised.shape}")
                    else:
                        print(f"    去噪图像类型: {type(denoised)}")
            
            # 只测试前2个批次
            if batch_idx >= 1:
                break
        
        print("\n数据加载测试完成!")
        
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading() 