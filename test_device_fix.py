#!/usr/bin/env python3
"""
测试设备问题修复
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.load_dmd_dataset import DMDDataLoader

def test_device_fix():
    """测试设备问题修复"""
    
    print("开始测试设备问题修复...")
    
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
        
        # 获取一个batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        
        # 检查数据形状和设备
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        
        # 测试移动到GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n测试移动到设备: {device}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_value = value.to(device)
                print(f"  {key}: 移动成功, device={moved_value.device}")
        
        print("\n设备问题修复测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_device_fix() 