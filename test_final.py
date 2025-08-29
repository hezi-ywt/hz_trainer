#!/usr/bin/env python3
"""
最终测试脚本
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.load_dmd_dataset import DMDDataLoader

def test_final():
    """最终测试"""
    
    print("开始最终测试...")
    
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
        
        # 检查是否所有tensor都在GPU上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_on_correct_device = True
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.device != device:
                    print(f"  警告: {key} 在设备 {value.device} 上，期望在 {device} 上")
                    all_on_correct_device = False
        
        if all_on_correct_device:
            print(f"\n✅ 所有tensor都在正确的设备 {device} 上!")
        else:
            print(f"\n❌ 有tensor在错误的设备上")
        
        print("\n最终测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final() 