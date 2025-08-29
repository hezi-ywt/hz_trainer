#!/usr/bin/env python3
"""
测试collate函数
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.load_dmd_dataset import DMDDataLoader

def test_collate():
    """测试collate函数"""
    
    data_dir = "/mnt/hz_trainer/batch_output_20250827_112221"
    
    print("开始测试collate函数...")
    
    try:
        # 创建数据加载器
        dataloader = DMDDataLoader.create_steps_dataloader(
            data_dir=data_dir,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            max_samples=2
        )
        
        print(f"数据加载器创建成功")
        
        # 测试第一个批次
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n批次 {batch_idx}:")
            print(f"  数据键: {list(batch.keys())}")
            
            # 检查数据类型
            for key, value in batch.items():
                if key == 'image':
                    continue
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: tensor shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f"  {key}: list length={len(value)}")
                else:
                    print(f"  {key}: type={type(value)}")
            
            # 只测试第一个批次
            break
        
        print("\ncollate函数测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_collate() 