#!/usr/bin/env python3
"""
简单的数据加载测试
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.load_dmd_dataset import DMDStepsDataset

def test_simple():
    """简单测试数据加载"""
    
    data_dir = "/mnt/hz_trainer/batch_output_20250827_112221"
    
    print("开始简单测试...")
    
    try:
        # 创建数据集
        dataset = DMDStepsDataset(
            data_dir=data_dir,
            load_images=True,
            load_steps_data=True,
            max_samples=2  # 只测试2个样本
        )
        
        print(f"数据集创建成功，共有 {len(dataset)} 个样本")
        
        # 测试第一个样本
        sample = dataset[0]
        print(f"\n第一个样本:")
        print(f"  文件名: {sample['filename']}")
        print(f"  数据键: {list(sample.keys())}")
        
        if 'steps_data' in sample:
            steps_data = sample['steps_data']
            print(f"  steps_data长度: {len(steps_data)}")
            
            if len(steps_data) > 0:
                first_step = steps_data[0]
                print(f"  第一个step的键: {list(first_step.keys())}")
                
                if 'data' in first_step:
                    step_data = first_step['data']
                    print(f"  step_data的键: {list(step_data.keys())}")
                    
                    # 检查关键数据
                    for key in ['sigma', 'denoised', 'cap_feats', 'cap_mask']:
                        if key in step_data:
                            data = step_data[key]
                            if hasattr(data, 'shape'):
                                print(f"    {key}: shape={data.shape}")
                            else:
                                print(f"    {key}: type={type(data)}")
                        else:
                            print(f"    {key}: 不存在")
        
        print("\n简单测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple() 