#!/usr/bin/env python3
"""
测试检查点保存功能
"""

import os
import torch
from safetensors.torch import save_file

def test_safetensors_metadata():
    """测试safetensors元数据保存"""
    print("测试safetensors元数据保存...")
    
    # 创建测试数据
    test_state_dict = {
        "test_param": torch.randn(10, 10)
    }
    
    # 正确的元数据（字符串类型）
    metadata = {
        "epoch": "1",
        "global_step": "100",
        "model_type": "test_model",
        "version": "1.0"
    }
    
    try:
        save_file(test_state_dict, "test_checkpoint.safetensors", metadata=metadata)
        print("✅ safetensors保存成功")
        
        # 清理测试文件
        os.remove("test_checkpoint.safetensors")
        
    except Exception as e:
        print(f"❌ safetensors保存失败: {e}")

def test_lightweight_checkpoint():
    """测试轻量级检查点功能"""
    print("\n测试轻量级检查点功能...")
    
    try:
        from lightweight_checkpoint import LightweightCheckpoint
        print("✅ 轻量级检查点导入成功")
        
        # 创建检查点回调
        checkpoint = LightweightCheckpoint(
            dirpath="./test_checkpoints",
            filename="test_model_{epoch:02d}",
            every_n_epochs=1
        )
        print("✅ 检查点回调创建成功")
        
    except Exception as e:
        print(f"❌ 轻量级检查点测试失败: {e}")

if __name__ == "__main__":
    test_safetensors_metadata()
    test_lightweight_checkpoint()
    print("\n测试完成!") 