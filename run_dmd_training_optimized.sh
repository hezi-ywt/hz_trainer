#!/bin/bash

# 内存优化的DMD训练脚本

echo "🚀 开始内存优化的DMD训练..."

# 设置环境变量以优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 清理GPU内存
echo "🧹 清理GPU内存..."
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f'GPU内存已清理，可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"

# 检查GPU状态
echo "📊 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

# 设置较小的world_size以减少内存占用
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=2

echo "🔧 使用配置: WORLD_SIZE=$WORLD_SIZE, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 运行训练
echo "🎯 启动训练..."
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_dmd_test.py \
    --config configs/dmd_train_config.yaml \
    --seed 42

echo "✅ 训练完成!"



