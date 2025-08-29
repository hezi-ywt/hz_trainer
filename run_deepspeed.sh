#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "开始DMD训练 (DeepSpeed版本)..."

# 运行训练
python train_dmd_deepspeed.py \
    --config configs/dmd_train_config.yaml \
    --deepspeed_config deepspeed_config.json \
    --seed 42 \
    --num_gpus 1

echo "训练完成!" 