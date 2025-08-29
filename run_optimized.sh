#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "开始DMD训练 (优化版本)..."

# 运行训练
python train_dmd_test.py \
    --config configs/dmd_train_config.yaml \
    --seed 42

echo "训练完成!" 