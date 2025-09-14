#!/bin/bash

# DMD训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建必要的目录
mkdir -p checkpoints
mkdir -p logs
mkdir -p output

# 运行训练
python train_dmd_test.py \
    --config configs/dmd_train_config.yaml \

echo "训练完成!" 