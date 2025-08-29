#!/bin/bash

echo "测试检查点修复..."

# 测试检查点功能
python test_checkpoint.py

echo ""

# 测试模型访问
echo "测试模型访问..."
python test_model_access.py

echo "测试完成!" 