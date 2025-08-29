#!/bin/bash

# hz_trainer Git 上传脚本

set -e  # 遇到错误时退出

echo "=== hz_trainer Git 上传脚本 ==="

# 检查是否在正确的目录
if [ ! -f "train_dmd_test.py" ]; then
    echo "错误: 请在 hz_trainer 目录下运行此脚本"
    exit 1
fi

# 检查 git 是否已初始化
if [ ! -d ".git" ]; then
    echo "初始化 Git 仓库..."
    git init
fi

# 添加所有文件到暂存区
echo "添加文件到暂存区..."
git add .

# 检查是否有更改
if git diff --cached --quiet; then
    echo "没有新的更改需要提交"
    exit 0
fi

# 显示将要提交的文件
echo "将要提交的文件:"
git diff --cached --name-only

# 获取提交信息
if [ -z "$1" ]; then
    echo "请输入提交信息 (或按回车使用默认信息):"
    read commit_msg
    if [ -z "$commit_msg" ]; then
        commit_msg="Update hz_trainer: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
else
    commit_msg="$1"
fi

# 提交更改
echo "提交更改: $commit_msg"
git commit -m "$commit_msg"

# 检查是否有远程仓库
if git remote -v | grep -q origin; then
    echo "推送到远程仓库..."
    git push origin main 2>/dev/null || git push origin master 2>/dev/null || {
        echo "推送失败，请检查远程仓库配置"
        echo "当前远程仓库:"
        git remote -v
        echo ""
        echo "您可以手动运行以下命令推送:"
        echo "git push origin <branch-name>"
    }
else
    echo "没有配置远程仓库"
    echo "您可以使用以下命令添加远程仓库:"
    echo "git remote add origin <repository-url>"
    echo "然后运行: git push -u origin main"
fi

echo "=== Git 上传完成 ===" 