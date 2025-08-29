#!/usr/bin/env python3
"""
检查点大小监控脚本
"""

import os
import glob
from pathlib import Path
from datetime import datetime

def get_file_size_mb(filepath):
    """获取文件大小（MB）"""
    return os.path.getsize(filepath) / (1024 * 1024)

def analyze_checkpoints(checkpoint_dir="./checkpoints"):
    """分析检查点目录"""
    print(f"分析检查点目录: {checkpoint_dir}")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_dir):
        print("检查点目录不存在")
        return
    
    # 查找所有检查点文件
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    safetensors_files = glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
    
    total_size = 0
    
    print("Lightning检查点文件 (.ckpt):")
    for ckpt_file in sorted(ckpt_files):
        size_mb = get_file_size_mb(ckpt_file)
        total_size += size_mb
        mtime = datetime.fromtimestamp(os.path.getmtime(ckpt_file))
        print(f"  {os.path.basename(ckpt_file)}: {size_mb:.1f}MB ({mtime})")
    
    print("\nSafetensors文件 (.safetensors):")
    for st_file in sorted(safetensors_files):
        size_mb = get_file_size_mb(st_file)
        total_size += size_mb
        mtime = datetime.fromtimestamp(os.path.getmtime(st_file))
        print(f"  {os.path.basename(st_file)}: {size_mb:.1f}MB ({mtime})")
    
    print(f"\n总大小: {total_size:.1f}MB")
    
    # 分析文件大小分布
    if ckpt_files or safetensors_files:
        sizes = [get_file_size_mb(f) for f in ckpt_files + safetensors_files]
        print(f"平均文件大小: {sum(sizes)/len(sizes):.1f}MB")
        print(f"最大文件大小: {max(sizes):.1f}MB")
        print(f"最小文件大小: {min(sizes):.1f}MB")

def cleanup_old_checkpoints(checkpoint_dir="./checkpoints", keep_top_k=3):
    """清理旧的检查点文件"""
    print(f"\n清理旧检查点 (保留前{keep_top_k}个)...")
    
    # 获取所有检查点文件
    all_files = []
    for ext in ["*.ckpt", "*.safetensors"]:
        all_files.extend(glob.glob(os.path.join(checkpoint_dir, ext)))
    
    # 按修改时间排序
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 保留最新的文件
    files_to_keep = all_files[:keep_top_k]
    files_to_delete = all_files[keep_top_k:]
    
    print(f"保留文件:")
    for file in files_to_keep:
        print(f"  {os.path.basename(file)}")
    
    print(f"删除文件:")
    for file in files_to_delete:
        size_mb = get_file_size_mb(file)
        print(f"  {os.path.basename(file)} ({size_mb:.1f}MB)")
        os.remove(file)
    
    if files_to_delete:
        print(f"已删除 {len(files_to_delete)} 个文件")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检查点监控工具")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="检查点目录")
    parser.add_argument("--cleanup", action="store_true", help="清理旧检查点")
    parser.add_argument("--keep-top-k", type=int, default=3, help="保留的检查点数量")
    
    args = parser.parse_args()
    
    analyze_checkpoints(args.checkpoint_dir)
    
    if args.cleanup:
        cleanup_old_checkpoints(args.checkpoint_dir, args.keep_top_k) 