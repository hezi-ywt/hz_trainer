#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU显存监控脚本
实时监控GPU显存使用情况
"""

import time
import subprocess
import psutil
import os

def get_gpu_memory():
    """获取GPU显存信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id = parts[0]
                    used = int(parts[1])
                    total = int(parts[2])
                    free = int(parts[3])
                    gpu_info.append({
                        'id': gpu_id,
                        'used': used,
                        'total': total,
                        'free': free,
                        'usage_percent': (used / total) * 100
                    })
        return gpu_info
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return []

def get_process_memory():
    """获取当前进程内存使用"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def monitor_gpu(interval=5):
    """持续监控GPU显存"""
    print("🔍 GPU显存监控启动...")
    print("按 Ctrl+C 停止监控")
    print("=" * 80)
    
    try:
        while True:
            # 清屏
            os.system('clear')
            
            # 获取GPU信息
            gpu_info = get_gpu_memory()
            process_memory = get_process_memory()
            
            print(f"🕐 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💻 当前进程内存: {process_memory:.2f} MB")
            print("=" * 80)
            
            if gpu_info:
                print("GPU显存使用情况:")
                print(f"{'GPU':<4} {'已用(MB)':<10} {'可用(MB)':<10} {'总计(MB)':<10} {'使用率(%)':<10}")
                print("-" * 50)
                
                for gpu in gpu_info:
                    print(f"{gpu['id']:<4} {gpu['used']:<10} {gpu['free']:<10} {gpu['total']:<10} {gpu['usage_percent']:<10.1f}")
                
                # 显示显存使用图表
                print("\n显存使用图表:")
                for gpu in gpu_info:
                    bar_length = 30
                    used_bars = int((gpu['used'] / gpu['total']) * bar_length)
                    free_bars = bar_length - used_bars
                    
                    bar = "█" * used_bars + "░" * free_bars
                    print(f"GPU{gpu['id']}: [{bar}] {gpu['usage_percent']:.1f}%")
            else:
                print("❌ 无法获取GPU信息")
            
            print("\n" + "=" * 80)
            print(f"刷新间隔: {interval}秒 | 按 Ctrl+C 停止")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 监控已停止")

if __name__ == "__main__":
    monitor_gpu()



