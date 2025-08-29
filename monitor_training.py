#!/usr/bin/env python3
"""
DMD训练监控脚本
"""

import os
import time
import psutil
import GPUtil
from datetime import datetime
import json

def get_system_info():
    """获取系统信息"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': gpu.memoryUtil * 100,
                'temperature': gpu.temperature,
                'load': gpu.load * 100 if gpu.load else 0
            })
    except:
        gpu_info = []
    
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3),
        'gpus': gpu_info
    }

def monitor_training(log_dir="./logs", interval=30):
    """监控训练过程"""
    print("开始监控DMD训练...")
    print(f"监控间隔: {interval}秒")
    print(f"日志目录: {log_dir}")
    
    # 创建监控日志文件
    monitor_log = os.path.join(log_dir, "training_monitor.jsonl")
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        while True:
            # 获取系统信息
            info = get_system_info()
            
            # 保存到日志文件
            with open(monitor_log, 'a') as f:
                f.write(json.dumps(info) + '\n')
            
            # 打印当前状态
            print(f"\n[{info['timestamp']}] 系统状态:")
            print(f"  CPU使用率: {info['cpu_percent']:.1f}%")
            print(f"  内存使用率: {info['memory_percent']:.1f}% ({info['memory_used_gb']:.1f}GB / {info['memory_total_gb']:.1f}GB)")
            
            if info['gpus']:
                print("  GPU状态:")
                for gpu in info['gpus']:
                    print(f"    GPU {gpu['id']} ({gpu['name']}):")
                    print(f"      内存: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_percent']:.1f}%)")
                    print(f"      温度: {gpu['temperature']}°C")
                    print(f"      负载: {gpu['load']:.1f}%")
            
            # 检查训练日志
            tensorboard_log = os.path.join(log_dir, "lightning_logs")
            if os.path.exists(tensorboard_log):
                print(f"  TensorBoard日志目录: {tensorboard_log}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def check_training_progress(log_dir="./logs"):
    """检查训练进度"""
    print("检查训练进度...")
    
    # 检查检查点
    checkpoint_dir = "./checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            print(f"找到 {len(checkpoints)} 个检查点:")
            for ckpt in sorted(checkpoints):
                ckpt_path = os.path.join(checkpoint_dir, ckpt)
                size_mb = os.path.getsize(ckpt_path) / (1024**2)
                mtime = datetime.fromtimestamp(os.path.getmtime(ckpt_path))
                print(f"  {ckpt} ({size_mb:.1f}MB, {mtime})")
        else:
            print("未找到检查点文件")
    
    # 检查TensorBoard日志
    tensorboard_log = os.path.join(log_dir, "lightning_logs")
    if os.path.exists(tensorboard_log):
        versions = [d for d in os.listdir(tensorboard_log) if d.startswith('version_')]
        if versions:
            latest_version = max(versions, key=lambda x: int(x.split('_')[1]))
            events_file = os.path.join(tensorboard_log, latest_version, "events.out.tfevents.*")
            print(f"TensorBoard日志版本: {latest_version}")
        else:
            print("未找到TensorBoard日志")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DMD训练监控")
    parser.add_argument("--mode", choices=["monitor", "check"], default="monitor", 
                       help="监控模式: monitor(持续监控) 或 check(检查进度)")
    parser.add_argument("--log-dir", default="./logs", help="日志目录")
    parser.add_argument("--interval", type=int, default=30, help="监控间隔(秒)")
    
    args = parser.parse_args()
    
    if args.mode == "monitor":
        monitor_training(args.log_dir, args.interval)
    else:
        check_training_progress(args.log_dir) 