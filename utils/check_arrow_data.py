#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from io import BytesIO
import pyarrow as pa
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

def check_single_arrow(arrow_path, num_samples=5, verbose=True):
    """检查单个Arrow文件的数据"""
    print(f"\n检查Arrow文件: {arrow_path}")
    
    # 读取Arrow文件
    try:
        reader = pa.ipc.open_file(arrow_path)
        table = reader.read_all()
        
        # 基本信息
        num_rows = table.num_rows
        column_names = table.column_names
        
        print(f"  记录总数: {num_rows}")
        print(f"  列名: {column_names}")
        
        # 将Arrow表转换为Pandas DataFrame以便处理
        # 不转换image列以避免内存问题
        columns_to_convert = [col for col in column_names if col != 'image']
        df_meta = table.select(columns_to_convert).to_pandas()
        
        # 数据统计
        if 'width' in column_names and 'height' in column_names:
            avg_width = df_meta['width'].mean()
            avg_height = df_meta['height'].mean()
            print(f"  平均图像尺寸: {avg_width:.2f} x {avg_height:.2f}")
            
        # 检查JSON文本字段
        text_fields = [col for col in column_names if col.startswith('text')]
        for field in text_fields:
            valid_json = 0
            json_keys = set()
            
            for text in df_meta[field]:
                try:
                    if text and isinstance(text, str):
                        json_obj = json.loads(text)
                        valid_json += 1
                        if isinstance(json_obj, dict):
                            for key in json_obj.keys():
                                json_keys.add(key)
                except:
                    pass
            
            print(f"  字段 '{field}': 有效JSON数: {valid_json}/{num_rows}")
            print(f"  JSON键集合: {json_keys}")
        
        # 抽样检查
        if verbose and num_samples > 0:
            print("\n抽样数据检查:")
            
            # 随机选择样本
            indices = np.random.choice(num_rows, min(num_samples, num_rows), replace=False)
            
            for idx in indices:
                print(f"\n  样本 #{idx}:")
                
                # 显示元数据
                for col in columns_to_convert:
                    value = df_meta.iloc[idx][col]
                    if col in text_fields and isinstance(value, str):
                        try:
                            json_obj = json.loads(value)
                            if 'caption' in json_obj:
                                print(f"    {col}: caption = {json_obj['caption'][:100]}...")
                            else:
                                print(f"    {col}: {str(json_obj)[:100]}...")
                        except:
                            print(f"    {col}: {value[:100]}...")
                    else:
                        print(f"    {col}: {value}")
                
                # 检查图像数据
                if 'image' in column_names:
                    image_bytes = table['image'][idx].as_py()
                    try:
                        img = Image.open(BytesIO(image_bytes))
                        width, height = img.size
                        print(f"    图像: 格式={img.format}, 尺寸={width}x{height}, 模式={img.mode}")
                    except Exception as e:
                        print(f"    图像: 无法解析 ({str(e)})")
        
        return True
        
    except Exception as e:
        print(f"  错误: {str(e)}")
        return False

def check_arrow_dir(arrow_dir, num_files=0, num_samples=5, verbose=True):
    """检查目录中的所有Arrow文件"""
    if not os.path.exists(arrow_dir):
        print(f"目录不存在: {arrow_dir}")
        return
    
    arrow_files = [f for f in os.listdir(arrow_dir) if f.endswith('.arrow')]
    arrow_files.sort()
    
    if num_files > 0:
        arrow_files = arrow_files[:num_files]
    
    print(f"找到 {len(arrow_files)} 个Arrow文件")
    
    valid_files = 0
    for arrow_file in tqdm(arrow_files):
        arrow_path = os.path.join(arrow_dir, arrow_file)
        if check_single_arrow(arrow_path, num_samples, verbose):
            valid_files += 1
    
    print(f"\n总结: 检查了 {len(arrow_files)} 个文件，{valid_files} 个有效")

def get_all_text_zh(arrow_path):
    """获取Arrow文件中所有的text_zh字段，返回列表"""
    print(f"正在读取Arrow文件: {arrow_path}")
    
    try:
        # 读取Arrow文件
        reader = pa.ipc.open_file(arrow_path)
        table = reader.read_all()
        
        # 检查是否存在text_zh列
        if 'text_zh' not in table.column_names:
            print("错误: 文件中没有找到 'text_zh' 列")
            return []
        
        # 获取text_zh列的所有数据
        text_zh_column = table['text_zh']
        text_zh_list = []
        
        print(f"开始提取 {table.num_rows} 条记录的text_zh数据...")
        
        # 逐行提取text_zh数据
        for i in tqdm(range(table.num_rows), desc="提取text_zh"):
            text_zh_value = text_zh_column[i].as_py()
            text_zh_list.append(text_zh_value)
        
        print(f"成功提取了 {len(text_zh_list)} 条text_zh数据")
        return text_zh_list
        
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return []

def get_all_text_zh_from_dir(arrow_dir, num_files=0,exclude_dirs=[]):
    """从目录中的所有Arrow文件获取text_zh字段，返回合并的列表"""
    if not os.path.exists(arrow_dir):
        print(f"目录不存在: {arrow_dir}")
        return []
    #walk   
    arrow_files = []
    for root, dirs, files in os.walk(arrow_dir):
        for file in files:
            if file.endswith('.arrow'):
                path = os.path.join(root, file)
                if any(exclude_dir in path for exclude_dir in exclude_dirs):
                    continue
                arrow_files.append(path)
    arrow_files.sort()
    
    if num_files > 0:
        arrow_files = arrow_files[:num_files]
    
    print(f"找到 {len(arrow_files)} 个Arrow文件")
    
    all_text_zh = []
    
    for arrow_file in tqdm(arrow_files, desc="处理文件"):
        arrow_path = os.path.join(arrow_dir, arrow_file)
        text_zh_list = get_all_text_zh(arrow_path)
        all_text_zh.extend(text_zh_list)
    
    print(f"总共提取了 {len(all_text_zh)} 条text_zh数据")
    return all_text_zh


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="检查Arrow文件中的数据")
    # parser.add_argument('path', help='Arrow文件路径或包含Arrow文件的目录')
    # parser.add_argument('--samples', type=int, default=5, help='每个文件检查的样本数')
    # parser.add_argument('--files', type=int, default=0, help='检查的文件数量（0表示全部）')
    # parser.add_argument('--quiet', action='store_true', help='不显示详细信息')
    
    # args = parser.parse_args()
    
    # if os.path.isdir(args.path):
    #     check_arrow_dir(args.path, args.files, args.samples, not args.quiet)
    # elif os.path.isfile(args.path) and args.path.endswith('.arrow'):
    #     check_single_arrow(args.path, args.samples, not args.quiet)
    # else:
    #     print("请提供有效的Arrow文件或包含Arrow文件的目录") 
    
    # 示例：获取单个文件的text_zh列表
    text_zh_list = get_all_text_zh('/mnt/huggingface/add_arrow4/good_arrow/00000.arrow')
    
    # 打印前几个样本
    print(f"\n前5个text_zh样本:")
    for i, text_zh in enumerate(text_zh_list[:5]):
        print(f"{i+1}. {text_zh[:200]}...")
    

    # 示例：获取目录中所有文件的text_zh列表
    # all_text_zh = get_all_text_zh_from_dir('/mnt/huggingface/add_arrow4/good_arrow', num_files=3)
    # print(f"\n总共获取了 {len(all_text_zh)} 条text_zh数据")
    arrow_dir = '/mnt/huggingface/add_arrow4/最终版_arrow'
    all_text_zh = get_all_text_zh_from_dir(arrow_dir)
    print(f"\n总共获取了 {len(all_text_zh)} 条text_zh数据")
    # 保存为json
    with open('/mnt/huggingface/add_arrow4/all_text_zh1.json', 'w',encoding='utf-8') as f:
        json.dump(all_text_zh, f, ensure_ascii=False, indent=2)