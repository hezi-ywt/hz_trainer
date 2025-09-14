#!/usr/bin/env python3
"""
调试脚本 - 检查图像文件是否存在
"""

import json
import os
import sys

def check_images(json_file_path):
    """检查JSON文件中引用的图像文件是否存在"""
    
    print(f"检查JSON文件: {json_file_path}")
    
    if not os.path.exists(json_file_path):
        print(f"❌ JSON文件不存在: {json_file_path}")
        return
    
    # 读取JSON数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_summary = data['test_summary']
    test_results = data['test_results']
    
    output_dir = test_summary['test_config']['output_dir']
    print(f"输出目录: {output_dir}")
    
    # 检查输出目录是否存在
    if not os.path.exists(output_dir):
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    print(f"✅ 输出目录存在")
    
    # 检查每个测试结果的图像文件
    total_images = 0
    existing_images = 0
    missing_images = 0
    
    print(f"\n检查图像文件:")
    print("-" * 80)
    
    for result in test_results:
        if result['status'] == 'success' and result['output_image']:
            total_images += 1
            image_path = os.path.join(output_dir, result['output_image'])
            
            if os.path.exists(image_path):
                existing_images += 1
                file_size = os.path.getsize(image_path)
                print(f"✅ {result['test_id']}: {result['output_image']} ({file_size} bytes)")
            else:
                missing_images += 1
                print(f"❌ {result['test_id']}: {result['output_image']} (文件不存在)")
        else:
            print(f"⚠️  {result['test_id']}: 跳过 (状态: {result['status']})")
    
    print("-" * 80)
    print(f"总结:")
    print(f"  总图像数: {total_images}")
    print(f"  存在图像: {existing_images}")
    print(f"  缺失图像: {missing_images}")
    print(f"  成功率: {existing_images/total_images*100:.1f}%" if total_images > 0 else "  成功率: 0%")
    
    # 列出输出目录中的所有文件
    print(f"\n输出目录中的所有文件:")
    print("-" * 80)
    try:
        files = os.listdir(output_dir)
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                print(f"📄 {file} ({file_size} bytes)")
            else:
                print(f"📁 {file}/")
    except Exception as e:
        print(f"❌ 无法列出目录内容: {e}")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python debug_images.py <json_file_path>")
        print("示例: python debug_images.py ./output/simple_test/test_results.json")
        return
    
    json_file_path = sys.argv[1]
    check_images(json_file_path)

if __name__ == "__main__":
    main() 