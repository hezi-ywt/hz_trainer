#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lumina_arrow_prompt_process import LuminaArrowPromptProcessor, process_single_text_zh

def test_prompt_processor():
    """测试提示词处理器"""
    
    # 示例text_zh数据
    sample_text_zh = {
        "gemini_caption_v6": {
            "midjourney_style_summary_zh": "魅惑眼神，柔媚肌肤，Raiden Shogun躺卧姿态，白色绷带缠绕，花瓣点缀，性感氛围"
        },
        "wd_tagger": "1girl, long_hair, blue_eyes, white_hair, dress, sitting, flower",
        "wd_tagger_metadata": {"confidence": 0.95}
    }
    
    print("=== 测试提示词处理器 ===")
    
    # 创建处理器
    processor = LuminaArrowPromptProcessor()
    
    # 处理数据
    result = processor.process_text_zh(sample_text_zh)
    print(f"处理结果: {result}")
    
    # 测试便捷函数
    result2 = process_single_text_zh(json.dumps(sample_text_zh))
    print(f"便捷函数结果: {result2}")
    
    # 测试更多样本
    print("\n=== 测试更多样本 ===")
    
    test_cases = [
        {
            "gemini_caption_v6": {
                "Detailed_caption_en": "A beautiful anime girl with long blue hair, wearing a white dress, sitting in a garden full of flowers"
            }
        },
        {
            "wd_tagger": "1girl, blue_hair, white_dress, garden, flowers, sitting, anime_style"
        },
        {
            "danbooru_meta": {
                "character": ["sakura", "haruhi"],
                "artist": ["artist1", "artist2"],
                "general": ["1girl", "long_hair", "blue_eyes", "dress"],
                "rating_tags": ["safe"],
                "quality_tags": ["high_quality"]
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n样本 {i+1}:")
        result = processor.process_text_zh(test_case)
        print(f"结果: {result[:200]}...")

if __name__ == "__main__":
    test_prompt_processor() 