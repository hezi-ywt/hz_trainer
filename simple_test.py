#!/usr/bin/env python3
"""
简化测试脚本 - 只对核心变量进行多种测试
核心变量：提示词、模型、步数、CFG
"""

import os
import sys
import time
import random
import json
from omegaconf import OmegaConf
import torch

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lumina2_model_dmd import Lumina2ModelDMD

# ========== 测试配置 ==========

# 模型列表 - 在这里添加要测试的模型
MODELS = [
    "/mnt/hz_trainer/checkpoints/dmd_model_epoch=62_step=144207_val_loss=0.0000.safetensors",
    "/mnt/hz_trainer/checkpoints/dmd_model_epoch=39_step=091560_val_loss=0.0000.safetensors",
    # "/mnt/hz_trainer/checkpoints/dmd_model_epoch=12_step=029757_val_loss=0.0000.safetensors",
    "/mnt/hz_trainer/checkpoints/dmd_model_epoch=08_step=020601_val_loss=0.0000.safetensors",
    # "/mnt/hz_trainer/checkpoints/dmd_model_epoch=01_step=004578_val_loss=0.0000.safetensors",
    "/mnt/hz_trainer/checkpoints/dmd_model_epoch=06_step=003549_val_loss=0.0000.safetensors",
    "/mnt/huggingface/Neta-Lumina/Unet/neta-lumina-v1.0.safetensors"
]

# 提示词列表 - 在这里添加要测试的提示词
PROMPTS = [
    "you are an assistant designed to generate anime images based on textual prompts. <Prompt Start> 1girl, beautiful, detailed",
    "you are an assistant designed to generate anime images based on textual prompts. <Prompt Start> The artwork, by terigumi, presents a POV shot of Jane Doe from Zenless Zone Zero, kneeling and looking down at the viewer with a flirty smile. Jane has short, silver-gray hair, large aqua eyes and pink animal ears, as well as a black tail with a gold-colored tip. She wears a revealing black outfit that exposes her cleavage, paired with black pantyhose. Her arms extend towards the viewer, as if to embrace them.",
    "you are an assistant designed to generate anime images based on textual prompts. <Prompt Start> 1girl, portrait, beautiful face, detailed eyes",
]

# 步数列表 - 在这里添加要测试的步数
STEPS_LIST = [4, 12, 15]

# CFG列表 - 在这里添加要测试的CFG值
CFG_LIST = [3.0, 5.0]

# 负向提示词
NEGATIVE_PROMPT = "You are an assistant designed to generate anime images based on textual prompts. <Prompt Start> blurry, worst quality, low quality, jpeg artifacts, signature, watermark, username, error, deformed hands, bad anatomy, extra limbs, poorly drawn hands, poorly drawn face, mutation, deformed, extra eyes, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross‑eyed, bad proportions, missing arms, missing legs, extra digit, fewer digits, cropped, normal quality"

# 其他固定参数
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
SEED = 12345
OUTPUT_DIR = "/mnt/hz_trainer/output/simple_test"
JSON_OUTPUT_FILE = "/mnt/hz_trainer/output/simple_test/test_results.json"

# ========== 配置结束 ==========

def create_config(model_path, steps, cfg):
    """创建配置"""
    return OmegaConf.create({
        "model": {
            "model_name": "NextDiT_2B_GQA_patch2_Adaln_Refiner",
            "model_path": "/mnt/huggingface/Lumina-Image-2.0",
            "vae_path": "/mnt/huggingface/Lumina-Image-2.0/vae",
            "text_encoder_path": "/mnt/huggingface/Neta-Lumina/Text Encoder/gemma_2_2b_fp16.safetensors",
            "init_from": model_path,
            "use_ema": False,
            "use_fake_model": False,
            "use_real_model": False,
        },
        "advanced": {"use_ema": False},
        "trainer": {"batch_size": 1, "grad_clip": 1.0},
        "generation": {
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "steps": steps,
            "guidance_scale": cfg,
            "cfg_trunc_ratio": 1.0,
            "seed": SEED,
            "discrete_flow_shift": 6.0,
            "renorm_cfg": 1.0
        },
        "k_diffusion": {
            "use_k_diffusion": True,
            "sampler": "euler_ancestral_RF",
            "scheduler_func": "linear_quadratic_schedule",
        },
        "output": {"output_dir": OUTPUT_DIR}
    })

def run_single_test_with_model(model, model_path, prompt, steps, cfg, test_id):
    """使用已加载的模型运行单个测试"""
    print(f"\n=== 测试 {test_id} ===")
    print(f"模型: {os.path.basename(model_path)}")
    print(f"提示词: {prompt[:50]}...")
    print(f"步数: {steps}")
    print(f"CFG: {cfg}")
    
    try:
        # 创建配置
        config = create_config(model_path, steps, cfg)
        
        # 生成图像
        print("开始生成图像...")
        start_time = time.time()
        
        pil_image = model.generate_image(
            model.model,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            device="cuda",
            dtype=torch.bfloat16,
            hook_fn=None,
            config=config
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"图像生成完成，耗时: {generation_time:.2f}秒")
        print(f"测试 {test_id} 完成!")
        pil_image.save(f"{OUTPUT_DIR}/image_{test_id}.png")
        # 收集测试结果数据
        test_result = {
            "test_id": test_id,
            "model_path": model_path,
            "model_name": os.path.basename(model_path),
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "steps": steps,
            "cfg": cfg,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "seed": SEED,
            "generation_time": generation_time,
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_image": f"{OUTPUT_DIR}/image_{test_id}.png"  # 图像文件名
        }
        
        return True, test_result
        
    except Exception as e:
        print(f"测试 {test_id} 失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 收集失败结果数据
        test_result = {
            "test_id": test_id,
            "model_path": model_path,
            "model_name": os.path.basename(model_path),
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "steps": steps,
            "cfg": cfg,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "seed": SEED,
            "generation_time": 0,
            "status": "failed",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_image": None
        }
        
        return False, test_result

def load_model(model_path):
    """加载模型"""
    print(f"\n=== 加载模型: {os.path.basename(model_path)} ===")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    try:
        # 创建配置
        config = create_config(model_path, STEPS_LIST[0], CFG_LIST[0])  # 使用第一个参数作为默认值
        
        # 初始化模型
        print("正在初始化模型...")
        model = Lumina2ModelDMD(config, device="cuda")
        print("模型初始化完成!")
        return model
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=== 简化测试脚本 ===")
    print(f"将测试 {len(MODELS)} 个模型")
    print(f"将测试 {len(PROMPTS)} 个提示词")
    print(f"将测试 {len(STEPS_LIST)} 种步数")
    print(f"将测试 {len(CFG_LIST)} 种CFG值")
    print(f"总共 {len(MODELS) * len(PROMPTS) * len(STEPS_LIST) * len(CFG_LIST)} 个测试")
    
    # 设置随机种子
    random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"设置随机种子: {SEED}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 运行所有测试
    test_count = 0
    success_count = 0
    all_results = []
    
    for model_path in MODELS:
        print(f"\n{'='*60}")
        print(f"开始测试模型: {os.path.basename(model_path)}")
        print(f"{'='*60}")
        
        # 加载模型
        model = load_model(model_path)
        if model is None:
            print(f"跳过模型: {os.path.basename(model_path)}")
            continue
        
        # 计算当前模型的测试数量
        current_model_tests = len(PROMPTS) * len(STEPS_LIST) * len(CFG_LIST)
        print(f"当前模型将运行 {current_model_tests} 个测试")
        
        # 运行当前模型的所有测试
        for prompt in PROMPTS:
            for steps in STEPS_LIST:
                for cfg in CFG_LIST:
                    test_count += 1
                    test_id = f"{test_count:03d}"
                    
                    success, test_result = run_single_test_with_model(model, model_path, prompt, steps, cfg, test_id)
                    if success:
                        success_count += 1
                    
                    if test_result:
                        all_results.append(test_result)
        
        print(f"\n模型 {os.path.basename(model_path)} 测试完成!")
        print(f"当前进度: {test_count} / {len(MODELS) * len(PROMPTS) * len(STEPS_LIST) * len(CFG_LIST)}")
        
        # 清理模型内存（可选）
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("模型已卸载，内存已清理")
    
    # 保存所有结果到JSON文件
    results_data = {
        "test_summary": {
            "total_tests": test_count,
            "successful_tests": success_count,
            "failed_tests": test_count - success_count,
            "success_rate": success_count/test_count*100 if test_count > 0 else 0,
            "test_config": {
                "models": MODELS,
                "prompts": PROMPTS,
                "steps_list": STEPS_LIST,
                "cfg_list": CFG_LIST,
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH,
                "seed": SEED,
                "output_dir": OUTPUT_DIR
            }
        },
        "test_results": all_results
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(JSON_OUTPUT_FILE), exist_ok=True)
    
    # 保存JSON文件
    with open(JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 测试完成 ===")
    print(f"总测试数: {test_count}")
    print(f"成功数: {success_count}")
    print(f"失败数: {test_count - success_count}")
    print(f"成功率: {success_count/test_count*100:.1f}%")
    print(f"结果已保存到: {JSON_OUTPUT_FILE}")

if __name__ == "__main__":
    main() 