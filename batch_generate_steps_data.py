import os
import json
import time
import random
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
import pickle
from tqdm import tqdm

# 导入您的模型类
from lumina2_model_dmd import Lumina2ModelDMD

class StepsDataCollector:
    def __init__(self):
        self.steps_data = []
        self.current_step = 0
        
    def add_step(self, step_data):
        """添加一个step的数据"""
        step_info = {
            "step": self.current_step,
            "timestamp": time.time(),
            "data": step_data
        }
        self.steps_data.append(step_info)
        self.current_step += 1
        
    def get_steps_data(self):
        """获取所有收集的steps数据"""
        return self.steps_data.copy()
        
    def clear_steps_data(self):
        """清空steps数据"""
        self.steps_data = []
        self.current_step = 0

def get_sample_hook(collector):
    """返回一个hook函数，用于收集采样过程中的数据"""
    def hook_fn(**kwargs):
        # 提取关键数据
        step_data = {
            "x": kwargs.get("x", None),
            "sigma": kwargs.get("sigma", None),
            "denoised": kwargs.get("denoised", None),
            "cap_feats": kwargs.get("cap_feats", None),
            "cap_mask": kwargs.get("cap_mask", None),
            "neg_feats": kwargs.get("neg_feats", None),
            "neg_mask": kwargs.get("neg_mask", None),
            "v_pred": kwargs.get("v_pred", None),
            "v_pred_cond": kwargs.get("v_pred_cond", None),
            "v_pred_uncond": kwargs.get("v_pred_uncond", None),
            "current_timestep": kwargs.get("current_timestep", None),
            "guidance_scale": kwargs.get("guidance_scale", None),
            "shift": kwargs.get("shift", None),
        }
        
        # 将张量转换为numpy数组以便保存
        for key, value in step_data.items():
            if isinstance(value, torch.Tensor):
                # 先转换为float32，再转换为numpy
                step_data[key] = value.detach().cpu().float().numpy()
        
        collector.add_step(step_data)
    
    return hook_fn

def save_steps_data(steps_data, output_path):
    """保存steps数据到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(steps_data, f)
    
    print(f"Steps data saved to: {output_path}")

def batch_generate_with_steps_data(
    model,
    prompts,
    negative_prompts,
    output_dir,
    config,
    num_samples_per_prompt=1,
    cfg_range=None
):
    """批量生成图像并保存steps数据"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "steps_data"), exist_ok=True)
    
    collector = StepsDataCollector()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    total_samples = len(prompts) * num_samples_per_prompt
    sample_count = 0
    
    print(f"开始批量生成，总共 {total_samples} 个样本...")
    
    for prompt_idx, (prompt, negative_prompt) in tqdm(enumerate(zip(prompts, negative_prompts))):
        print(f"\n处理提示词 {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
        
        for sample_idx in range(num_samples_per_prompt):
            collector.clear_steps_data()
            
            seed = random.randint(0, 2**32 - 1)
            config.generation.seed = seed
            hook_fn = get_sample_hook(collector)
            # if cfg_range is not None:
            #     guidance_scale = random.uniform(cfg_range[0], cfg_range[1])
            # else:
            #     guidance_scale = config.generation.guidance_scale
            # # try:
            print(f"  生成样本 {sample_idx + 1}/{num_samples_per_prompt} (seed: {seed})...")
            
            pil_image = model.generate_image(
                model.model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                config=config,
                hook_fn=hook_fn
            )
            
            steps_data = collector.get_steps_data()
            
            # 保存图像
            image_filename = f"image_{timestamp}_prompt{prompt_idx:03d}_sample{sample_idx:02d}_seed{seed}.png"
            image_path = os.path.join(output_dir, "images", image_filename)
            pil_image.save(image_path)
            
            # 保存steps数据
            steps_filename = f"steps_{timestamp}_prompt{prompt_idx:03d}_sample{sample_idx:02d}_seed{seed}.pkl"
            steps_path = os.path.join(output_dir, "steps_data", steps_filename)
            save_steps_data(steps_data, steps_path)
            
            # 保存元数据
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "timestamp": timestamp,
                "image_path": image_path,
                "steps_data_path": steps_path,
                "num_steps": len(steps_data),
            }
            
            metadata_filename = f"metadata_{timestamp}_prompt{prompt_idx:03d}_sample{sample_idx:02d}_seed{seed}.json"
            metadata_path = os.path.join(output_dir, "steps_data", metadata_filename)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            sample_count += 1
            print(f"  ✓ 样本 {sample_count}/{total_samples} 完成")
            print(f"    收集到 {len(steps_data)} 个steps")
            
            collector.clear_steps_data()
                

    
    print(f"\n批量生成完成！总共生成 {sample_count} 个样本")
    print(f"输出目录: {output_dir}")

def main(output_dir = f'./batch_output_{time.strftime('%Y%m%d_%H%M%S')}',id = 0):
    """主函数"""
    if os.path.exists(output_dir):
        print(f"输出目录已存在: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "steps_data"), exist_ok=True)
        
    config = OmegaConf.create({
        "model": {
            "model_name": "NextDiT_2B_GQA_patch2_Adaln_Refiner",
            "model_path": "/mnt/huggingface/Lumina-Image-2.0",
            "vae_path": "/mnt/huggingface/Neta-Lumina/VAE/ae.safetensors",
            "text_encoder_path": "/mnt/huggingface/Neta-Lumina/Text Encoder/gemma_2_2b_fp16.safetensors",
            "init_from": "/mnt/huggingface/Neta-Lumina/Unet/neta-lumina-v1.0.safetensors",
            "use_ema": False
        },
        "advanced": {
            "use_ema": False
        },
        "trainer": {
            "batch_size": 1,
            "grad_clip": 1.0
        },
        "generation": {
            "image_height": 1024,
            "image_width": 1024,
            "steps": 30,
            "guidance_scale": 5.0,
            "cfg_trunc_ratio": 1.0,
            "seed": None,
            "discrete_flow_shift": 6.0,
            "renorm_cfg": 1.0
        },
        "k_diffusion": {
            "use_k_diffusion": True,
            "sampler": "euler_ancestral_RF",
            "scheduler_func": "linear_quadratic_schedule",
        },
        "output": {
            "output_dir": "./output"
        }
    })
    
    print("初始化模型...")
    # device_id = 3
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = Lumina2ModelDMD(config, device="cuda")
    
    # prompts = [
    #     "you are an assistant designed to generate anime images based on textual prompts. <Prompt Start> The artwork, by terigumi, presents a POV shot of Jane Doe from Zenless Zone Zero, kneeling and looking down at the viewer with a flirty smile. Jane has short, silver-gray hair, large aqua eyes and pink animal ears, as well as a black tail with a gold-colored tip. She wears a revealing black outfit that exposes her cleavage, paired with black pantyhose. Her arms extend towards the viewer, as if to embrace them.",
    #     "you are an assistant designed to generate anime images based on textual prompts. <Prompt Start> A beautiful anime girl with long blue hair and golden eyes, wearing a white dress, standing in a magical forest with glowing butterflies.",
    #     "you are an assistant designed to generate anime images based on textual prompts. <Prompt Start> A cute anime character with pink hair and green eyes, wearing a school uniform, sitting in a classroom with cherry blossoms outside the window.",
        
    # ]

    prompt_path = f'/mnt/huggingface/add_arrow4/all_text_zh1_prompt.json'
    with open(prompt_path, 'r',encoding='utf-8') as f:
        prompts_list = json.load(f)
    
    if id is not None:
        prompts = prompts_list[id::4]
        print(f"分片处理第 {id} 个文件，共 {len(prompts_list)} 个文件")
    else:
        prompts = prompts_list
    negative_prompts = [
        "You are an assistant designed to generate anime images based on textual prompts. <Prompt Start> blurry, worst quality, low quality, jpeg artifacts, signature, watermark, username, error, deformed hands, bad anatomy, extra limbs, poorly drawn hands, poorly drawn face, mutation, deformed, extra eyes, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross‑eyed, bad proportions, missing arms, missing legs, extra digit, fewer digits, cropped, normal quality"
    ] * len(prompts)
    
    batch_generate_with_steps_data(
        model=model,
        prompts=prompts,
        negative_prompts=negative_prompts,
        output_dir=output_dir,
        config=config,
        num_samples_per_prompt=2
    )

if __name__ == "__main__":
    main(output_dir = f'/mnt/hz_trainer/batch_output',id = 1) 