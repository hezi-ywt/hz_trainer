# -*- coding: utf-8 -*-
import PIL
import torch
import pyarrow as pa
import numpy as np
import json
import os
import pandas as pd

def save_latent_dict_to_arrow(data_list, filepath):
    """
    将包含latent的字典保存到Arrow文件
    
    Args:
        latent_dict: 包含latent张量的字典
        filepath: 保存文件路径
    """
    arrow_data = {}
    for latent_dict in data_list:
        bs = []
        for key, value in latent_dict.items():
            # if isinstance(value, torch.Tensor):
            #     value = value.numpy().flatten().tolist()
            # elif isinstance(value, list):
            #     value = [v.numpy().flatten().tolist() for v in value]
            # elif isinstance(value, dict):
            #     value = json.dumps(value)
            # else:
            #     value = value
            # bs.append(value)
            pass
        columns_list = ["prompt", "negative_prompt", "sigmas", "latent", "uncond_latent", "prompt_embeds", "negative_prompt_embeds", "num_inference_steps", "guidance_scale", "generated_image"]
        row_data = {}
        for key in columns_list:
            if key in latent_dict:
                value = latent_dict[key]
                # Convert tensors to flattened lists
                if isinstance(value, torch.Tensor):
                    value = value.numpy().flatten().tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    value = [v.numpy().flatten().tolist() for v in value]
                elif isinstance(value, dict):
                    value = json.dumps(value)
                row_data[key] = [value]
            else:
                row_data[key] = [None]
        dataframe = pd.DataFrame(row_data)
        table = pa.Table.from_pandas(dataframe)
        
        # 创建目录（如果文件路径包含目录）
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # 保存Arrow文件
        with pa.OSFile(filepath, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
    
    print(f"已保存到: {filepath}")

def load_latent_dict_from_arrow(filepath):
    """
    从Arrow文件加载latent字典
    
    Args:
        filepath: 文件路径
    Returns:
        包含latent张量的字典
    """
    with pa.OSFile(filepath, "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    
    loaded_dict = {}
    
    # 获取所有列名
    column_names = table.column_names
    
    # 找出latent相关的列
    latent_keys = set()
    for col in column_names:
        if not col.endswith('_shape') and not col.endswith('_dtype'):
            latent_keys.add(col)
    
    for key in latent_keys:
        if f"{key}_shapes" in column_names and f"{key}_dtypes" in column_names:
            # 这是一个张量列表
            tensor_list = []
            shapes_list = table.column(f"{key}_shapes").to_pylist()
            dtypes_list = table.column(f"{key}_dtypes").to_pylist()
            tensor_arrays = table.column(key).to_pylist()
            
            for i, (tensor_flat, shape, dtype) in enumerate(zip(tensor_arrays, shapes_list, dtypes_list)):
                latent_np = np.array(tensor_flat).reshape(shape)
                tensor_list.append(torch.from_numpy(latent_np))
            
            loaded_dict[key] = tensor_list
        elif f"{key}_shape" in column_names and f"{key}_dtype" in column_names:
            # 这是一个latent张量
            latent_flat = table.column(key)[0].as_py()
            shape = table.column(f"{key}_shape")[0].as_py()
            
            # 重新构建张量
            latent_np = np.array(latent_flat).reshape(shape)
            loaded_dict[key] = torch.from_numpy(latent_np)
        else:
            # 其他类型的数据
            value = table.column(key)[0].as_py()
            if isinstance(value, str) and value.startswith('{'):
                # 尝试解析为JSON
                try:
                    loaded_dict[key] = json.loads(value)
                except:
                    loaded_dict[key] = value
            else:
                loaded_dict[key] = value
    
    return loaded_dict

# 示例使用
if __name__ == "__main__":
    # 创建latent张量
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # time schedule
    latent = [torch.randn(1, 4, 64, 64) for _ in range(len(sigmas))]  #  positive latent
    uncond_latent = [torch.randn(1, 4, 64, 64) for _ in range(len(sigmas))]  #  negative latent
    prompt = "a beautiful girl"
    negative_prompt = "ugly, bad"
    prompt_embeds = {"cap_feat":[],
                     "cap_mask":[]}
    negative_prompt_embeds = {"cap_feat":[],
                     "cap_mask":[]}
    num_inference_steps = 10 # latent的步数
    guidance_scale = 5 # cfg引导尺度
    generated_image = None  # PIL.Image 
    
    
    # 创建包含latent的字典
    latent_dict = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sigmas": sigmas,
        "latent": latent,
        "uncond_latent": uncond_latent,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generated_image": generated_image
    }
    
    # 保存到Arrow文件
    save_latent_dict_to_arrow([latent_dict], "latent_dict.arrow")
    
    # 从Arrow文件读取
    loaded_dict = load_latent_dict_from_arrow("latent_dict.arrow")
    
    # 验证数据
    print(loaded_dict)