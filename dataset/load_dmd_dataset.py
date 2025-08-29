import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image
import h5py
from tqdm import tqdm
import logging
import torchvision.transforms as T
import random

logger = logging.getLogger(__name__)

flip_norm = T.Compose(
            [   
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                
            ]
        )
 

class DMDStepsDataset(Dataset):
    """
    DMD (Diffusion Model Distillation) 数据集加载器
    用于加载批量生成过程中收集的steps数据
    """
    
    def __init__(
        self, 
        data_dir: str,
        load_images: bool = True,
        load_steps_data: bool = True,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径，包含images和steps_data子目录
            load_images: 是否加载图像数据
            load_steps_data: 是否加载steps数据
            transform: 图像变换函数
            max_samples: 最大样本数量，None表示加载所有样本
        """
        self.data_dir = Path(data_dir)
        self.load_images = load_images
        self.load_steps_data = load_steps_data
        self.transform = transform
        self.max_samples = max_samples
        
        # 验证目录结构
        self.images_dir = self.data_dir / "images"
        self.steps_data_dir = self.data_dir / "steps_data"
        
        if not self.images_dir.exists():
            raise ValueError(f"图像目录不存在: {self.images_dir}")
        if not self.steps_data_dir.exists():
            raise ValueError(f"Steps数据目录不存在: {self.steps_data_dir}")
        
        # 扫描数据文件
        self.samples = self._scan_samples()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"加载了 {len(self.samples)} 个样本")
    
    def _scan_samples(self) -> List[Dict[str, str]]:
        """扫描数据目录，找到所有匹配的样本"""
        samples = []
        
        # 获取所有图像文件
        image_files = list(self.images_dir.glob("*.png"))
        image_files.extend(list(self.images_dir.glob("*.jpg")))
        image_files.extend(list(self.images_dir.glob("*.jpeg")))
        
        for image_file in image_files:
            # 从文件名提取信息
            filename = image_file.stem
            parts = filename.split('_')
            
            # 查找对应的steps数据和元数据文件
            steps_file = self.steps_data_dir / f"steps_{'_'.join(parts[1:])}.pkl"
            metadata_file = self.steps_data_dir / f"metadata_{'_'.join(parts[1:])}.json"
            
            if steps_file.exists() and metadata_file.exists():
                sample_info = {
                    "image_path": str(image_file),
                    "steps_data_path": str(steps_file),
                    "metadata_path": str(metadata_file),
                    "filename": filename
                }
                samples.append(sample_info)
        
        # 按文件名排序
        samples.sort(key=lambda x: x["filename"])
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        sample_info = self.samples[idx]
        
        result = {
            "sample_info": sample_info,
            "filename": sample_info["filename"]
        }
        
        # 加载元数据
        with open(sample_info["metadata_path"], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        result["metadata"] = metadata
        
        # 加载图像
        if self.load_images:
            image = Image.open(sample_info["image_path"]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            result["image"] = image
        
        # 加载steps数据
        if self.load_steps_data:
            with open(sample_info["steps_data_path"], 'rb') as f:
                steps_data = pickle.load(f)
            result["steps_data"] = steps_data
        
        return result
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """获取提示词统计信息"""
        prompts = []
        negative_prompts = []
        
        for sample in self.samples:
            with open(sample["metadata_path"], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            prompts.append(metadata.get("prompt", ""))
            negative_prompts.append(metadata.get("negative_prompt", ""))
        
        return {
            "total_samples": len(prompts),
            "unique_prompts": len(set(prompts)),
            "unique_negative_prompts": len(set(negative_prompts)),
            "prompt_examples": list(set(prompts))[:5],  # 前5个示例
            "negative_prompt_examples": list(set(negative_prompts))[:5]
        }
    
    def get_steps_statistics(self) -> Dict[str, Any]:
        """获取steps数据统计信息"""
        if not self.load_steps_data:
            return {"error": "Steps data not loaded"}
        
        num_steps_list = []
        step_keys = set()
        
        for i in range(min(10, len(self))):  # 只检查前10个样本
            sample = self[i]
            steps_data = sample["steps_data"]
            num_steps_list.append(len(steps_data))
            
            if steps_data:
                step_keys.update(steps_data[0]["data"].keys())
        
        return {
            "avg_steps": np.mean(num_steps_list) if num_steps_list else 0,
            "min_steps": min(num_steps_list) if num_steps_list else 0,
            "max_steps": max(num_steps_list) if num_steps_list else 0,
            "step_keys": list(step_keys)
        }


class DMDH5Dataset(Dataset):
    """
    用于加载HDF5格式的DMD数据集
    参考CIFAR Pairs的实现
    """
    
    def __init__(self, h5_path: str, transform=None):
        """
        初始化HDF5数据集
        
        Args:
            h5_path: HDF5文件路径
            transform: 图像变换函数
        """
        self.h5_path = h5_path
        self.transform = transform
        self.dataset = None
        
        # 检查文件是否存在
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5文件不存在: {h5_path}")
        
        # 获取样本数量
        with h5py.File(h5_path, "r") as file:
            self.num_samples = len(file["data"])
        
        logger.info(f"加载HDF5数据集: {h5_path}, 样本数量: {self.num_samples}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        # 延迟加载数据集以避免多进程问题
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, "r")["data"]
        
        sample = self.dataset[str(idx)]
        pairs = sample[()]
        attributes = sample.attrs
        
        image, latent = pairs
        
        result = {
            "instance_id": idx,
            "image": torch.from_numpy(image).float(),
            "latent": torch.from_numpy(latent).float(),
            "class_id": attributes["class_idx"],
            "seed": attributes["seed"],
        }
        
        if self.transform:
            result["image"] = self.transform(result["image"])
        
        return result


class DMDDataLoader:
    """
    DMD数据加载器工厂类
    提供便捷的数据加载接口
    """
    
    @staticmethod
    def create_steps_dataloader(
        data_dir: str,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        load_images: bool = True,
        load_steps_data: bool = True,
        transform=None,
        step_sample_num: int = 1,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> DataLoader:
        """
        创建steps数据的数据加载器
        
        Args:
            data_dir: 数据目录
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            load_images: 是否加载图像
            load_steps_data: 是否加载steps数据
            transform: 图像变换
            max_samples: 最大样本数
            **kwargs: 其他DataLoader参数
        
        Returns:
            DataLoader实例
        """
        dataset = DMDStepsDataset(
            data_dir=data_dir,
            load_images=load_images,
            load_steps_data=load_steps_data,
            transform=transform,
            max_samples=max_samples
        )
        # self.step_sample_num = step_sample_num
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=DMDDataLoader.custom_collate_fn,
            **kwargs
        )
    
    @staticmethod
    def create_h5_dataloader(
        h5_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        transform=None,
        **kwargs
    ) -> DataLoader:
        """
        创建HDF5数据的数据加载器
        
        Args:
            h5_path: HDF5文件路径
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            transform: 图像变换
            **kwargs: 其他DataLoader参数
        
        Returns:
            DataLoader实例
        """
        dataset = DMDH5Dataset(h5_path=h5_path, transform=transform)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
           
    @staticmethod
    def custom_collate_fn(batch):
        """
        自定义collate函数，处理包含PIL图像和复杂数据结构的数据
        """
        # 分离不同类型的数据
        images = []
        sigmas = []
        current_timestep = []
        x1 = []
        denoised = []
        cap_feats = []
        cap_masks = []
        v_preds = []
        t1 = []
        # v_pred_conds = []
        # v_pred_unconds = []
        # neg_feats = []
        # neg_masks = []
        # guidance_scales = []
        # shifts = []
        step_sample_num = 1
        for item in batch:
            if 'image' in item:
                images.append(item['image'])
            # filenames.append(item['filename'])
            # metadata_list.append(item['metadata'])
            if 'steps_data' in item:
                # _sigmas = []
                # _current_timestep = []
                # _x = []
                # _denoised = []
                # _cap_feats = []
                # _cap_masks = []
                # _v_preds = []

                # print(item['steps_data'])
                ids = random.sample(range(len(item['steps_data'])), step_sample_num)
                for id in ids:
                    step = item['steps_data'][id]
                    sigmas.append(step['data']['sigma'])
                    current_timestep.append(step['data']['current_timestep'])
                    x1.append(item['steps_data'][-1]['data']['denoised'])
                    t1.append(item['steps_data'][-1]['data']['sigma'])
                    denoised.append(step['data']['x'])
                    cap_feats.append(step['data']['cap_feats'])
                    cap_masks.append(step['data']['cap_mask'])
                    v_preds.append(step['data']['v_pred'])


                # stack - 保持tensor在CPU上，让模型自己移动到GPU
                if sigmas:
                    if isinstance(sigmas[0], torch.Tensor):
                        sigmas = torch.cat(sigmas)
                    else:
                        sigmas = torch.tensor(np.array(sigmas))
                    sigmas = sigmas.squeeze(1)
                if current_timestep:
                    if isinstance(current_timestep[0], torch.Tensor):
                        current_timestep = torch.cat(current_timestep)
                    else:
                        current_timestep = torch.tensor(np.array(current_timestep))
                    current_timestep = current_timestep.squeeze(1)
                if x1:
                    if isinstance(x1[0], torch.Tensor):
                        x1 = torch.cat(x1)
                    else:
                        x1 = torch.tensor(np.array(x1))
                    x1 = x1.squeeze(1)
                if denoised:
                    if isinstance(denoised[0], torch.Tensor):
                        denoised = torch.cat(denoised)
                    else:
                        denoised = torch.tensor(np.array(denoised))
                    denoised = denoised.squeeze(1)
                if cap_feats:
                    if isinstance(cap_feats[0], torch.Tensor):
                        cap_feats = torch.cat(cap_feats)
                    else:
                        cap_feats = torch.tensor(np.array(cap_feats))
                    cap_feats = cap_feats.squeeze(1)
                if cap_masks:
                    if isinstance(cap_masks[0], torch.Tensor):
                        cap_masks = torch.cat(cap_masks)
                    else:
                        cap_masks = torch.tensor(np.array(cap_masks))
                    cap_masks = cap_masks.squeeze(1)
                if v_preds:
                    if isinstance(v_preds[0], torch.Tensor):
                        v_preds = torch.cat(v_preds)
                    else:
                        v_preds = torch.tensor(np.array(v_preds))
                    v_preds = v_preds.squeeze(1)





                #     _sigmas.append(step['data']['sigma'])
                #     _current_timestep.append(step['data']['current_timestep'])
                #     _x.append(step['data']['x'])
                #     _denoised.append(step['data']['denoised'])
                #     _cap_feats.append(step['data']['cap_feats'])
                #     _cap_masks.append(step['data']['cap_mask'])
                #     _v_preds.append(step['data']['v_pred'])
                #     # v_pred_conds.append(step['data']['v_pred_cond'])
                #     # v_pred_unconds.append(step['data']['v_pred_uncond'])
                #     # neg_feats.append(step['data']['neg_feats'])
                #     # neg_masks.append(step['data']['neg_mask'])
                #     # guidance_scales.append(step['data']['guidance_scale'])
                #     # shifts.append(step['data']['shift'])
                # sigmas.append(_sigmas)
                # current_timestep.append(_current_timestep)
                # x.append(_x)
                # denoised.append(_denoised)
                # cap_feats.append(_cap_feats)
                # cap_masks.append(_cap_masks)
                # v_preds.append(_v_preds)

        # 构建结果字典，匹配模型期望的输入格式
        
        result = {
            'sigmas': sigmas,
            'current_timestep': current_timestep,
            'x1': x1,  # 目标图像
            'denoised': denoised,
            'cap_feats': cap_feats,
            'cap_masks': cap_masks.to(torch.int32),
            'v_preds': v_preds,
            't1': t1,
            # 'prompt_embeds': cap_feats,  # 为了兼容模型输入
            # 'prompt_masks': cap_masks,   # 为了兼容模型输入
        }
        
        # 如果有图像，转换为tensor
        if images:
            if isinstance(images[0], Image.Image):
                # 如果是PIL图像，需要转换为tensor
                # 这里假设transform已经处理了转换，如果没有则手动转换
                try:
                    result['image'] = torch.stack([flip_norm(img) for img in images])
                except:
                    # 如果转换失败，保持PIL格式
                    result['image'] = images
            else:
                result['image'] = images
        
        return result


def load_steps_data_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    从单个文件加载steps数据
    
    Args:
        file_path: steps数据文件路径
    
    Returns:
        steps数据列表
    """
    with open(file_path, 'rb') as f:
        steps_data = pickle.load(f)
    return steps_data


def save_steps_data_to_file(steps_data: List[Dict[str, Any]], file_path: str):
    """
    保存steps数据到文件
    
    Args:
        steps_data: steps数据
        file_path: 保存路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(steps_data, f)



def extract_steps_features(steps_data: List[Dict[str, Any]], feature_keys: List[str]) -> Dict[str, np.ndarray]:
    """
    从steps数据中提取特定特征
    
    Args:
        steps_data: steps数据列表
        feature_keys: 要提取的特征键名列表
    
    Returns:
        特征字典，每个特征对应一个numpy数组
    """
    features = {key: [] for key in feature_keys}
    
    for step_info in steps_data:
        step_data = step_info["data"]
        for key in feature_keys:
            if key in step_data and step_data[key] is not None:
                features[key].append(step_data[key])
    
    # 转换为numpy数组
    for key in features:
        if features[key]:
            features[key] = np.array(features[key])
        else:
            features[key] = np.array([])
    
    return features


def visualize_steps_data(steps_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    可视化steps数据（简单统计信息）
    
    Args:
        steps_data: steps数据
        save_path: 保存路径，None表示不保存
    """
    if not steps_data:
        print("没有steps数据")
        return
    
    print(f"Steps数据统计:")
    print(f"  总步数: {len(steps_data)}")
    
    # 分析第一个step的数据结构
    first_step = steps_data[0]["data"]
    print(f"  数据键: {list(first_step.keys())}")
    
    # 统计每个键的数据形状
    for key, value in first_step.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")


# 使用示例
if __name__ == "__main__":
    # 示例1: 加载steps数据集
    data_dir = "/mnt/hz_trainer/batch_output_20250827_112221"  # 替换为实际路径
    
    try:
        # 创建数据集
        dataset = DMDStepsDataset(
            data_dir=data_dir,
            load_images=True,
            load_steps_data=True,
            max_samples=None # 只加载前10个样本用于测试
        )
        
        # 打印统计信息
        print("提示词统计:")
        prompt_stats = dataset.get_prompt_statistics()
        for key, value in prompt_stats.items():
            print(f"  {key}: {value}")
        
        print("\nSteps数据统计:")
        steps_stats = dataset.get_steps_statistics()
        for key, value in steps_stats.items():
            print(f"  {key}: {value}")
        
        # 加载第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n第一个样本:")
            print(f"  文件名: {sample['filename']}")
            print(f"  提示词: {sample['metadata']['prompt'][:100]}...")
            print(f"  图像尺寸: {sample['image'].size if 'image' in sample else 'N/A'}")
            print(f"  Steps数量: {len(sample['steps_data']) if 'steps_data' in sample else 'N/A'}")
            
            # 可视化steps数据
            if 'steps_data' in sample:
                visualize_steps_data(sample['steps_data'])
        
    except Exception as e:
        print(f"加载数据集时出错: {e}")
    
    # 示例2: 使用DataLoader
    dataloader = DMDDataLoader.create_steps_dataloader(
        data_dir=data_dir,
        batch_size=1,
        shuffle=False,
        max_samples=4
    )
    
    print(f"\nDataLoader测试:")
    for batch_idx, batch in enumerate(dataloader):
        print(f"  批次 {batch_idx}: {len(batch['sigmas'])} 个样本")
        # print(batch['steps_data'][0].keys())
        if batch_idx >= 1:  # 只显示前2个批次
            print(batch['sigmas'])
            print(batch['current_timestep'])
            print(batch['t1'])
            # print(batch['x1'])
            print(batch['denoised'].shape)
            # print(batch['cap_feats'])
            # print(batch['cap_masks'])
            # print(batch['v_preds'])
            break

                