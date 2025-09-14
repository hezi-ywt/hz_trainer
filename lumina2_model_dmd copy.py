from pathlib import Path
import os
import time
from omegaconf import OmegaConf

from safetensors.torch import safe_open, save_file
from PIL import Image
from tqdm import tqdm
import functools
from functools import partial
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint
import lightning as pl
from copy import deepcopy
import shutil
import numpy as np
import math

# from modules.scheduler_utils import apply_zero_terminal_snr, cache_snr_values
# from common.utils import get_class, load_torch_file, EmptyInitWrapper, get_world_size
from common.logging import logger
import random

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from models.lumina import models
from models.lumina.transport import create_transport, Sampler
from lightning.pytorch.utilities import rank_zero_only
from safetensors.torch import save_file
# from modules.config_sdxl_base import model_config
from diffusers.training_utils import EMAModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)

from transformers import (
    AutoTokenizer,
    AutoModel,
)
from torchvision.transforms.functional import to_pil_image




def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * len(x[0].size())
    t = t.view(t.size(0), *dims)
    return t




class Lumina2ModelDMD(pl.LightningModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.target_device = device
        self.model_path = config.model.get("model_path", None)
        self.init_model()

    def init_model(self):
        self.build_models()
        self.to(self.target_device)
        
        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.advanced.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size

    def build_models(self):
        trainer_cfg = self.config.trainer
        config = self.config
        advanced = config.get("advanced", {})
        
        #tokenizer
        if self.config.model.get("tokenizer_path", None):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.tokenizer_path,
                use_fast=False,
                local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                subfolder="tokenizer",
                use_fast=False,
                local_files_only=True
            )
        self.tokenizer.padding_side = "right"

        #text_encoder   
        if self.config.model.get("text_encoder_path", None):
            if self.config.model.text_encoder_path.endswith(".safetensors"):
                # 如果是 .safetensors 文件，使用 load_gemma2 函数
                from models.lumina.models.extra_model import load_gemma2
                self.text_encoder = load_gemma2(
                    ckpt_path=self.config.model.text_encoder_path,
                    dtype=torch.bfloat16,
                    device=self.target_device
                )
            else:
                # 如果是模型目录，使用 AutoModel.from_pretrained
                self.text_encoder = AutoModel.from_pretrained(
                    self.config.model.text_encoder_path,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16
                ).cuda()
        else:
            self.text_encoder = AutoModel.from_pretrained(
                self.model_path,
                subfolder="text_encoder",
                local_files_only=True,
                torch_dtype=torch.bfloat16
            ).cuda()


        logger.info(f"text encoder: {type(self.text_encoder)}")
        self.cap_feat_dim = self.text_encoder.config.hidden_size
         

        # Create model:
        self.model = models.__dict__[self.config.model.model_name](
            in_channels=16,
            qk_norm=self.config.model.get("qk_norm", True),
            cap_feat_dim=self.cap_feat_dim,
        ).to(dtype=torch.bfloat16)
        logger.info(f"DiT Parameters: {self.model.parameter_count():,}")






        self.model_patch_size = self.model.patch_size

        if self.config.trainer.get("auto_resume", False) and self.config.trainer.resume is None:
            try:
                existing_checkpoints = os.listdir(self.config.trainer.checkpoint_dir)
                if len(existing_checkpoints) > 0:
                    existing_checkpoints.sort()
                    self.config.trainer.resume = os.path.join(self.config.trainer.checkpoint_dir, existing_checkpoints[-1])
            except Exception:
                    pass
        if self.config.model.get("resume", None) is not None:
            checkpoint_path = os.path.join(
                self.config.model.resume,
                f"consolidated.00-of-01.pth",
            )
            if os.path.exists(checkpoint_path):
                logger.info(f"Resuming model weights from: {checkpoint_path}")
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location="cpu"),
                    strict=True,
                )
            else:
                logger.warning(f"Checkpoint not found at: {checkpoint_path}")

            
        if self.config.model.get("resume", None) is not None:
            logger.info(f"Resuming model weights from: {self.config.model.resume}")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.config.model.resume,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
             # Note that parameter initialization is done within the DiT constructor
            if self.config.advanced.get("use_ema", True):
                logger.info("Using EMA")
                self.model_ema = deepcopy(self.model)
                self.model_ema.requires_grad_(False)  # EMA 模型不需要梯度
            

            # if hasattr(self, "model_ema") and os.path.exists(os.path.join(self.config.model.resume, "consolidated_ema.00-of-01.pth")):
            #     logger.info(f"Resuming ema weights from: {self.config.model.resume}")
            #     self.model_ema.load_state_dict(
            #         torch.load(
            #             os.path.join(
            #                 self.config.model.resume,
            #                 f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
            #             ),
            #             map_location="cpu",
            #         ),
            #         strict=True,
            #     )    

        elif self.config.model.get("init_from", None) is not None:
            
            logger.info(f"Initializing model weights from: {self.config.model.init_from}")
            if self.config.model.init_from.endswith(".pth") or self.config.model.init_from.endswith(".pt") or self.config.model.init_from.endswith(".safetensors") or self.config.model.init_from.endswith(".ckpt"):
                from common.model_loader import load_model_weights
                state_dict = load_model_weights(self.config.model.init_from, device=self.target_device)
            else:
                state_dict = torch.load(
                    os.path.join(
                        self.config.model.init_from,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                )

            size_mismatch_keys = []
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if self.config.advanced.get("use_ema", True):
                self.model_ema = deepcopy(self.model)
                missing_keys_ema, unexpected_keys_ema = self.model_ema.load_state_dict(state_dict, strict=False)
                assert set(missing_keys) == set(missing_keys_ema)
                assert set(unexpected_keys) == set(unexpected_keys_ema)
            del state_dict
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")

        # checkpointing (part1, should be called before FSDP wrapping)
        if self.config.trainer.get("checkpointing", False):
            checkpointing_list = list(self.model.get_checkpointing_wrap_module_list())
            if hasattr(self, "model_ema"):
                checkpointing_list_ema = list(self.model_ema.get_checkpointing_wrap_module_list())
        else:
            checkpointing_list = []
            checkpointing_list_ema = []

        # checkpointing (part2)
        if self.config.trainer.get("checkpointing", False):
            logger.info("apply gradient checkpointing")
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: submodule in checkpointing_list,
            )
            if hasattr(self, "model_ema"):
                apply_activation_checkpointing(
                    self.model_ema,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=lambda submodule: submodule in checkpointing_list_ema,
                )

        logger.info(f"model:\n{self.model}\n")
        
        if self.config.model.get("vae_path", None):
            if self.config.model.vae_path.endswith(".safetensors"):
                from models.lumina.models.extra_model import load_ae
                self.vae = load_ae(
                    ckpt_path=self.config.model.vae_path,
                    dtype=torch.float32,
                    device=self.target_device
                )
            else:
                self.vae = AutoencoderKL.from_pretrained(
                    self.config.model.vae_path,
                    torch_dtype=torch.float32
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(
                self.model_path,
                subfolder="vae",
                torch_dtype=torch.float
            )

       



        if advanced.get("latents_mean", None):
            self.latents_mean = torch.tensor(advanced.latents_mean)
            self.latents_std = torch.tensor(advanced.latents_std)
            self.latents_mean = self.latents_mean.view(1, 4, 1, 1).to(self.target_device)
            self.latents_std = self.latents_std.view(1, 4, 1, 1).to(self.target_device)
        
        self.vae.to(self.target_device)
        self.vae.requires_grad_(False)
        self.model.to(self.target_device)
        self.model.train()
        self.model.requires_grad_(True)
        if self.config.model.get("use_real_model", False):
            self.real_model = deepcopy(self.model)
            self.real_model.requires_grad_(False)
        if self.config.model.get("use_fake_model", False):
            self.fake_model = deepcopy(self.model)
            self.fake_model.requires_grad_(True)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # self.tokenizer.to(self.target_device)
        # self.tokenizer.requires_grad_(False)


    @torch.no_grad()
    def encode_prompt(self, prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
        captions = []
        if isinstance(prompt_batch, str):
            prompt_batch = [prompt_batch]
        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        # 将输入移动到正确的设备并设置数据类型
        text_input_ids = text_inputs.input_ids.to(self.target_device)
        prompt_masks = text_inputs.attention_mask.to(self.target_device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        # 确保 prompt_embeds 的类型与 x_embedder 的 Linear 层匹配
        prompt_embeds = prompt_embeds.to(dtype=self.model.x_embedder.weight.dtype)

        return prompt_embeds, prompt_masks

    @torch.no_grad()
    def encode_images(self, images):
        # VAE编码图像
        vae_scale = {
            "sdxl": 0.13025,
            "sd3": 1.5305,
            "ema": 0.18215,
            "mse": 0.18215,
            "cogvideox": 1.15258426,
            "flux": 0.3611,
        }["flux"]
        vae_shift = {
            "sdxl": 0.0,
            "sd3": 0.0609,
            "ema": 0.0,
            "mse": 0.0,
            "cogvideox": 0.0,
            "flux": 0.1159,
        }["flux"]
        # if not isinstance(images, list):
        #     images = [images]
        
        x = [img.to(self.target_device, non_blocking=True) for img in images]
   

        for i, img in enumerate(x):
            x[i] = (self.vae.encode(img[None]).latent_dist.mode()[0] - vae_shift) * vae_scale
            x[i] = x[i].float()

        return x
        
        

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 更新EMA模型
        if hasattr(self, "model_ema"):
            self.update_ema()
    




    
    @torch.no_grad()
    def update_ema(self, decay=0.95):
        """
        Step the EMA model towards the current model.
        """

        ema_params = OrderedDict(self.model_ema.named_parameters())
        model_params = OrderedDict(self.model.named_parameters())
        assert set(ema_params.keys()) == set(model_params.keys())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


    def save_checkpoint(self, model_path, metadata):
        weight_to_save = None
        if hasattr(self, "_fsdp_engine"):
            from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
            
            weight_to_save = {}    
            world_size = self._fsdp_engine.world_size
            with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
                weight_to_save = self.model._forward_module.state_dict()
            
        elif hasattr(self, "_deepspeed_engine"):
            from deepspeed import zero
            weight_to_save = {}
            with zero.GatheredParameters(self.model.parameters()):
                weight_to_save = self.model.state_dict()
                
        else:
            weight_to_save = self.model.state_dict()
                
        self._save_checkpoint(model_path, weight_to_save, metadata)

    @rank_zero_only
    def _save_checkpoint(self, model_path, state_dict, metadata):
        cfg = self.config.trainer
        # check if any keys startswith modules. if so, remove the modules. prefix
        # if any([key.startswith("module.") for key in state_dict.keys()]):
        #     state_dict = {
        #         key.replace("module.", ""): value for key, value in state_dict.items()
        #     }

        if cfg.get("save_format") == "safetensors":

            save_file(state_dict, model_path + ".safetensors", metadata=metadata)
            if hasattr(self, "model_ema"):
                
                save_file(self.model_ema.state_dict(), model_path + "_ema.safetensors", metadata=metadata)
        elif cfg.get("save_format") == "original":
            os.makedirs(model_path, exist_ok=True)
            torch.save(state_dict, os.path.join(model_path, "consolidated.00-of-01.pth"))
            if hasattr(self, "model_ema"):
                torch.save(self.model_ema.state_dict(), os.path.join(model_path, "consolidated_ema.00-of-01.pth"))
            #copy
            arg_path = os.path.join(self.config.trainer.model_path, "model_args.pth")
            shutil.copy(arg_path, os.path.join(model_path, "model_args.pth"))
            # opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{.get_world_size():05d}.pth"
            # torch.save(self.optimizer.state_dict(), os.path.join(model_path, opt_state_fn))
        else:
            state_dict = {"state_dict": state_dict, **metadata}
            model_path += ".ckpt"
            torch.save(state_dict, model_path)
        logger.info(f"Saved model to {model_path}")


    @torch.no_grad()
    def get_noise(self, x1):
                # 生成初始噪声
        if isinstance(x1, (list, tuple)):
            x0 = [torch.randn_like(img_start) for img_start in x1]
        else:
            x0 = torch.randn_like(x1)
        return x0

    @torch.no_grad()
    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t = t
        sigma_t = 1 - t
        if isinstance(x1, (list, tuple)):
            return [alpha_t[i] * x1[i] + sigma_t[i] * x0[i] for i in range(len(x1))]
        else:
            return alpha_t * x1 + sigma_t * x0



    def forward(self, batch):
        # 获取batch数据并移动到正确的设备
        prompt_embeds = batch["cap_feats"].to(self.model.x_embedder.weight.dtype)
        prompt_masks = batch["cap_masks"].to(torch.int32)
        sigmas = batch["sigmas"]
        latents = batch["denoised"]  # 使用denoised作为当前噪声图像
        x1 = batch["x1"]  # 目标图像
        batch_size = x1.shape[0]

        # image = self.vae.decode(x1)
        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # image = (image * 255).round().astype("uint8")
        # pil_image = Image.fromarray(image[0])
        # pil_image.save(f"test_denoised{time.time()}.png")
        # image = batch["image"]
        # image = image.to(self.target_device)
        # image = image.to(self.model.x_embedder.weight.dtype)
        # image = image.to(self.model.x_embedder.weight.dtype)
        # latents = self.vae.encode(image)
        # latents = latents.to(self.target_device)
        # latents = latents.to(self.model.x_embedder.weight.dtype)
        # image = self.vae.decode(latents)
        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # image = (image * 255).round().astype("uint8")
        # pil_image = Image.fromarray(image[0])
        # pil_image.save(f"test_latents_raw{time.time()}.png")
        # time.sleep(5)
        xt = latents
        ut = (x1 - xt) / sigmas.view(-1, 1, 1, 1)
        t = 1 - sigmas
        
        # 前向传播
        v_pred = self.model(xt, t, cap_feats=prompt_embeds, cap_mask=prompt_masks)
        # x1_pred = xt + v_pred * sigmas.view(-1, 1, 1, 1)
        # 计算损失
        terms = {}
        terms["task_loss"] = torch.mean(((v_pred - ut) ** 2), dim=list(range(1, len(x1.size()))))
        terms["loss"] = terms["task_loss"]
        terms["task_loss"] = terms["task_loss"].clone().detach()
        terms["t"] = t
        
        loss = terms["loss"]
        
        # 返回损失和额外的信息，让调用者处理日志记录
        return {
            "loss": loss,
            "task_loss": terms["task_loss"],
            "t": t
        }

    @torch.inference_mode()
    def generate_image(
        self,
        model,
        prompt,
        negative_prompt,
        device = "cuda",
        dtype = torch.bfloat16,
        hook_fn = None,
        config = None,
    ):
        #
        # 0. Prepare arguments
        #
        #
        # 2. Encode prompts
        #
        try:
            model.eval()
            self.vae.eval()
            self.text_encoder.eval()
        except Exception as e:
            print(e)
            pass
        
        from inference_tool.inference import denoise_k_diffusion, denoise
        from inference_tool.ori_schedulers import FlowMatchEulerDiscreteScheduler

        # Unpack Gemma2 outputs
        prompt_hidden_states,  prompt_attention_mask = self.encode_prompt(
            prompt,
            self.text_encoder,
            self.tokenizer,
            proportion_empty_prompts=0.0,
            is_train=False
        )
        uncond_hidden_states,  uncond_attention_mask = self.encode_prompt(
            negative_prompt,
            self.text_encoder,
            self.tokenizer,
            proportion_empty_prompts=0.0,
            is_train=False
        )

        # if config.device.offload:
        #     print("Offloading models to CPU to save VRAM...")
        #     gemma2.to("cpu")
        #     clean_memory()

        # model.to(device)

        #
        # 3. Prepare latents
        #
        seed = config.generation.seed if config.generation.seed is not None else random.randint(0, 2**32 - 1)
        logger.info(f"Seed: {seed}")
        torch.manual_seed(seed)

        latent_height = config.generation.image_height // 8
        latent_width = config.generation.image_width // 8
        latent_channels = 16

        latents = torch.randn(
            (1, latent_channels, latent_height, latent_width),
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        #
        # 4. Denoise
        #
        logger.info("Denoising...")
        
        with torch.autocast(device_type=device, dtype=dtype), torch.no_grad():
            if config.k_diffusion.use_k_diffusion:
                # 使用 k_diffusion 采样
                latents = denoise_k_diffusion(
                    model,
                    latents.to(device),
                    prompt_hidden_states.to(device),
                    prompt_attention_mask.to(device),
                    uncond_hidden_states.to(device),
                    uncond_attention_mask.to(device),
                    config.generation.steps,
                    config.generation.guidance_scale,
                    config.generation.cfg_trunc_ratio,
                    config.generation.renorm_cfg,
                    sampler=config.k_diffusion.sampler,
                    scheduler_func=config.k_diffusion.scheduler_func,
                    shift=config.generation.discrete_flow_shift,
                    hook_fn=hook_fn,
                )
            else:
                # 使用原始的调度器
                scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=config.generation.discrete_flow_shift)
                scheduler.set_timesteps(config.generation.steps, device=device)
                timesteps = scheduler.timesteps
                
                latents = denoise(
                    scheduler,
                    model,
                    latents.to(device),
                    prompt_hidden_states.to(device),
                    prompt_attention_mask.to(device),
                    uncond_hidden_states.to(device),
                    uncond_attention_mask.to(device),
                    timesteps,
                    config.generation.guidance_scale,
                    config.generation.cfg_trunc_ratio,
                    config.generation.renorm_cfg,
                )

        # if config.device.offload:
        #     model.to("cpu")
        #     clean_memory()
        #     ae.to(device)

        #
        # 5. Decode latents
        #
        logger.info("Decoding image...")
        latents = latents.to(self.vae.dtype)
        # latents = latents / ae.scale_factor + ae.shift_factor
        with torch.no_grad():
            if isinstance(self.vae, AutoencoderKL):
                image = self.vae.decode(latents / self.vae.config.scaling_factor + self.vae.config.shift_factor)[0]
            else:
                image = self.vae.decode(latents, refine=True)

        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")

        #
        # 6. Save image
        #
        pil_image = Image.fromarray(image[0])
        output_dir = config.output.output_dir
        os.makedirs(output_dir, exist_ok=True)
        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        seed_suffix = f"_{seed}"
        output_path = os.path.join(output_dir, f"image_{ts_str}{seed_suffix}.png")
        pil_image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        return pil_image




if __name__ == "__main__":


    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "model": {
            "model_name": "NextDiT_2B_GQA_patch2_Adaln_Refiner",
            "model_path": "/mnt/huggingface/Lumina-Image-2.0",
            "vae_path": "/mnt/huggingface/Lumina-Image-2.0/vae",
            "text_encoder_path": "/mnt/huggingface/Neta-Lumina/Text Encoder/gemma_2_2b_fp16.safetensors",
            # "tokenizer_path": "/mnt/huggingface/Neta-Lumina-diffusers/tokenizer",
            # "init_from": "/mnt/huggingface/Neta-Lumina/Unet/neta-lumina-v1.0.safetensors",
            "init_from": "/mnt/hz_trainer/checkpoints/dmd_model_epoch=97_step=224322_val_loss=0.0000.safetensors",
            "use_ema": False,
            "use_fake_model": False,
            "use_real_model": False,
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
            "steps": 15,
            "guidance_scale": 3.0,
            "cfg_trunc_ratio": 1.0,
            "seed": 114514,
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



    print(config.model.model_name)


    model = Lumina2ModelDMD(config, device="cuda")

    # images = model.sample(prompt="you are an assistant designed to generate anime images based on textual prompts. <Prompt Start> The artwork, by terigumi, presents a POV shot of Jane Doe from Zenless Zone Zero, kneeling and looking down at the viewer with a flirty smile. Jane has short, silver-gray hair, large aqua eyes and pink animal ears, as well as a black tail with a gold-colored tip. She wears a revealing black outfit that exposes her cleavage, paired with black pantyhose. Her arms extend towards the viewer, as if to embrace them.",
    # negative_prompt="You are an assistant designed to generate anime images based on textual prompts. <Prompt Start> blurry, worst quality, low quality, jpeg artifacts, signature, watermark, username, error, deformed hands, bad anatomy, extra limbs, poorly drawn hands, poorly drawn face, mutation, deformed, extra eyes, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross‑eyed, bad proportions, missing arms, missing legs, extra digit, fewer digits, cropped, normal quality",
    # size=(1024,1024),steps=25,guidance_scale=5.0,solver="euler",path_type="Linear",prediction="velocity")


    global diffusion_steps_data
    diffusion_steps_data = {}
    def get_sample_hook(**kwargs):
        # print(kwargs["current_timestep"])
        # print(kwargs.get("sigma",None))

        diffusion_steps_data[kwargs["current_timestep"].item()] = {
            "sigma": kwargs.get("sigma", None),
            "denoised": kwargs.get("denoised", None),
            "current_timestep": kwargs.get("current_timestep", None),
            "cap_feats": kwargs.get("cap_feats", None),
            "cap_mask": kwargs.get("cap_mask", None),
            "neg_feats": kwargs.get("neg_feats", None),
            "neg_mask": kwargs.get("neg_mask", None),
            "v_pred": kwargs.get("v_pred", None),
            "v_pred_cond": kwargs.get("v_pred_cond", None),
            "v_pred_uncond": kwargs.get("v_pred_uncond", None),
            "guidance_scale": kwargs.get("guidance_scale", None),
            "shift": kwargs.get("shift", None),
        }
        
        # return kwargs

    model.generate_image(model.model,
        # prompt="you are an assistant designed to generate anime images based on textual prompts. <Prompt Start> The artwork, by terigumi, presents a POV shot of Jane Doe from Zenless Zone Zero, kneeling and looking down at the viewer with a flirty smile. Jane has short, silver-gray hair, large aqua eyes and pink animal ears, as well as a black tail with a gold-colored tip. She wears a revealing black outfit that exposes her cleavage, paired with black pantyhose. Her arms extend towards the viewer, as if to embrace them.",
        prompt="You are an assistant designed to generate anime images based on textual prompts. <Prompt Start> freng style, Neta, girl, young woman, leaning on hand, smoking, cigarette, black hair, long hair, buns, purple and pink intake hair, purple eyes, shirt, blazer, red tie, black pantyhose, high heels, black heels, seated, on desk, office chair, looking up, thoughtful, smoke",
        negative_prompt="You are an assistant designed to generate anime images based on textual prompts. <Prompt Start> blurry, worst quality, low quality, jpeg artifacts, signature, watermark, username, error, deformed hands, bad anatomy, extra limbs, poorly drawn hands, poorly drawn face, mutation, deformed, extra eyes, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross‑eyed, bad proportions, missing arms, missing legs, extra digit, fewer digits, cropped, normal quality",
        config=config,
        hook_fn=get_sample_hook
        )
    # print(diffusion_steps_data)

    from dataset.load_dmd_dataset import DMDDataLoader
    # dataloader = DMDDataLoader.create_steps_dataloader(
    #     data_dir="/mnt/hz_trainer/batch_output_20250827_112221",
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     load_images=True,
    #     load_steps_data=True,
    #     transform=None,
    #     step_sample_num=1,
    #     max_samples=None,
    # )
    # for batch in dataloader:
    #     print(batch)
    #     x1 = batch["x1"]
    #     # print(x1.shape)
    #     # x1 = x1.to(model.target_device)
    #     # x1 = x1.to(model.vae.dtype)
    #     # print(x1.dtype)
    #     # print(model.vae.dtype)
    #     # print(model.vae.config.scaling_factor)
    #     # print(model.vae.config.shift_factor)
    #     # if isinstance(model.vae, AutoencoderKL):
    #     #     latents = (model.vae.encode(x1).latent_dist.mode()[0] - model.vae.config.shift_factor) * model.vae.config.scaling_factor
    #     # else:
    #     #     latents = model.vae.encode(x1)
    #     # print(x1.shape)

    #     latents = x1.to(model.target_device)
    #     latents = latents.to(model.vae.dtype)
    #     print(latents.shape)
    #     # latents = latents.unsqueeze(0)
    #     image = model.vae.decode(latents)
    #     # 处理DecoderOutput对象
    #     if hasattr(image, 'sample'):
    #         image = image.sample
    #     image = (image / 2 + 0.5).clamp(0, 1)
    #     image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    #     image = (image * 255).round().astype("uint8")
    #     pil_image = Image.fromarray(image[0])
    #     pil_image.save(f"test_x1{time.time()}.png")
        

    #     break




    # for image in images:
    #     image.save("test.png")