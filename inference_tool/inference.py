from torch import Tensor
import inference_tool.k_diffusion_adapter as k_diffusion_adapter
import torch
from typing import Union, List
from tqdm import tqdm
from typing import Callable

def denoise(
    scheduler,
    model,
    img: Tensor,
    txt: Tensor,
    txt_mask: Tensor,
    neg_txt: Tensor,
    neg_txt_mask: Tensor,
    timesteps: Union[List[float], torch.Tensor],
    guidance_scale: float = 5.0,
    cfg_trunc_ratio: float = 1.0,
    renorm_cfg: float = 1.0,
    hook_fn: Callable = None,
):
    for i, t in enumerate(tqdm(timesteps)):
        # model.prepare_block_swap_before_forward()

        # reverse the timestep since Lumina uses t=0 as the noise and t=1 as the image
        current_timestep = 1 - t / scheduler.config.num_train_timesteps
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        current_timestep = current_timestep * torch.ones(
            img.shape[0], device=img.device
        )

        noise_pred_cond = model(
            img,
            current_timestep,
            cap_feats=txt,  # Gemma2的hidden states作为caption features
            cap_mask=txt_mask.to(dtype=torch.int32),  # Gemma2的attention mask
        )

        # compute whether to apply classifier-free guidance based on current timestep
        if current_timestep[0] < cfg_trunc_ratio:
            # model.prepare_block_swap_before_forward()
            noise_pred_uncond = model(
                img,
                current_timestep,
                cap_feats=neg_txt,  # Gemma2的hidden states作为caption features
                cap_mask=neg_txt_mask.to(dtype=torch.int32),  # Gemma2的attention mask
            )
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            # apply normalization after classifier-free guidance
            if float(renorm_cfg) > 0.0:
                cond_norm = torch.linalg.vector_norm(
                    noise_pred_cond,
                    dim=tuple(range(1, len(noise_pred_cond.shape))),
                    keepdim=True,
                )
                max_new_norms = cond_norm * float(renorm_cfg)
                noise_norms = torch.linalg.vector_norm(
                    noise_pred, dim=tuple(range(1, len(noise_pred.shape))), keepdim=True
                )
                # Iterate through batch
                for i, (noise_norm, max_new_norm) in enumerate(zip(noise_norms, max_new_norms)):
                    if noise_norm >= max_new_norm:
                        noise_pred[i] = noise_pred[i] * (max_new_norm / noise_norm)
        else:
            noise_pred = noise_pred_cond

        img_dtype = img.dtype

        if img.dtype != img_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                img = img.to(img_dtype)

        # compute the previous noisy sample x_t -> x_t-1
        noise_pred = -noise_pred
        img = scheduler.step(noise_pred, t, img, return_dict=False)[0]

    # model.prepare_block_swap_before_forward()
    return img

def denoise_k_diffusion(
    model,
    img: Tensor,
    txt: Tensor,
    txt_mask: Tensor,
    neg_txt: Tensor,
    neg_txt_mask: Tensor,
    steps: int,
    guidance_scale: float = 4.0,
    cfg_trunc_ratio: float = 1.0,
    renorm_cfg: float = 1.0,
    sampler: str = "euler",
    scheduler_func: str = "normal_scheduler",
    shift: float = 6.0,
    hook_fn: Callable = None,
    **sampler_kwargs
):
    """
    使用 k_diffusion 采样函数进行去噪
    """
    return k_diffusion_adapter.sample_with_k_diffusion(
        model=model,
        text_encoder_hidden_states=txt,
        text_encoder_attention_mask=txt_mask,
        neg_hidden_states=neg_txt,
        neg_attention_mask=neg_txt_mask,
        latents=img,
        steps=steps,
        guidance_scale=guidance_scale,
        cfg_trunc_ratio=cfg_trunc_ratio,
        renorm_cfg=renorm_cfg,
        sampler=sampler,
        scheduler_func=scheduler_func,
        shift=shift,
        hook_fn=hook_fn,
        **sampler_kwargs
    )
    