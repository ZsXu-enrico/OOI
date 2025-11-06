#!/usr/bin/env python3
"""
Unified SDS (Score Distillation Sampling) Implementation
Supports both MVDream and Stable Diffusion XL for 3D generation

Based on:
- GradeADreamer's MVDream SDS implementation
- Stable Diffusion XL for high-quality guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ==================== Helper Classes ====================

class SpecifyGradient(torch.autograd.Function):
    """
    Custom gradient function to directly specify the gradient for SDS.
    Based on DreamerXL's implementation.
    """
    @staticmethod
    def forward(ctx, input_tensor, gradient):
        ctx.save_for_backward(gradient)
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        gradient, = ctx.saved_tensors
        return gradient, None


class SDSGuidance(nn.Module):
    """
    Unified SDS guidance supporting both MVDream and Stable Diffusion XL

    Usage:
        # MVDream (multi-view diffusion)
        sds = SDSGuidance(device, model_type='mvdream', guidance_scale=100.0)

        # Stable Diffusion XL
        sds = SDSGuidance(device, model_type='sdxl', guidance_scale=7.5)
    """

    def __init__(
        self,
        device: torch.device,
        model_type: str = 'mvdream',  # 'mvdream' or 'sdxl'
        model_name: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        guidance_scale: float = 100.0,
        t_range: Tuple[float, float] = (0.02, 0.98),
        use_fp16: bool = True,
    ):
        """
        Args:
            device: torch device
            model_type: 'mvdream' or 'sdxl'
            model_name: Model name/path (optional, uses defaults)
            ckpt_path: Checkpoint path for MVDream (optional)
            guidance_scale: Classifier-free guidance scale
            t_range: Timestep sampling range (min, max)
            use_fp16: Use float16 for efficiency
        """
        super().__init__()

        self.device = device
        self.model_type = model_type.lower()
        self.guidance_scale = guidance_scale
        self.t_range = t_range
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Initialize the appropriate model
        if self.model_type == 'mvdream':
            self._init_mvdream(model_name, ckpt_path)
        elif self.model_type == 'sdxl':
            self._init_sdxl(model_name)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'mvdream' or 'sdxl'")

        # Timestep range
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        # Alpha values for SDS
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        # Text embeddings storage
        self.embeddings = {}

        logger.info(f"Initialized {model_type.upper()} SDS guidance:")
        logger.info(f"  Guidance scale: {guidance_scale}")
        logger.info(f"  Timestep range: {t_range} -> steps [{self.min_step}, {self.max_step}]")

    def _init_mvdream(self, model_name: Optional[str], ckpt_path: Optional[str]):
        """Initialize MVDream model"""
        logger.info("Loading MVDream model...")

        from mvdream.camera_utils import normalize_camera
        from mvdream.model_zoo import build_model
        from diffusers import DDIMScheduler

        self.normalize_camera = normalize_camera

        # Model name defaults
        if model_name is None:
            model_name = 'sd-v2.1-base-4view'

        self.model_name = model_name
        self.ckpt_path = ckpt_path

        # Build MVDream model
        # If ckpt_path is None, it will auto-download from HuggingFace
        # Use cache_dir to specify where to cache the models
        cache_dir = '/data4/zishuo/mvdream_models'
        self.model = build_model(model_name, ckpt_path=ckpt_path, cache_dir=cache_dir).eval().to(self.device)
        self.model.device = self.device

        # Scheduler (with cache dir)
        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="scheduler",
            torch_dtype=self.dtype,
            cache_dir="/data4/zishuo/huggingface_cache"
        )

        logger.info(f"  Model: {model_name}")
        logger.info(f"  4-view multi-view diffusion enabled")
        logger.info(f"  Model cache directory: {cache_dir}")

    def _init_sdxl(self, model_name: Optional[str]):
        """Initialize Stable Diffusion XL model"""
        logger.info("Loading Stable Diffusion XL model...")

        from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL

        # Model name defaults
        if model_name is None:
            model_name = 'stabilityai/stable-diffusion-xl-base-1.0'

        self.model_name = model_name

        # CRITICAL: Load VAE separately in float32 to avoid fp16 NaN issues
        # Following Dreamer-XL's approach
        vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=torch.float32  # Always use float32 for VAE
        )

        # Load SDXL pipeline (with cache dir)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            vae=vae,  # Use float32 VAE
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None,
            cache_dir="/data4/zishuo/huggingface_cache"
        )
        pipe.to(self.device)

        # Extract components
        self.vae = pipe.vae  # VAE is float32
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.model = pipe.unet

        # Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler",
            torch_dtype=self.dtype
        )

        del pipe
        torch.cuda.empty_cache()

        logger.info(f"  Model: {model_name}")
        logger.info(f"  Using SDXL for high-quality single-view guidance")

    @torch.no_grad()
    def get_text_embeds(self, prompt: str, negative_prompt: str = ""):
        """
        Encode text prompts to embeddings

        Args:
            prompt: Positive text prompt
            negative_prompt: Negative text prompt
        """
        if self.model_type == 'mvdream':
            self._get_text_embeds_mvdream(prompt, negative_prompt)
        elif self.model_type == 'sdxl':
            self._get_text_embeds_sdxl(prompt, negative_prompt)

    def _get_text_embeds_mvdream(self, prompt: str, negative_prompt: str):
        """Get text embeddings for MVDream (4 views)"""
        pos_embeds = self.model.get_learned_conditioning([prompt]).to(self.device)
        neg_embeds = self.model.get_learned_conditioning([negative_prompt]).to(self.device)

        # Repeat for 4 views
        self.embeddings['pos'] = pos_embeds.repeat(4, 1, 1)  # [4, 77, 768]
        self.embeddings['neg'] = neg_embeds.repeat(4, 1, 1)

        logger.info(f"Encoded MVDream text embeddings (4 views)")

    def _get_text_embeds_sdxl(self, prompt: str, negative_prompt: str):
        """Get text embeddings for SDXL (dual text encoders)"""
        # SDXL uses two text encoders
        def encode_prompt_sdxl(prompt_text):
            # First text encoder
            text_inputs = self.tokenizer(
                prompt_text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            prompt_embeds = self.text_encoder(text_input_ids, output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]  # Penultimate layer

            # Second text encoder
            text_inputs_2 = self.tokenizer_2(
                prompt_text,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids_2 = text_inputs_2.input_ids.to(self.device)
            prompt_embeds_2 = self.text_encoder_2(text_input_ids_2, output_hidden_states=True)
            pooled_prompt_embeds_2 = prompt_embeds_2[0]
            prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

            # Concatenate embeddings from both encoders
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

            return prompt_embeds, pooled_prompt_embeds_2

        pos_embeds, pooled_pos = encode_prompt_sdxl(prompt)
        neg_embeds, pooled_neg = encode_prompt_sdxl(negative_prompt)

        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
        self.embeddings['pooled_pos'] = pooled_pos
        self.embeddings['pooled_neg'] = pooled_neg

        logger.info(f"Encoded SDXL text embeddings")

    def train_step(
        self,
        pred_rgb: torch.Tensor,
        step: int,
        camera: Optional[torch.Tensor] = None,
        azimuth: Optional[float] = None,
        as_latent: bool = False,
        return_sds_images: bool = False,
        step_ratio: Optional[float] = None,
    ):
        """
        Compute SDS loss for a rendered image

        Args:
            pred_rgb: Rendered RGB image [B, C, H, W] or [B, H, W, C] in [0, 1]
                     For MVDream: B must be multiple of 4 (4 views)
                     For SDXL: B can be any size
            step: Current training step (for logging purposes)
            camera: Camera parameters [B, 4, 4] (required for MVDream)
            azimuth: Camera azimuth angle in radians (optional, for view-dependent prompts in SDXL)
            as_latent: If True, pred_rgb is already in latent space
            return_sds_images: If True, return (loss, sds_images, timestep) tuple
            step_ratio: Training progress ratio [0, 1]. If provided, use annealing from high to low noise.
                       If None, randomly sample timestep.

        Returns:
            loss: SDS loss value (scalar)
            OR (loss, sds_images, timestep) if return_sds_images=True
        """
        # Handle input format: convert [B, H, W, C] to [B, C, H, W] if needed
        if pred_rgb.ndim == 4 and pred_rgb.shape[-1] == 3:
            pred_rgb = pred_rgb.permute(0, 3, 1, 2)

        if self.model_type == 'mvdream':
            return self._train_step_mvdream(pred_rgb, step, camera, as_latent, return_sds_images, step_ratio)
        elif self.model_type == 'sdxl':
            return self._train_step_sdxl(pred_rgb, step, azimuth, as_latent, return_sds_images, step_ratio)

    def _train_step_mvdream(
        self,
        pred_rgb: torch.Tensor,
        step: int,
        camera: torch.Tensor,
        as_latent: bool = False,
        return_sds_images: bool = False,
        step_ratio: Optional[float] = None,
    ):
        """
        MVDream SDS loss computation (following DreamGaussian's timestep annealing)

        Args:
            pred_rgb: [B, 3, H, W] where B is multiple of 4
            camera: [B, 4, 4] camera matrices
            return_sds_images: If True, return (loss, sds_images, timestep) tuple
            step_ratio: Training progress [0, 1] for timestep annealing

        Returns:
            loss or (loss, sds_images, timestep) if return_sds_images=True
        """
        batch_size = pred_rgb.shape[0]
        assert batch_size % 4 == 0, f"MVDream requires batch size multiple of 4, got {batch_size}"

        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)

        # Encode to latents
        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # Resize to 256x256 for VAE
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
            latents = self._encode_imgs_mvdream(pred_rgb_256)

        # Timestep sampling (following DreamGaussian)
        if step_ratio is not None:
            # Annealing: high noise -> low noise as training progresses
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            # Random sampling
            t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)

        # Convert camera to MVDream format (blender convention)
        camera = camera[:, [0, 2, 1, 3]]  # Flip y & z axis
        camera[:, 1] *= -1
        camera = self.normalize_camera(camera).view(batch_size, 16)

        # Prepare embeddings for CFG
        camera = camera.repeat(2, 1)
        embeddings = torch.cat([
            self.embeddings['neg'].repeat(real_batch_size, 1, 1),
            self.embeddings['pos'].repeat(real_batch_size, 1, 1)
        ], dim=0)
        context = {
            "context": embeddings,
            "camera": camera,
            "num_frames": 4
        }

        # Predict noise with UNet (no grad)
        with torch.no_grad():
            # Add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)

            # Predict noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # SDS gradient with DreamerXL weight function
        # w = sqrt((1 - alpha) / alpha)  - This is the standard SDS weight
        w = (((1 - self.alphas[t]) / self.alphas[t]) ** 0.5).view(batch_size, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # Clamp gradients to avoid NaN (important for numerical stability)
        grad = grad.clamp(-1, 1)

        # Use SpecifyGradient to apply custom gradient (DreamerXL style)
        latents_with_grad = SpecifyGradient.apply(latents, grad)

        # Convert to scalar loss for backpropagation
        loss = latents_with_grad.sum() / batch_size

        # Optionally return the predicted clean images
        if return_sds_images:
            with torch.no_grad():
                # Compute predicted clean latent (x0) using DDPM formula
                # x0 = (x_t - sqrt(1-alpha_t) * noise_pred) / sqrt(alpha_t)
                sqrt_alpha_t = torch.sqrt(self.alphas[t]).view(batch_size, 1, 1, 1)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas[t]).view(batch_size, 1, 1, 1)
                pred_x0 = (latents_noisy - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

                # Decode to RGB images
                sds_images = self.decode_latents(pred_x0)  # [B, 3, H, W]

            return loss, sds_images, t[0].item()  # Return timestep value
        else:
            return loss

    def _train_step_sdxl(
        self,
        pred_rgb: torch.Tensor,
        step: int,
        azimuth: Optional[float] = None,
        as_latent: bool = False,
        return_sds_images: bool = False,
        step_ratio: Optional[float] = None,
    ):
        """
        Stable Diffusion XL SDS loss computation (following DreamGaussian's timestep annealing)

        Args:
            pred_rgb: [B, 3, H, W] in [0, 1]
            azimuth: Camera azimuth (optional, for view-dependent prompts)
            return_sds_images: If True, return (loss, sds_images, timestep) tuple
            step_ratio: Training progress [0, 1] for timestep annealing

        Returns:
            loss or (loss, sds_images, timestep) if return_sds_images=True
        """
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        # Encode to latents
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # Resize to 512x512 for VAE
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            latents = self._encode_imgs_sdxl(pred_rgb_512)

        # Timestep sampling (following DreamGaussian)
        if step_ratio is not None:
            # Annealing: high noise -> low noise as training progresses
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            # Random sampling
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        # Get text embeddings
        text_embeds = self.embeddings['pos'].expand(batch_size, -1, -1)
        neg_embeds = self.embeddings['neg'].expand(batch_size, -1, -1)

        # SDXL additional conditioning
        pooled_embeds = self.embeddings['pooled_pos'].expand(batch_size, -1)
        pooled_neg = self.embeddings['pooled_neg'].expand(batch_size, -1)

        # Get original size and target size
        original_size = (512, 512)
        target_size = (512, 512)
        crops_coords_top_left = (0, 0)

        # Get add_time_ids
        add_time_ids = self._get_add_time_ids_sdxl(
            original_size, crops_coords_top_left, target_size, batch_size
        )

        # Prepare conditioning for CFG
        with torch.no_grad():
            # Add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # Predict noise with CFG
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            encoder_hidden_states = torch.cat([neg_embeds, text_embeds])
            add_text_embeds = torch.cat([pooled_neg, pooled_embeds])
            add_time_ids_input = torch.cat([add_time_ids] * 2)

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids_input,
            }

            noise_pred = self.model(
                latent_model_input,
                tt,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Debug: Check noise prediction
            if torch.isnan(noise_pred).any():
                logger.error(f"NaN detected in noise prediction!")
                logger.error(f"  noise_pred_uncond: min={noise_pred_uncond.min()}, max={noise_pred_uncond.max()}, has_nan={torch.isnan(noise_pred_uncond).any()}")
                logger.error(f"  noise_pred_text: min={noise_pred_text.min()}, max={noise_pred_text.max()}, has_nan={torch.isnan(noise_pred_text).any()}")

        # SDS gradient with DreamerXL weight function
        # w = sqrt((1 - alpha) / alpha)  - This is the standard SDS weight
        # Avoids the issue where w becomes too small when alpha approaches 1
        w = (((1 - self.alphas[t]) / self.alphas[t]) ** 0.5).view(batch_size, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # Clamp gradients to avoid NaN (important for numerical stability)
        grad = grad.clamp(-1, 1)

        # Use SpecifyGradient to apply custom gradient (DreamerXL style)
        # This directly sets the gradient instead of using MSE loss
        latents_with_grad = SpecifyGradient.apply(latents, grad)

        # Convert to scalar loss for backpropagation
        # The actual gradient is already set by SpecifyGradient
        loss = latents_with_grad.sum() / batch_size

        # Debug: Check final loss
        if torch.isnan(loss):
            logger.error(f"NaN detected in final SDS loss!")
            logger.error(f"  latents: min={latents.min()}, max={latents.max()}, has_nan={torch.isnan(latents).any()}")
            logger.error(f"  grad: min={grad.min()}, max={grad.max()}, has_nan={torch.isnan(grad).any()}")
            logger.error(f"  w: min={w.min()}, max={w.max()}")
            logger.error(f"  t: {t}")

        # Optionally return the predicted clean images
        if return_sds_images:
            with torch.no_grad():
                # Compute predicted clean latent (x0) using DDPM formula
                # x0 = (x_t - sqrt(1-alpha_t) * noise_pred) / sqrt(alpha_t)
                sqrt_alpha_t = torch.sqrt(self.alphas[t]).view(batch_size, 1, 1, 1)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas[t]).view(batch_size, 1, 1, 1)
                pred_x0 = (latents_noisy - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

                # Decode to RGB images
                sds_images = self.decode_latents(pred_x0)  # [B, 3, H, W]

            return loss, sds_images, t[0].item()  # Return timestep value
        else:
            return loss

    def _encode_imgs_mvdream(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode images to latents with MVDream VAE"""
        # imgs: [B, 3, 256, 256] in [0, 1]
        # MVDream VAE requires float32, convert if needed
        imgs = imgs.float()  # Ensure float32
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32]

    def _encode_imgs_sdxl(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode images to latents with SDXL VAE

        Note: VAE encoder is already in float32 (set during init) to avoid NaN issues.
        """
        # imgs: [B, 3, 512, 512] in [0, 1]

        # Check input validity - FAIL FAST
        if torch.isnan(imgs).any():
            raise ValueError(f"NaN detected in input images before VAE encoding! "
                           f"shape: {imgs.shape}, min: {imgs.min()}, max: {imgs.max()}")

        if torch.isinf(imgs).any():
            raise ValueError(f"Inf detected in input images before VAE encoding!")

        # Clamp to ensure values are in valid range [0, 1]
        imgs = torch.clamp(imgs, 0.0, 1.0)

        # Normalize to [-1, 1]
        imgs = 2 * imgs - 1

        # Clamp again to ensure no overflow
        imgs = torch.clamp(imgs, -1.0, 1.0)

        # Encode with float32 VAE encoder (already set during init)
        # Input must be float32 to match encoder dtype
        imgs_for_vae = imgs.float()
        posterior = self.vae.encode(imgs_for_vae).latent_dist
        latents = posterior.mean * self.vae.config.scaling_factor

        # Convert latents to expected dtype
        latents = latents.to(self.dtype)

        # Check latents - FAIL FAST
        if torch.isnan(latents).any():
            # Save debug image before raising error
            import torchvision
            debug_path = f"/tmp/debug_nan_input_step.png"
            torchvision.utils.save_image((imgs + 1) / 2, debug_path)
            raise ValueError(f"NaN detected in VAE latents! "
                           f"shape: {latents.shape}, min: {latents.min()}, max: {latents.max()}\n"
                           f"Saved input image to {debug_path} for debugging")

        if torch.isinf(latents).any():
            raise ValueError(f"Inf detected in VAE latents!")

        return latents  # [B, 4, 64, 64]

    def _get_add_time_ids_sdxl(
        self,
        original_size: Tuple[int, int],
        crops_coords_top_left: Tuple[int, int],
        target_size: Tuple[int, int],
        batch_size: int,
    ) -> torch.Tensor:
        """Get additional time IDs for SDXL"""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1).to(self.device)
        return add_time_ids

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to RGB images (following Dreamer-XL's approach)"""
        target_dtype = latents.dtype  # Save original dtype

        if self.model_type == 'mvdream':
            # MVDream VAE requires float32
            latents = latents.float()
            imgs = self.model.decode_first_stage(latents)
            imgs = ((imgs + 1) / 2).clamp(0, 1)
        elif self.model_type == 'sdxl':
            latents = latents / self.vae.config.scaling_factor
            # Convert to VAE's dtype (float32), following Dreamer-XL
            imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            # Convert back to original dtype
            imgs = imgs.to(target_dtype)

        return imgs


# ==================== Factory Function ====================

def create_sds_guidance(
    device: torch.device,
    model_type: str = 'mvdream',
    **kwargs
) -> SDSGuidance:
    """
    Factory function to create SDS guidance

    Args:
        device: torch device
        model_type: 'mvdream' or 'sdxl'
        **kwargs: Additional arguments for SDSGuidance

    Returns:
        SDSGuidance instance

    Example:
        >>> # MVDream
        >>> sds = create_sds_guidance(device, model_type='mvdream', guidance_scale=100.0)
        >>> sds.get_text_embeds("a man sitting on a chair", "ugly, blurry")
        >>> loss = sds.train_step(rendered_images, step=100, camera=camera_matrix)

        >>> # Stable Diffusion XL
        >>> sds = create_sds_guidance(device, model_type='sdxl', guidance_scale=7.5)
        >>> sds.get_text_embeds("a man sitting on a chair", "ugly, blurry")
        >>> loss = sds.train_step(rendered_images, step=100)
    """
    return SDSGuidance(device=device, model_type=model_type, **kwargs)


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print("Testing MVDream SDS")
    print("="*60)

    # Test MVDream
    sds_mvdream = create_sds_guidance(
        device=device,
        model_type='mvdream',
        guidance_scale=100.0,
        t_range=(0.02, 0.98)
    )

    sds_mvdream.get_text_embeds(
        prompt="a man sitting on a chair",
        negative_prompt="ugly, blurry, low quality"
    )

    # Test with dummy 4-view images
    dummy_images = torch.rand(4, 3, 256, 256).to(device)  # 4 views
    dummy_camera = torch.eye(4).unsqueeze(0).repeat(4, 1, 1).to(device)  # 4 cameras

    loss = sds_mvdream.train_step(dummy_images, step=100, camera=dummy_camera)
    print(f"MVDream SDS Loss: {loss.item():.4f}")

    print("\n" + "="*60)
    print("Testing Stable Diffusion XL SDS")
    print("="*60)

    # Test SDXL
    sds_sdxl = create_sds_guidance(
        device=device,
        model_type='sdxl',
        guidance_scale=7.5,
        t_range=(0.02, 0.98)
    )

    sds_sdxl.get_text_embeds(
        prompt="a man sitting on a chair",
        negative_prompt="ugly, blurry, low quality"
    )

    # Test with dummy single-view image
    dummy_image = torch.rand(1, 3, 512, 512).to(device)

    loss = sds_sdxl.train_step(dummy_image, step=100)
    print(f"SDXL SDS Loss: {loss.item():.4f}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
