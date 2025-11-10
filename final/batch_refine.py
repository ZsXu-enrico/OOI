#!/usr/bin/env python3
"""
Batch 3D Refinement using 3DEnhancer
Reads phase2_humanoid.ply from batch_recon3_output, enhances multi-view images, and refines the model
"""
import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from pathlib import Path
import json
import torch
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from glob import glob
import math

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Add 3DEnhancer to path
enhancer_path = SCRIPT_DIR.parent / "3DEnhancer" / "src"
if enhancer_path.exists():
    sys.path.insert(0, str(enhancer_path))

# Import TRELLIS components
from trellis.representations import Gaussian
from trellis.utils import render_utils
from trellis.renderers import GaussianRenderer
from easydict import EasyDict as edict

# Import 3DEnhancer
from enhancer import Enhancer
from utils.camera import get_c2ws

# Import mesh guidance for connectivity
from mesh_guidance import MeshToGaussianConverter

# Import loss computation
from final_loss import LossComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchRefinementTrainer:
    def __init__(self,
                 batch_recon_output_dir: str = None,
                 mesh_gaussian_dir: str = None,
                 output_base_dir: str = None,
                 target_instances: list = None,
                 device: str = "cuda",
                 use_sds: bool = False):

        # 设置默认路径（相对于脚本目录）
        if batch_recon_output_dir is None:
            batch_recon_output_dir = str(SCRIPT_DIR / "batch_recon3_output")
        if mesh_gaussian_dir is None:
            mesh_gaussian_dir = str(SCRIPT_DIR / "mesh_gaussian")
        if output_base_dir is None:
            output_base_dir = str(SCRIPT_DIR / "batch_refine_output")

        self.batch_recon_output_dir = Path(batch_recon_output_dir)
        self.mesh_gaussian_dir = Path(mesh_gaussian_dir)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True, parents=True)
        self.device = torch.device(device)

        # Target instances to process
        self.target_instances = target_instances or []

        # SDS mode flag
        self.use_sds = use_sds

        # 3DEnhancer model (lazy loading)
        self.enhancer_model = None
        self.enhancer_model_path = str(SCRIPT_DIR.parent / "pretrained_models" / "3DEnhancer" / "model.safetensors")
        self.enhancer_config_path = str(SCRIPT_DIR.parent / "3DEnhancer" / "src" / "configs" / "config.py")
        self.pixart_pretrained_path = str(SCRIPT_DIR.parent / "pretrained_models" / "pixart_sigma_sdxlvae_T5_diffusers")

        # Rendering configuration
        self.reference_elevation = 0.0
        self.reference_radius = 1.2
        self.reference_fov = 60.0
        self.render_resolution = 512  # 3DEnhancer requires 512x512
        self.bg_color = (1.0, 1.0, 1.0)  # White background

        # Training configuration
        if use_sds:
            # SDS mode: more steps, smaller learning rate (following batch_sds.py)
            self.learning_rate_gaussian = 1e-4
            self.weight_arap = 4.0  # Stronger ARAP in SDS mode
            self.weight_sds = 1e-4  # SDS loss weight (same as SDXL)
            self.weight_rgb = 0.5  # GT reconstruction weight
            self.weight_mask = 0.5
            self.refinement_steps = 5000  # SDS需要更多步数

            # SDS hyperparameters
            self.sds_t_range = (0.02, 0.98)  # Timestep range [20, 980] - wider range than SDXL
            self.sds_cfg_scale = 4.5  # CFG scale for 3D Enhancer
            self.sds_grad_clip = 1.0  # Gradient clipping
            self.use_noise_curriculum = True  # Curriculum learning for noise_level

            logger.info("=" * 60)
            logger.info("SDS Mode Configuration (3D Enhancer):")
            logger.info(f"  Steps: {self.refinement_steps}")
            logger.info(f"  Learning Rate: {self.learning_rate_gaussian}")
            logger.info(f"  SDS Weight: {self.weight_sds}")
            logger.info(f"  ARAP Weight: {self.weight_arap}")
            logger.info(f"  Timestep Range: {self.sds_t_range} → [{int(self.sds_t_range[0]*1000)}, {int(self.sds_t_range[1]*1000)}]")
            logger.info(f"  CFG Scale: {self.sds_cfg_scale}")
            logger.info(f"  Gradient Clip: {self.sds_grad_clip}")
            logger.info(f"  Noise Curriculum: {self.use_noise_curriculum}")
            logger.info("=" * 60)
        else:
            # Standard reconstruction mode
            self.learning_rate_gaussian = 0.0005
            self.weight_arap = 1.0
            self.weight_rgb = 1.0
            self.weight_mask = 1.0
            self.refinement_steps = 2000
            logger.info("Using reconstruction mode: 2000 steps, LR=5e-4")

        # Initialize loss computer
        self.loss_computer = LossComputer(
            device=self.device,
            bg_color=[1.0, 1.0, 1.0],
            use_scale_normalization=False
        )

    def load_enhancer_model(self):
        """Load 3DEnhancer model (lazy loading)"""
        if self.enhancer_model is None:
            logger.info("Loading 3DEnhancer model...")
            logger.info("Note: First run will download PixArt-Sigma model from HuggingFace (~5GB)")

            # Let it use the default HuggingFace cache (will auto-download)
            self.enhancer_model = Enhancer(
                model_path=self.enhancer_model_path,
                config_path=self.enhancer_config_path
            )

            logger.info("3DEnhancer model loaded successfully!")
        return self.enhancer_model

    def load_gaussian_from_ply(self, ply_path: Path) -> Gaussian:
        """Load Gaussian from PLY file"""
        gaussian = Gaussian(
            aabb=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            sh_degree=0,
            device=str(self.device)
        )
        gaussian.load_ply(str(ply_path))
        return gaussian

    def render_gaussian_view(self, gaussian: Gaussian, azimuth: float, elevation: float,
                            radius: float, resolution: int = 512, return_mask: bool = False, bg_color=None):
        """Render a single view of a Gaussian with gradient support"""
        # Get extrinsics and intrinsics
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws=float(azimuth),
            pitchs=float(elevation),
            rs=float(radius),
            fovs=self.reference_fov
        )

        # Use provided bg_color or default to white
        if bg_color is None:
            bg_color = self.bg_color

        # Setup renderer
        renderer = GaussianRenderer()
        renderer.rendering_options = edict({
            'resolution': resolution,
            'near': 0.8,
            'far': 1.6,
            'bg_color': bg_color,
            'ssaa': 1
        })
        renderer.pipe = edict({
            'kernel_size': 0.1,
            'use_mip_gaussian': True,
            'convert_SHs_python': False,
            'compute_cov3D_python': False,
            'debug': False,
            'scale_modifier': 1.0
        })

        # Render with gradient
        result = renderer.render(gaussian, extrinsics, intrinsics)

        # Extract color: [3, H, W] -> [H, W, 3]
        rendered_image = result['color'].permute(1, 2, 0)
        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

        # Extract alpha/mask if available
        if 'alpha' in result:
            rendered_alpha = result['alpha'].permute(1, 2, 0)  # [H, W, 1]
        else:
            # Compute DIFFERENTIABLE mask from color deviation
            bg_color_tensor = torch.tensor(bg_color, device=rendered_image.device).view(1, 1, 3)
            color_diff = torch.abs(rendered_image - bg_color_tensor).sum(dim=-1, keepdim=True)
            threshold = 0.01
            sharpness = 1000.0
            rendered_alpha = torch.sigmoid((color_diff - threshold) * sharpness)

        rendered_alpha = torch.clamp(rendered_alpha, 0.0, 1.0)

        if return_mask:
            return rendered_image, rendered_alpha
        else:
            return rendered_image

    def enhance_multiview_images(self, mv_images: torch.Tensor, prompt: str,
                                elevations: list = None, azimuths: list = None,
                                noise_level: int = 0, cfg_scale: float = 4.5,
                                sample_steps: int = 20):
        """
        Enhance multi-view images using 3DEnhancer

        Args:
            mv_images: Multi-view images [N, 3, H, W] in range [0, 1]
            prompt: Text prompt for the object
            elevations: List of elevation angles (default: [0, 0, 0])
            azimuths: List of azimuth angles (default: [0, 120, 240])
            noise_level: Noise level for enhancement (0-300)
            cfg_scale: CFG scale for enhancement
            sample_steps: Number of diffusion steps

        Returns:
            Enhanced images [N, 3, H, W] in range [0, 1]
        """
        enhancer = self.load_enhancer_model()

        # Default camera poses
        if elevations is None:
            elevations = [0, 0, 0]
        if azimuths is None:
            azimuths = [0, 120, 240]

        # Get camera poses
        c2ws = get_c2ws(elevations, azimuths)

        logger.info(f"Enhancing {mv_images.shape[0]} views with 3DEnhancer...")
        logger.info(f"  Prompt: {prompt}")
        logger.info(f"  Noise level: {noise_level}")
        logger.info(f"  CFG scale: {cfg_scale}")
        logger.info(f"  Sample steps: {sample_steps}")

        # Enhance images
        enhanced_images = enhancer.inference(
            mv_imgs=mv_images,
            c2ws=c2ws,
            prompt=prompt,
            fov=math.radians(self.reference_fov),
            noise_level=noise_level,
            cfg_scale=cfg_scale,
            sample_steps=sample_steps,
            color_shift=None
        )

        return enhanced_images

    def compute_sds_loss(self, rendered_images: torch.Tensor, prompt: str,
                        elevations: list, azimuths: list,
                        t_range: tuple = (0.02, 0.5), cfg_scale: float = 4.5,
                        step_ratio: float = 0.0, noise_level_curriculum: bool = True):
        """
        Compute SDS loss using 3D Enhancer's diffusion model

        SDS公式: ∇L = E_t[w(t) * (ε_θ(z_t, t, c) - ε) * ∂z_0/∂θ]

        Args:
            rendered_images: Rendered multi-view images [N, 3, H, W] in range [0, 1]
            prompt: Text prompt
            elevations: List of elevation angles
            azimuths: List of azimuth angles
            t_range: Timestep range (min, max) as fraction of 1000 steps
            cfg_scale: Classifier-free guidance scale
            step_ratio: Training progress (0.0-1.0) for curriculum learning
            noise_level_curriculum: Use curriculum learning for noise_level

        Returns:
            SDS loss scalar
        """
        enhancer = self.load_enhancer_model()
        n_views = rendered_images.shape[0]

        # === Step 1: Prepare camera embeddings ===
        c2ws = get_c2ws(elevations, azimuths)
        from utils.camera import get_camera_poses
        cur_camera_pose, epipolar_constrains, cam_distances = get_camera_poses(
            c2ws=c2ws,
            fov=math.radians(self.reference_fov),
            h=512, w=512
        )
        cur_camera_pose = cur_camera_pose.to(self.device)
        epipolar_constrains = epipolar_constrains.to(self.device)
        cam_distances = cam_distances.to(enhancer.weight_dtype).to(self.device)

        # === Step 2: Encode text prompt ===
        caption_embs, emb_masks = enhancer._encode_prompt(prompt, n_views)
        null_y = enhancer.model.y_embedder.y_embedding[None].repeat(n_views, 1, 1)[:, None]

        # === Step 3: Convert rendered images to PixArt format [-1, 1] ===
        rendered_images = F.interpolate(rendered_images, size=(512, 512), mode='bilinear', align_corners=False)
        rendered_images_pixart = 2.0 * rendered_images - 1.0  # [0,1] → [-1,1]

        # === Step 3.5: Curriculum learning - add noise to input condition ===
        if noise_level_curriculum:
            # 早期: 给rendered image加很多noise → 后期: 加少量noise
            # 这让z_lq从"模糊condition"逐渐变为"清晰condition"
            max_condition_noise = 300
            min_condition_noise = 0
            current_condition_noise_level = int(max_condition_noise * (1.0 - step_ratio))

            if current_condition_noise_level > 0:
                # 给rendered image加noise（类似inference中的做法）
                condition_noise_t = torch.full((n_views,), current_condition_noise_level - 1, dtype=torch.long).to(self.device)
                rendered_images_noisy = enhancer.noise_maker.q_sample(rendered_images_pixart, condition_noise_t)
            else:
                rendered_images_noisy = rendered_images_pixart
        else:
            rendered_images_noisy = rendered_images_pixart
            current_condition_noise_level = 0

        # === Step 4: Encode rendered images to latent space ===
        with torch.no_grad():
            latents = enhancer.vae.encode(rendered_images_pixart.to(enhancer.weight_dtype)).latent_dist.sample()
            latents = latents * 0.13025  # PixArt-Sigma scale_factor

        # Require gradient for latents (for SDS)
        latents = latents.detach().requires_grad_(True)

        # === Step 5: Encode rendered images as condition (z_lq) ===
        # Use potentially noisy version for encoding
        z_lq = enhancer.model.encode(
            rendered_images_noisy.to(enhancer.weight_dtype),
            cur_camera_pose.to(enhancer.weight_dtype),
            n_views=n_views
        )

        # === Step 6: Sample random timestep with annealing ===
        min_t, max_t = t_range
        min_step = int(min_t * 1000)  # 20
        max_step = int(max_t * 1000)  # 980
        # Timestep annealing: 980 → 490 as training progresses
        annealed_max_step = int(max_step * (1.0 - 0.5 * step_ratio))
        t = torch.randint(min_step, max(annealed_max_step, min_step + 1),
                         (n_views,), device=self.device, dtype=torch.long)

        # === Step 7: Forward diffusion - add noise to latents ===
        noise = torch.randn_like(latents)
        noisy_latents = enhancer.noise_maker.q_sample(latents, t)

        # === Step 8: Predict noise with CFG ===
        # Use the same noise_level as used for condition encoding
        noise_level = torch.full((n_views,), current_condition_noise_level, dtype=torch.long).to(self.device)

        # Prepare model kwargs
        model_kwargs_cond = dict(
            data_info={},
            c=z_lq,
            noise_level=noise_level,
            epipolar_constrains=epipolar_constrains,
            cam_distances=cam_distances,
            n_views=n_views
        )

        # Conditional noise prediction (with text prompt)
        noise_pred_cond = enhancer.model.forward_with_dpmsolver(
            noisy_latents,
            t,
            caption_embs,
            **model_kwargs_cond
        )

        # Unconditional noise prediction (no text)
        noise_pred_uncond = enhancer.model.forward_with_dpmsolver(
            noisy_latents,
            t,
            null_y,
            **model_kwargs_cond
        )

        # Classifier-free guidance
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        # === Step 9: Compute SDS gradient ===
        # Weight function: w(t) = (1 - t/T)
        w_t = (1.0 - t.float() / 1000.0).view(-1, 1, 1, 1)

        # SDS gradient: w(t) * (ε_θ - ε)
        # Detach noise to prevent backprop through random noise
        grad = w_t * (noise_pred - noise.detach())

        # === Step 10: SDS loss ===
        # Use gradient as "target direction" for latents
        # This is a trick: we want grad to flow back to rendering
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction='sum') / n_views

        return loss_sds

    def refine_instance(self, instance_name: str):
        """
        Refine a single instance using 3DEnhancer

        Args:
            instance_name: e.g., "291_humanoid_038"
        """
        logger.info("=" * 80)
        logger.info(f"REFINING INSTANCE: {instance_name}")
        logger.info("=" * 80)

        # Load phase2_humanoid.ply from batch_recon3_output
        instance_recon_dir = self.batch_recon_output_dir / instance_name
        phase2_ply_path = instance_recon_dir / "phase2_humanoid.ply"

        if not phase2_ply_path.exists():
            logger.error(f"Phase2 PLY not found: {phase2_ply_path}")
            return False

        logger.info(f"Loading phase2_humanoid.ply from {phase2_ply_path}")
        gaussian_phase2 = self.load_gaussian_from_ply(phase2_ply_path)
        logger.info(f"Loaded gaussian: {len(gaussian_phase2._xyz)} points")

        # Create output directory for this instance
        instance_output_dir = self.output_base_dir / instance_name
        instance_output_dir.mkdir(exist_ok=True, parents=True)


        # Step 1: Prepare 3-view images (225°, 345°, 105°) - 120° intervals
        # 225° uses original humanoid_segmented.png (GT image from training)
        # Other 2 angles are rendered from phase2 gaussian
        logger.info("Step 1: Preparing 3-view images (225° from GT, others rendered)...")
        render_angles = [225.0, 345.0, 105.0]  # 3 views at 120° intervals
        mv_images_list = []
        mv_masks_list = []

        render_dir = instance_output_dir / "phase2_renders"
        render_dir.mkdir(exist_ok=True, parents=True)

        for i, angle in enumerate(render_angles):
            if i == 0:  # 225° - use original GT image
                # Use humanoid_segmented.png as GT image for 225° view
                gt_image_path = self.batch_recon_output_dir.parent / "flux_edit_multiview" / instance_name / "humanoid_segmented.png"

                if gt_image_path.exists():
                    logger.info(f"✓ Using original GT image for 225°: {gt_image_path}")
                    gt_image = Image.open(gt_image_path).convert('RGB')
                    logger.info(f"  Loaded GT image size: {gt_image.size}")
                    gt_image = gt_image.resize((self.render_resolution, self.render_resolution), Image.LANCZOS)
                    gt_rgb = torch.from_numpy(np.array(gt_image)).float() / 255.0  # [H, W, 3]
                    gt_rgb = gt_rgb.to(self.device)
                    logger.info(f"  GT tensor shape: {gt_rgb.shape}, mean: {gt_rgb.mean():.3f}")

                    # Create mask from GT (non-white pixels)
                    gt_mask = (gt_rgb.sum(dim=-1, keepdim=True) < 2.9).float()  # [H, W, 1]

                    # Save for debugging
                    img_save = (gt_rgb.cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_save).save(render_dir / f"render_{int(angle):03d}_GT.png")

                    mask_save = (gt_mask.cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_save.squeeze(-1), mode='L').save(render_dir / f"mask_{int(angle):03d}_GT.png")

                    # Convert to [C, H, W] format
                    mv_images_list.append(gt_rgb.permute(2, 0, 1))  # [3, H, W]
                    mv_masks_list.append(gt_mask.permute(2, 0, 1))  # [1, H, W]
                else:
                    logger.warning(f"GT image not found at {humanoid_segmented_path}, using rendered image instead")
                    # Fall back to rendering
                    azimuth = np.radians(angle)
                    rendered_rgb, rendered_mask = self.render_gaussian_view(
                        gaussian_phase2,
                        azimuth=azimuth,
                        elevation=self.reference_elevation,
                        radius=self.reference_radius,
                        resolution=self.render_resolution,
                        return_mask=True,
                        bg_color=self.bg_color
                    )
                    img_save = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_save).save(render_dir / f"render_{int(angle):03d}.png")
                    mv_images_list.append(rendered_rgb.permute(2, 0, 1))
                    mv_masks_list.append(rendered_mask.permute(2, 0, 1))
            else:
                # Render other angles from phase2 gaussian
                azimuth = np.radians(angle)
                rendered_rgb, rendered_mask = self.render_gaussian_view(
                    gaussian_phase2,
                    azimuth=azimuth,
                    elevation=self.reference_elevation,
                    radius=self.reference_radius,
                    resolution=self.render_resolution,
                    return_mask=True,
                    bg_color=self.bg_color
                )

                # Save renders for debugging
                img_save = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img_save).save(render_dir / f"render_{int(angle):03d}.png")

                mask_save = (rendered_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(mask_save.squeeze(-1), mode='L').save(render_dir / f"mask_{int(angle):03d}.png")

                # Convert to [C, H, W] format for 3DEnhancer
                mv_images_list.append(rendered_rgb.permute(2, 0, 1))  # [3, H, W]
                mv_masks_list.append(rendered_mask.permute(2, 0, 1))  # [1, H, W]

        # Stack to [N, 3, H, W]
        mv_images = torch.stack(mv_images_list, dim=0)
        mv_masks = torch.stack(mv_masks_list, dim=0)

        logger.info(f"Prepared 3 views: {mv_images.shape} (225° from GT, 345°/105° rendered)")

        # Step 2: Load metadata to get prompt
        metadata_path = self.batch_recon_output_dir.parent / "flux_edit_multiview" / instance_name / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            humanoid_caption = metadata.get('humanoid_caption', 'a person')
        else:
            logger.warning(f"Metadata not found at {metadata_path}, using default prompt")
            humanoid_caption = 'a person'

        logger.info(f"Using prompt: {humanoid_caption}")

        # Step 2: Enhance the 3 multi-view images using 3DEnhancer (only for non-SDS mode)
        if not self.use_sds:
            logger.info("Step 2: Enhancing 3-view images with 3DEnhancer...")
            enhanced_images = self.enhance_multiview_images(
                mv_images=mv_images,
                prompt=humanoid_caption,
                elevations=[0, 0, 0],
                azimuths=[0, 120, 240],
                noise_level=300,  # Official default: 0 (minimal enhancement)
                cfg_scale=4.5,
                sample_steps=20
            )
        else:
            # SDS mode: will generate on-the-fly, skip static enhancement
            logger.info("Step 2: Skipping static enhancement (using SDS mode)")
            enhanced_images = None

        # Save enhanced images (only in reconstruction mode)
        if not self.use_sds and enhanced_images is not None:
            enhanced_dir = instance_output_dir / "enhanced_images"
            enhanced_dir.mkdir(exist_ok=True, parents=True)

            for i, (angle, enhanced_img) in enumerate(zip(render_angles, enhanced_images)):
                # enhanced_img is [3, H, W] in range [0, 1]
                img_save = (enhanced_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img_save).save(enhanced_dir / f"enhanced_{int(angle):03d}.png")

            logger.info(f"Enhanced images saved to {enhanced_dir}")

        # Step 3: Refine gaussian using enhanced images as targets
        logger.info("Step 3: Refining gaussian using enhanced images...")

        # Load connectivity for ARAP loss
        humanoid_name = metadata.get('humanoid_name', instance_name.split('_')[0] + '_' + instance_name.split('_')[1])
        connectivity_path = self.mesh_gaussian_dir / "humanoid" / humanoid_name / f"{humanoid_name}_connectivity.pkl"

        if connectivity_path.exists():
            import pickle
            with open(connectivity_path, 'rb') as f:
                self.humanoid_connectivity = pickle.load(f)
            logger.info(f"Loaded connectivity: {len(self.humanoid_connectivity)} vertices")
        else:
            logger.warning(f"Connectivity not found at {connectivity_path}, ARAP loss will use K-NN")
            self.humanoid_connectivity = None

        # Initialize trainable parameters (delta only)
        n_points = gaussian_phase2._xyz.shape[0]
        self.xyz_init = gaussian_phase2._xyz.clone().detach()
        self.delta_xyz = torch.nn.Parameter(torch.zeros_like(gaussian_phase2._xyz))
        self.delta_rotation = torch.nn.Parameter(torch.zeros_like(gaussian_phase2._rotation))

        # Store original gaussian
        self.gaussian_original = gaussian_phase2

        # Setup optimizer
        optimizer = torch.optim.Adam([
            {'params': [self.delta_xyz, self.delta_rotation], 'lr': self.learning_rate_gaussian},
        ], betas=(0.9, 0.99), eps=1e-15)

        # Convert enhanced images to targets [H, W, 3] (only for non-SDS mode)
        if not self.use_sds:
            enhanced_targets = {}
            for i, angle in enumerate(render_angles):
                enhanced_targets[angle] = enhanced_images[i].permute(1, 2, 0).to(self.device)  # [H, W, 3]

        # Store GT image for reference view reconstruction loss
        gt_image_tensor = mv_images_list[0].permute(1, 2, 0).to(self.device)  # 225° GT [H, W, 3]
        gt_mask_tensor = mv_masks_list[0].permute(1, 2, 0).to(self.device)  # [H, W, 1]

        # Training loop
        logger.info(f"Starting refinement training for {self.refinement_steps} steps...")
        logger.info(f"Mode: {'SDS' if self.use_sds else 'Reconstruction'}")

        for step in range(self.refinement_steps):
            optimizer.zero_grad()

            # Compose current gaussian
            current_gaussian = self.compose_gaussian()

            # Compute step ratio for timestep annealing
            step_ratio = step / self.refinement_steps

            # === Loss Computation ===
            total_loss = torch.tensor(0.0, device=self.device)
            loss_rgb_recon = torch.tensor(0.0, device=self.device)
            loss_sds = torch.tensor(0.0, device=self.device)

            if self.use_sds:
                # === SDS MODE ===
                # Render 3 views for SDS loss
                rendered_views = []
                for angle in render_angles:
                    azimuth = np.radians(angle)
                    rendered_rgb = self.render_gaussian_view(
                        current_gaussian,
                        azimuth=azimuth,
                        elevation=self.reference_elevation,
                        radius=self.reference_radius,
                        resolution=self.render_resolution,
                        bg_color=self.bg_color
                    )
                    # [H, W, 3] → [3, H, W]
                    rendered_views.append(rendered_rgb.permute(2, 0, 1))

                # Stack to [N, 3, H, W]
                rendered_mv = torch.stack(rendered_views, dim=0)

                # Compute SDS loss with 3D Enhancer
                loss_sds = self.compute_sds_loss(
                    rendered_images=rendered_mv,
                    prompt=humanoid_caption,
                    elevations=[0, 0, 0],
                    azimuths=[0, 120, 240],
                    t_range=self.sds_t_range,
                    cfg_scale=self.sds_cfg_scale,
                    step_ratio=step_ratio,
                    noise_level_curriculum=self.use_noise_curriculum
                )
                total_loss += self.weight_sds * loss_sds

                # Also add GT reconstruction loss for 225° view (anchor)
                ref_rendered_rgb, ref_rendered_mask = self.render_gaussian_view(
                    current_gaussian,
                    azimuth=np.radians(225.0),
                    elevation=self.reference_elevation,
                    radius=self.reference_radius,
                    resolution=self.render_resolution,
                    return_mask=True,
                    bg_color=self.bg_color
                )
                loss_rgb_recon = F.mse_loss(ref_rendered_rgb, gt_image_tensor)
                loss_mask_recon = F.mse_loss(ref_rendered_mask, gt_mask_tensor)
                total_loss += self.weight_rgb * loss_rgb_recon + self.weight_mask * loss_mask_recon

            else:
                # === RECONSTRUCTION MODE ===
                # Multi-view reconstruction loss with enhanced images as targets
                num_views = len(render_angles)

                for angle in render_angles:
                    azimuth = np.radians(angle)

                    # Render current view
                    rendered_rgb, rendered_mask = self.render_gaussian_view(
                        current_gaussian,
                        azimuth=azimuth,
                        elevation=self.reference_elevation,
                        radius=self.reference_radius,
                        resolution=self.render_resolution,
                        return_mask=True,
                        bg_color=self.bg_color
                    )

                    # Get enhanced target
                    target_rgb = enhanced_targets[angle]

                    # RGB loss
                    loss_rgb_recon += F.mse_loss(rendered_rgb, target_rgb)

                # Average over views
                loss_rgb_recon = loss_rgb_recon / num_views
                total_loss += loss_rgb_recon * self.weight_rgb

            # ARAP loss
            if self.humanoid_connectivity is not None:
                try:
                    if not hasattr(self, '_arap_connectivity_cache'):
                        # Build connectivity cache
                        logger.info("Building ARAP connectivity cache...")
                        K = 3
                        n_humanoid = self.xyz_init.shape[0]

                        ii_list, jj_list, nn_list, weight_list = [], [], [], []
                        for i, neighbors in self.humanoid_connectivity.items():
                            if i >= n_humanoid:
                                continue

                            neighbor_items = sorted(neighbors.items(), key=lambda x: x[1])[:K]
                            valid_neighbor_idx = 0
                            for j, dist in neighbor_items:
                                if j >= n_humanoid:
                                    continue

                                ii_list.append(i)
                                jj_list.append(j)
                                nn_list.append(valid_neighbor_idx)
                                weight_list.append(1.0 / (dist + 1e-6))
                                valid_neighbor_idx += 1

                        ii = torch.tensor(ii_list, dtype=torch.long).to(self.device)
                        jj = torch.tensor(jj_list, dtype=torch.long).to(self.device)
                        nn = torch.tensor(nn_list, dtype=torch.long).to(self.device)

                        weight_tensor = torch.zeros(n_humanoid, K, dtype=torch.float32).to(self.device)
                        weight_tensor[ii, nn] = torch.tensor(weight_list, dtype=torch.float32).to(self.device)
                        weight_sum = weight_tensor.sum(dim=1, keepdim=True).clamp(min=1e-6)
                        weight = weight_tensor / weight_sum

                        self._arap_connectivity_cache = (ii, jj, nn, weight, K)
                        logger.info(f"ARAP cache built: {len(ii)} edges")

                    generated_parts = [{'connectivity': self.humanoid_connectivity}]
                    loss_arap, _ = self.loss_computer.compute_arap_loss(
                        self.xyz_init,
                        self.delta_xyz,
                        self.gaussian_original.aabb,
                        generated_parts,
                        connectivity_cache=getattr(self, '_arap_connectivity_cache', None)
                    )
                    total_loss += loss_arap * self.weight_arap
                except Exception as e:
                    logger.warning(f"ARAP loss failed: {e}")
                    loss_arap = torch.tensor(0.0, device=self.device)
            else:
                loss_arap = torch.tensor(0.0, device=self.device)

            # Backward and step
            total_loss.backward()

            # Gradient clipping (following batch_sds.py)
            if self.use_sds:
                torch.nn.utils.clip_grad_norm_([self.delta_xyz, self.delta_rotation], max_norm=self.sds_grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_([self.delta_xyz, self.delta_rotation], max_norm=1.0)

            optimizer.step()

            # Parameter health check (following batch_sds.py - FAIL FAST)
            if torch.isnan(self.delta_xyz).any():
                raise ValueError(f"NaN detected in delta_xyz at step {step}")
            if torch.isnan(self.delta_rotation).any():
                raise ValueError(f"NaN detected in delta_rotation at step {step}")

            # Check if parameters are exploding
            max_delta_xyz = self.delta_xyz.abs().max().item()
            if max_delta_xyz > 10.0:
                logger.warning(f"⚠️  Large delta_xyz detected at step {step}: {max_delta_xyz:.4f}")

            max_delta_rotation = self.delta_rotation.abs().max().item()
            if max_delta_rotation > 5.0:
                logger.warning(f"⚠️  Large delta_rotation detected at step {step}: {max_delta_rotation:.4f}")

            # Log every 200 steps
            if step % 200 == 0:
                if self.use_sds:
                    # Calculate current curriculum values for logging
                    if self.use_noise_curriculum:
                        current_noise = int(300 * (1.0 - step_ratio))
                    else:
                        current_noise = 0
                    # Annealed max timestep: 980 → 490
                    current_t_max = int(980 * (1.0 - 0.5 * step_ratio))

                    log_msg = f"Step {step}/{self.refinement_steps} [{step_ratio*100:.1f}%]: "
                    log_msg += f"Loss={total_loss.item():.6f}, "
                    log_msg += f"SDS={loss_sds.item():.6f}, "
                    log_msg += f"RGB={loss_rgb_recon.item():.6f}, "
                    log_msg += f"ARAP={loss_arap.item():.6f}"
                    if self.use_noise_curriculum:
                        log_msg += f" | Noise={current_noise}, T∈[20,{current_t_max}]"
                    logger.info(log_msg)
                else:
                    logger.info(f"Step {step}/{self.refinement_steps}: Loss={total_loss.item():.6f}, "
                               f"RGB={loss_rgb_recon.item():.6f}, ARAP={loss_arap.item():.6f}")

                # Save intermediate renders
                step_dir = instance_output_dir / f"refine_step_{step:04d}"
                step_dir.mkdir(exist_ok=True, parents=True)

                for angle in render_angles:
                    azimuth = np.radians(angle)
                    rendered_rgb = self.render_gaussian_view(
                        current_gaussian,
                        azimuth=azimuth,
                        elevation=self.reference_elevation,
                        radius=self.reference_radius,
                        resolution=self.render_resolution,
                        bg_color=self.bg_color
                    )
                    img_save = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_save).save(step_dir / f"render_{int(angle):03d}.png")

        # Step 4: Save refined gaussian
        logger.info("Saving refined gaussian...")
        refined_gaussian = self.compose_gaussian()
        refined_path = instance_output_dir / "refined_humanoid.ply"
        refined_gaussian.save_ply(str(refined_path))
        logger.info(f"Saved refined gaussian to {refined_path}")

        # Render final multi-angle views
        logger.info("Rendering final multi-angle views...")
        final_render_dir = instance_output_dir / "final_renders"
        final_render_dir.mkdir(exist_ok=True, parents=True)

        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            azimuth = np.radians(angle)
            rendered_rgb = self.render_gaussian_view(
                refined_gaussian,
                azimuth=azimuth,
                elevation=0.0,
                radius=self.reference_radius,
                resolution=self.render_resolution
            )
            img_save = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_save).save(final_render_dir / f"view_{angle:03d}.png")

        logger.info(f"Final renders saved to {final_render_dir}")
        logger.info(f"Instance {instance_name} refinement complete!")

        return True

    def compose_gaussian(self) -> Gaussian:
        """Compose gaussian with trainable deltas"""
        gaussian = Gaussian(
            aabb=self.gaussian_original.aabb.clone(),
            sh_degree=0,
            device=str(self.device)
        )

        # Apply deltas
        gaussian._xyz = self.xyz_init + self.delta_xyz
        gaussian._features_dc = self.gaussian_original._features_dc.clone()
        gaussian._scaling = self.gaussian_original._scaling.clone()
        gaussian._rotation = self.gaussian_original._rotation + self.delta_rotation
        gaussian._opacity = self.gaussian_original._opacity.clone()

        return gaussian

    def run(self):
        """Run batch refinement"""
        logger.info("=" * 80)
        logger.info("BATCH 3D REFINEMENT WITH 3DENHANCER")
        logger.info("=" * 80)
        logger.info(f"Target instances: {self.target_instances}")

        for instance_name in self.target_instances:
            try:
                success = self.refine_instance(instance_name)
                if success:
                    logger.info(f"✓ {instance_name} refined successfully")
                else:
                    logger.error(f"✗ {instance_name} failed")
            except Exception as e:
                logger.error(f"✗ {instance_name} failed with exception: {e}")
                import traceback
                traceback.print_exc()

        logger.info("=" * 80)
        logger.info("BATCH REFINEMENT COMPLETE")
        logger.info("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch 3D refinement using 3DEnhancer')
    parser.add_argument('--batch_recon_output_dir', type=str,
                       default='/data4/zishuo/TRELLIS/batch_recon3_output',
                       help='Directory containing batch_recon3 output')
    parser.add_argument('--mesh_gaussian_dir', type=str,
                       default='/data4/zishuo/mesh_gaussian',
                       help='Directory containing mesh_gaussian humanoids')
    parser.add_argument('--output_dir', type=str,
                       default='/data4/zishuo/TRELLIS/batch_refine_output',
                       help='Output directory for refined results')
    parser.add_argument('--instances', nargs='+',
                       default=None,
                       help='List of instances to refine (e.g., 291_humanoid_038)')
    parser.add_argument('--all', action='store_true',
                       help='Process all instances in batch_recon3_output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--use_sds', action='store_true',
                       help='Use SDS loss instead of reconstruction loss')

    args = parser.parse_args()

    # Determine which instances to process
    if args.all:
        # Scan batch_recon3_output directory for all instances
        recon_output_path = Path(args.batch_recon_output_dir)
        all_instances = []
        for item in sorted(recon_output_path.iterdir()):
            if item.is_dir() and (item / "phase2_humanoid.ply").exists():
                all_instances.append(item.name)
        logger.info(f"Found {len(all_instances)} instances with phase2_humanoid.ply")
        logger.info(f"Instances: {all_instances}")
        target_instances = all_instances
    elif args.instances:
        target_instances = args.instances
    else:
        logger.error("Please specify --instances or --all")
        return

    trainer = BatchRefinementTrainer(
        batch_recon_output_dir=args.batch_recon_output_dir,
        mesh_gaussian_dir=args.mesh_gaussian_dir,
        output_base_dir=args.output_dir,
        target_instances=target_instances,
        device=args.device,
        use_sds=args.use_sds
    )

    trainer.run()


if __name__ == '__main__':
    main()
