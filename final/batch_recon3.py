#!/usr/bin/env python3
"""
Batch Reconstruction Training V3 (Phase 1 + Phase 2)

Phase 1: Train object2 pose using object2_segmented (1000 steps)
  - RGB loss + Mask MSE + IOU loss at 225°
  - MVAdapter multi-view mask guidance (if available):
    * Uses object masks from mvadapter_object directory
    * MVAdapter angles: 45, 90, 180, 270, 315 (0° excluded)
    * Angle mapping: TRELLIS_angle = (225 - MVAdapter_angle) % 360
    * Only mask IOU loss, weight = 1.0 (same as main mask weight)

Phase 2: Train humanoid using multi-view mask guidance (4000 steps)
  - Generates reference gaussian from humanoid_segmented.png using TRELLIS
  - Renders reference masks at 0, 90, 180, 270 degrees
  - Trains humanoid with: RGB loss (225°) + Main view mask + Multi-view mask guidance + ARAP

Object2 initial position:
- If "object_init_position" specified in metadata.json (1-4), use that quadrant
- If not specified, auto-detect from object2_mask.png:
  * Analyzes mask coverage in 4 quadrants
  * Initializes object2 in the quadrant with highest coverage
- Quadrant mapping:
  * 1: left-top (x=-0.5, z=0.5)
  * 2: right-top (x=0.5, z=0.5)
  * 3: left-bottom (x=-0.5, z=-0.5)
  * 4: right-bottom (x=0.5, z=-0.5)
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
import torchvision.transforms.functional as TF

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Add VGG-T to path (可能需要，保留绝对路径)
vggt_path = SCRIPT_DIR.parent / "vggt"
if vggt_path.exists():
    sys.path.insert(0, str(vggt_path))

# Import TRELLIS components
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian
from trellis.utils import render_utils
from trellis.renderers import GaussianRenderer
from easydict import EasyDict as edict

# Import mesh guidance for topology-aware gaussian generation
from mesh_guidance import convert_mesh_to_topology_gaussian

# Import loss computation
from final_loss import LossComputer

# Import VGG-T for depth estimation
from vggt.models.vggt import VGGT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchReconstructionTrainerV3:
    def __init__(self,
                 flux_edit_dir: str = None,
                 mesh_gaussian_dir: str = None,
                 renders_cond_dir: str = None,
                 output_base_dir: str = None,
                 target_instances: list = None,
                 device: str = "cuda"):

        # 设置默认路径（相对于脚本目录）
        if flux_edit_dir is None:
            flux_edit_dir = str(SCRIPT_DIR / "flux_edit_multiview")
        if mesh_gaussian_dir is None:
            mesh_gaussian_dir = str(SCRIPT_DIR / "mesh_gaussian")
        if renders_cond_dir is None:
            renders_cond_dir = str(SCRIPT_DIR / "renders_cond")
        if output_base_dir is None:
            output_base_dir = str(SCRIPT_DIR / "batch_recon3_output")

        self.flux_edit_dir = Path(flux_edit_dir)
        self.mesh_gaussian_dir = Path(mesh_gaussian_dir)
        self.renders_cond_dir = Path(renders_cond_dir)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True, parents=True)
        self.device = torch.device(device)

        # Target instances to process (e.g., ["291_humanoid_038", "066_character_132"])
        self.target_instances = target_instances or []

        # TRELLIS pipeline (preload in __init__ to avoid CUDA conflicts)
        logger.info("Loading TRELLIS image-to-3D pipeline...")
        self.image_pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.image_pipeline.to(self.device)
        logger.info("TRELLIS image-to-3D pipeline loaded!")

        # VGG-T model (lazy loading - loaded later to save memory)
        self.vggt_model = None

        # Training configuration
        self.learning_rate_gaussian = 0.001
        self.learning_rate_pose = 0.003
        self.weight_arap = 1.0
        self.weight_rgb = 1.0
        self.weight_mask = 1.0
        self.weight_depth = 1.0  # Weight for VGG-T depth loss
        self.bg_color = (1.0, 1.0, 1.0)  # White background

        # Phase configuration
        self.phase1_end = 1000  # End of Phase 1 (train object2 first)
        self.phase1_stage2_start = 4000  # When to increase ARAP weight (not used, ARAP weight is fixed at 1.0)
        self.phase2_end = 5000  # End of Phase 2 (train humanoid, 4000 steps)

        # Rendering configuration (changed to 518 for VGG-T)
        self.reference_azimuth = 225.0  # degrees (back-side view)
        self.reference_elevation = 0.0
        self.reference_radius = 1.2
        self.reference_fov = 60.0
        self.render_resolution = 518  # VGG-T compatible resolution

        # Initialize loss computer (will be initialized after bg_color is set)
        self.loss_computer = LossComputer(
            device=self.device,
            bg_color=[1.0, 1.0, 1.0],  # White background
            use_scale_normalization=False
        )

    def load_image_to_3d_pipeline(self):
        """Return the preloaded TRELLIS image-to-3D pipeline"""
        return self.image_pipeline

    def load_vggt_model(self):
        """Load VGG-T model (lazy loading)"""
        if self.vggt_model is None:
            logger.info("Loading VGG-T model...")
            self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
            self.vggt_model.eval()
            # Freeze VGG-T parameters - we only use it for depth prediction
            self.vggt_model.requires_grad_(False)
            logger.info("VGG-T model loaded successfully!")
        return self.vggt_model

    def predict_depth_vggt(self, image_rgb: torch.Tensor, enable_grad: bool = False):
        """
        Predict depth using VGG-T

        Args:
            image_rgb: Input image [H, W, 3] in range [0, 1]
            enable_grad: If True, compute with gradients; if False, use no_grad

        Returns:
            depth: Predicted depth map [H, W]
        """
        model = self.load_vggt_model()

        # Convert to VGG-T format: [1, 3, H, W]
        if image_rgb.dim() == 3:
            image_vggt = image_rgb.permute(2, 0, 1).unsqueeze(0)  # [H, W, 3] -> [1, 3, H, W]
        else:
            image_vggt = image_rgb

        # Ensure image is in [0, 1] range
        image_vggt = torch.clamp(image_vggt, 0.0, 1.0)

        # Predict depth
        if enable_grad:
            # With gradient for training
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = model(image_vggt)
        else:
            # Without gradient for reference
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = model(image_vggt)

        # Extract depth
        depth = predictions["depth"]  # [B, H, W, 1] or similar

        # Squeeze to [H, W]
        while depth.dim() > 2:
            if depth.shape[0] == 1:
                depth = depth.squeeze(0)
            elif depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            else:
                break

        return depth.float()  # Convert back to float32

    def generate_humanoid_gaussian_with_mesh_guidance(self, sha256: str, humanoid_name: str, caption: str = ""):
        """
        Generate humanoid gaussian with mesh guidance from renders_cond image
        Saves to mesh_gaussian/humanoid/{humanoid_name}/
        If PLY exists but mesh/connectivity missing, supplements them.

        Args:
            sha256: SHA256 hash for the humanoid in renders_cond
            humanoid_name: Name like "066_character" or "291_humanoid"
            caption: Optional caption for the humanoid

        Returns:
            Path to the generated gaussian ply file
        """
        logger.info(f"=== Generating Humanoid Gaussian with Mesh Guidance ===")
        logger.info(f"  Humanoid: {humanoid_name}")
        logger.info(f"  SHA256: {sha256}")

        # Output directory
        output_dir = self.mesh_gaussian_dir / "human_gaussian" / humanoid_name
        output_dir.mkdir(exist_ok=True, parents=True)

        # Check what files exist
        ply_path = output_dir / f"{humanoid_name}_topology_gaussian.ply"
        mesh_path = output_dir / f"{humanoid_name}_mesh.glb"
        connectivity_path = output_dir / f"{humanoid_name}_connectivity.pkl"

        # If PLY exists, check if we need to supplement mesh/connectivity
        if ply_path.exists():
            logger.info(f"  Gaussian PLY already exists at {ply_path}")
            if mesh_path.exists() and connectivity_path.exists():
                logger.info(f"  Mesh and connectivity already exist, COMPLETE!")
                return ply_path
            else:
                # Need to supplement missing files
                missing = []
                if not mesh_path.exists():
                    missing.append("mesh")
                if not connectivity_path.exists():
                    missing.append("connectivity")
                logger.info(f"  SUPPLEMENTING missing files: {', '.join(missing)}")

                # Load image from renders_cond
                image_path = self.renders_cond_dir / sha256 / "000.png"
                if not image_path.exists():
                    logger.error(f"  Image not found: {image_path}")
                    return ply_path  # Return existing PLY even if supplement fails

                logger.info(f"  Loading image from {image_path}")
                image = Image.open(image_path).convert('RGB')

                # Generate 3D to get mesh
                pipeline = self.load_image_to_3d_pipeline()
                logger.info(f"  Generating mesh from image...")
                outputs = pipeline.run(
                    image=image,
                    seed=42,
                    formats=["gaussian", "mesh"],
                    sparse_structure_sampler_params={
                        "steps": 25,
                        "cfg_strength": 7.5,
                    },
                    slat_sampler_params={
                        "steps": 25,
                        "cfg_strength": 7.5,
                    },
                )

                mesh_trellis = outputs['mesh'][0]
                gaussian_trellis = outputs['gaussian'][0]
                logger.info(f"  Generated: {len(mesh_trellis.vertices)} vertices, {len(gaussian_trellis._xyz)} gaussians")

                # Save mesh if missing
                if not mesh_path.exists():
                    import trimesh
                    mesh_to_save = trimesh.Trimesh(
                        vertices=mesh_trellis.vertices.cpu().numpy(),
                        faces=mesh_trellis.faces.cpu().numpy()
                    )
                    mesh_to_save.export(str(mesh_path))
                    logger.info(f"  Saved mesh to {mesh_path}")

                # Extract and save connectivity if missing
                if not connectivity_path.exists():
                    from mesh_guidance import MeshToGaussianConverter
                    logger.info(f"  Extracting connectivity from mesh topology...")
                    converter = MeshToGaussianConverter(device=str(self.device))
                    connectivity = converter.extract_mesh_connectivity(
                        vertices=mesh_trellis.vertices.to(self.device),
                        faces=mesh_trellis.faces.to(self.device)
                    )
                    num_edges = sum(len(v) for v in connectivity.values()) // 2
                    logger.info(f"  Connectivity extracted: {len(connectivity)} vertices, {num_edges} edges")

                    import pickle
                    with open(connectivity_path, 'wb') as f:
                        pickle.dump(connectivity, f)
                    logger.info(f"  Saved connectivity to {connectivity_path}")

                logger.info(f"  ✓ Supplemented missing files for {humanoid_name}")
                return ply_path

        # Load image from renders_cond
        image_path = self.renders_cond_dir / sha256 / "000.png"
        if not image_path.exists():
            logger.error(f"  Image not found: {image_path}")
            return None

        logger.info(f"  Loading image from {image_path}")
        image = Image.open(image_path).convert('RGB')

        # Generate 3D using TRELLIS image-to-3D
        pipeline = self.load_image_to_3d_pipeline()
        logger.info(f"  Generating 3D from image...")
        outputs = pipeline.run(
            image=image,
            seed=42,
            formats=["gaussian", "mesh"],
            sparse_structure_sampler_params={
                "steps": 25,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 25,
                "cfg_strength": 7.5,
            },
        )

        gaussian_trellis = outputs['gaussian'][0]
        mesh_trellis = outputs['mesh'][0]
        logger.info(f"  Generated: {len(gaussian_trellis._xyz)} Gaussians, {len(mesh_trellis.vertices)} vertices")

        # Apply mesh guidance to create topology-aware gaussian
        logger.info(f"  Converting mesh to topology-aware Gaussian (64 views, 10k steps)...")
        video_path = output_dir / f"{humanoid_name}_topology_gaussian.mp4"

        gaussian_final, connectivity = convert_mesh_to_topology_gaussian(
            mesh_trellis=mesh_trellis,
            gaussian_trellis=gaussian_trellis,
            device=str(self.device),
            optimize=True,
            num_steps=10000,
            num_views=64,
            resolution=512,
            scale_factor=1.0/1.1,
            render_video=True,
            video_output_path=str(video_path),
            video_num_frames=120,
            video_resolution=512
        )

        logger.info(f"  Final Gaussian: {len(gaussian_final._xyz)} points with topology")
        logger.info(f"  Topology edges: {sum(len(v) for v in connectivity.values()) // 2}")

        # Save gaussian
        gaussian_final.save_ply(str(ply_path))
        logger.info(f"  Saved gaussian to {ply_path}")

        # Save mesh
        import trimesh
        mesh_to_save = trimesh.Trimesh(
            vertices=mesh_trellis.vertices.cpu().numpy(),
            faces=mesh_trellis.faces.cpu().numpy()
        )
        mesh_to_save.export(str(mesh_path))
        logger.info(f"  Saved mesh to {mesh_path}")

        # Save connectivity
        import pickle
        connectivity_path = output_dir / f"{humanoid_name}_connectivity.pkl"
        with open(connectivity_path, 'wb') as f:
            pickle.dump(connectivity, f)
        logger.info(f"  Saved connectivity to {connectivity_path}")

        # Save metadata
        info = {
            'sha256': sha256,
            'caption': caption,
            'safe_caption': humanoid_name,
            'num_points': len(gaussian_final._xyz),
            'num_topology_edges': sum(len(v) for v in connectivity.values()) // 2,
            'num_views': 64,
            'num_steps': 10000,
            'scale_factor': 1.0/1.1,
            'use_mesh_guidance': True,
            'image_path': str(image_path)
        }

        info_path = output_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"  Saved info to {info_path}")

        # Save caption
        caption_path = output_dir / "caption.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption)

        return ply_path

    def load_gaussian_from_ply(self, ply_path: Path) -> Gaussian:
        """Load Gaussian from PLY file"""
        gaussian = Gaussian(
            aabb=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            sh_degree=0,
            device=str(self.device)
        )
        gaussian.load_ply(str(ply_path))
        return gaussian

    def generate_object2_gaussian_from_image(self, object_text: str) -> Gaussian:
        """
        Generate object2 gaussian from object_data image using TRELLIS

        Args:
            object_text: Object description (e.g., "a violin")

        Returns:
            Gaussian object
        """
        logger.info(f"=== Generating Object2 Gaussian from Image ===")
        logger.info(f"  Object text: {object_text}")

        # Extract object name from text (e.g., "a violin" -> "violin")
        # Remove common prefixes like "a ", "an ", "the "
        object_name = object_text.lower().strip()
        for prefix in ["a ", "an ", "the "]:
            if object_name.startswith(prefix):
                object_name = object_name[len(prefix):]
                break

        logger.info(f"  Object name: {object_name}")

        # Try to find image in multiple directories with priority:
        # 1. new/
        # 2. animal_object/object/
        # 3. object_data/
        search_dirs = [
            Path("/data4/zishuo/TRELLIS/new"),
            Path("/data4/zishuo/TRELLIS/animal_object/object"),
            Path("/data4/zishuo/TRELLIS/object_data")
        ]

        image_path = None
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Try exact match
            candidate = search_dir / f"{object_name}.png"
            if candidate.exists():
                image_path = candidate
                logger.info(f"  Found image in {search_dir.name}: {candidate.name}")
                break

            # Try with underscores instead of spaces
            object_name_underscore = object_name.replace(" ", "_")
            candidate = search_dir / f"{object_name_underscore}.png"
            if candidate.exists():
                image_path = candidate
                logger.info(f"  Found image in {search_dir.name}: {candidate.name}")
                break

        if image_path is None:
            logger.error(f"  Image not found for object '{object_text}' in any directory")
            raise FileNotFoundError(f"Image not found for object '{object_text}'")

        logger.info(f"  Loading image from {image_path}")
        image = Image.open(image_path).convert('RGB')

        # Generate 3D using TRELLIS image-to-3D
        pipeline = self.load_image_to_3d_pipeline()
        logger.info(f"  Generating 3D from image...")
        outputs = pipeline.run(
            image=image,
            seed=42,
            formats=["gaussian"],
            sparse_structure_sampler_params={
                "steps": 25,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 25,
                "cfg_strength": 7.5,
            },
        )

        gaussian_trellis = outputs['gaussian'][0]
        logger.info(f"  Generated: {len(gaussian_trellis._xyz)} Gaussians")

        return gaussian_trellis

    def align_reference_to_target(self, ref_rgb: torch.Tensor, ref_mask: torch.Tensor,
                                  target_rgb: torch.Tensor, target_mask: torch.Tensor,
                                  angle: float = 0.0):
        """
        Align reference render to target render (like align_mvadapter_to_humanoid_render_bbox)

        Steps:
        1. Detect bbox from target render
        2. Detect bbox from reference render and crop
        3. Resize reference to match target bbox height
        4. Place on canvas at target bbox position with angle-specific horizontal alignment

        Args:
            ref_rgb: Reference RGB image [H, W, 3], white background
            ref_mask: Reference mask [H, W, 1]
            target_rgb: Target RGB image [H, W, 3], white background
            target_mask: Target mask [H, W, 1]
            angle: View angle in degrees (0, 90, 180, 270) for determining horizontal alignment

        Returns:
            aligned_rgb: [H, W, 3] same size as target
            aligned_mask: [H, W, 1] same size as target
        """
        device = ref_rgb.device
        white_value = 1.0

        # 1. Detect bbox from target image
        target_diff = torch.abs(target_rgb - white_value).mean(dim=-1)  # [H, W]
        target_fg = (target_diff > 0.05).float()

        y_sum = target_fg.sum(dim=1)
        x_sum = target_fg.sum(dim=0)

        y_nonzero = torch.where(y_sum > 0)[0]
        x_nonzero = torch.where(x_sum > 0)[0]

        if len(y_nonzero) > 0 and len(x_nonzero) > 0:
            target_y0 = max(y_nonzero[0].item() - 1, 0)
            target_y1 = min(y_nonzero[-1].item() + 2, target_rgb.shape[0])
            target_x0 = max(x_nonzero[0].item() - 1, 0)
            target_x1 = min(x_nonzero[-1].item() + 2, target_rgb.shape[1])
            target_height = target_y1 - target_y0
            target_width = target_x1 - target_x0
        else:
            # No foreground in target, use full image
            target_y0, target_y1 = 0, target_rgb.shape[0]
            target_x0, target_x1 = 0, target_rgb.shape[1]
            target_height = target_rgb.shape[0]
            target_width = target_rgb.shape[1]

        # 2. Detect bbox from reference image and crop
        ref_diff = torch.abs(ref_rgb - white_value).mean(dim=-1)
        ref_fg = (ref_diff > 0.05).float()

        y_sum = ref_fg.sum(dim=1)
        x_sum = ref_fg.sum(dim=0)

        y_nonzero = torch.where(y_sum > 0)[0]
        x_nonzero = torch.where(x_sum > 0)[0]

        if len(y_nonzero) > 0 and len(x_nonzero) > 0:
            ref_y0 = max(y_nonzero[0].item() - 1, 0)
            ref_y1 = min(y_nonzero[-1].item() + 2, ref_rgb.shape[0])
            ref_x0 = max(x_nonzero[0].item() - 1, 0)
            ref_x1 = min(x_nonzero[-1].item() + 2, ref_rgb.shape[1])

            # Crop reference to its bbox
            ref_crop = ref_rgb[ref_y0:ref_y1, ref_x0:ref_x1, :]
            mask_crop = ref_mask[ref_y0:ref_y1, ref_x0:ref_x1, :]
            H_crop, W_crop = ref_crop.shape[:2]
        else:
            # No foreground in reference
            ref_crop = ref_rgb
            mask_crop = ref_mask
            H_crop, W_crop = ref_crop.shape[:2]

        # 3. Resize reference to target bbox HEIGHT (maintain aspect ratio)
        H_new = target_height
        W_new = int(W_crop * H_new / H_crop)

        ref_resized = F.interpolate(
            ref_crop.permute(2, 0, 1).unsqueeze(0),
            size=(H_new, W_new),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        mask_resized = F.interpolate(
            mask_crop.permute(2, 0, 1).unsqueeze(0),
            size=(H_new, W_new),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # 4. Create canvas same size as target with white background
        canvas_rgb = torch.ones_like(target_rgb)
        canvas_mask = torch.zeros_like(target_mask)

        # Place reference aligned to target bbox position
        # Vertical: bottom-align to target bbox bottom
        start_h = target_y1 - H_new  # Bottom align to target bbox bottom

        # Horizontal: align to target bbox with angle-specific strategy
        #   - 90°: right-align to target bbox right edge (person facing right)
        #   - 270°: left-align to target bbox left edge (person facing left)
        #   - 0°, 180°: center-align to target bbox center
        if abs(angle - 90.0) < 1.0:  # 90 degrees: right-align to target bbox
            start_w = target_x1 - W_new
        elif abs(angle - 270.0) < 1.0:  # 270 degrees: left-align to target bbox
            start_w = target_x0
        else:  # 0 or 180 degrees: center-align to target bbox
            target_bbox_center_x = (target_x0 + target_x1) // 2
            start_w = target_bbox_center_x - W_new // 2

        # Ensure bounds
        start_w = max(0, min(start_w, target_rgb.shape[1] - 1))
        start_h = max(0, min(start_h, target_rgb.shape[0] - 1))
        end_h = min(start_h + H_new, target_rgb.shape[0])
        end_w = min(start_w + W_new, target_rgb.shape[1])
        actual_h = end_h - start_h
        actual_w = end_w - start_w

        canvas_rgb[start_h:end_h, start_w:end_w] = ref_resized[:actual_h, :actual_w]
        canvas_mask[start_h:end_h, start_w:end_w] = mask_resized[:actual_h, :actual_w]

        return canvas_rgb, canvas_mask

    def align_to_bbox(self, rendered_rgb: torch.Tensor, rendered_mask: torch.Tensor, bbox_info: dict):
        """
        Align rendered image to target bbox (similar to batch_recon.py's align_mvadapter_to_humanoid_render_bbox)

        Steps:
        1. Detect bbox from rendered image
        2. Crop to bbox
        3. Resize to match target bbox height
        4. Place on canvas at target bbox position

        Args:
            rendered_rgb: Rendered RGB image [H, W, 3], white background
            rendered_mask: Rendered mask [H, W, 1]
            bbox_info: Dict with keys: y0, y1, x0, x1, height, width, image_height, image_width

        Returns:
            aligned_rgb: [image_height, image_width, 3]
            aligned_mask: [image_height, image_width, 1]
        """
        device = rendered_rgb.device
        H_orig, W_orig = rendered_rgb.shape[:2]
        white_value = 1.0

        # Target canvas size from bbox_info
        target_height = bbox_info['image_height']
        target_width = bbox_info['image_width']
        target_bbox_y1 = bbox_info['y1']
        target_bbox_height = bbox_info['height']

        # 1. Detect foreground in rendered image (non-white pixels)
        white_diff = torch.abs(rendered_rgb - white_value).mean(dim=-1)  # [H, W]
        foreground_mask = (white_diff > 0.05).float()  # [H, W]

        # Get bounding box of foreground
        y_sum = foreground_mask.sum(dim=1)  # [H]
        x_sum = foreground_mask.sum(dim=0)  # [W]

        y_nonzero = torch.where(y_sum > 0)[0]
        x_nonzero = torch.where(x_sum > 0)[0]

        if len(y_nonzero) == 0 or len(x_nonzero) == 0:
            # No foreground detected, use entire image
            img_crop = rendered_rgb
            mask_crop = rendered_mask
            H_crop, W_crop = H_orig, W_orig
        else:
            # Get bounding box with 1-pixel padding
            y0 = max(y_nonzero[0].item() - 1, 0)
            y1 = min(y_nonzero[-1].item() + 2, H_orig)
            x0 = max(x_nonzero[0].item() - 1, 0)
            x1 = min(x_nonzero[-1].item() + 2, W_orig)

            # Crop to bounding box
            img_crop = rendered_rgb[y0:y1, x0:x1, :]  # [H_crop, W_crop, 3]
            mask_crop = rendered_mask[y0:y1, x0:x1, :]
            H_crop, W_crop = img_crop.shape[:2]

        # 2. Resize to target bbox HEIGHT (keep aspect ratio)
        H_new = target_bbox_height
        W_new = int(W_crop * H_new / H_crop)

        # Resize image and mask
        img_resized = F.interpolate(
            img_crop.permute(2, 0, 1).unsqueeze(0),  # [1, 3, H_crop, W_crop]
            size=(H_new, W_new),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # [H_new, W_new, 3]

        mask_resized = F.interpolate(
            mask_crop.permute(2, 0, 1).unsqueeze(0),
            size=(H_new, W_new),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # 3. Create target canvas with white background
        canvas_rgb = torch.ones(target_height, target_width, 3, device=device)
        canvas_mask = torch.zeros(target_height, target_width, 1, device=device)

        # 4. Place resized object:
        # - Vertical: Bottom-align with target bbox bottom (feet alignment)
        # - Horizontal: Center on entire canvas
        start_h = target_bbox_y1 - H_new  # Bottom-align to bbox bottom
        start_w = (target_width - W_new) // 2  # Horizontal center on full canvas

        # Paste resized image onto canvas
        end_h = min(start_h + H_new, target_height)
        end_w = min(start_w + W_new, target_width)
        actual_h = end_h - start_h
        actual_w = end_w - start_w

        canvas_rgb[start_h:end_h, start_w:end_w] = img_resized[:actual_h, :actual_w]
        canvas_mask[start_h:end_h, start_w:end_w] = mask_resized[:actual_h, :actual_w]

        return canvas_rgb, canvas_mask

    def render_gaussian_view(self, gaussian: Gaussian, azimuth: float, elevation: float,
                            radius: float, resolution: int = 518, return_mask: bool = False, bg_color=None):
        """Render a single view of a Gaussian with gradient support

        Args:
            gaussian: Gaussian object to render
            azimuth: Camera azimuth angle (radians)
            elevation: Camera elevation angle (radians)
            radius: Camera radius
            resolution: Rendering resolution (default 512)
            return_mask: If True, return (rgb, mask) tuple; otherwise return rgb only
            bg_color: Optional background color (R, G, B) in [0, 1]. If None, use default white (1.0, 1.0, 1.0)

        Returns:
            If return_mask=False: rendered_image [H, W, 3]
            If return_mask=True: (rendered_image [H, W, 3], rendered_mask [H, W, 1])
        """
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

        # Ensure range [0, 1]
        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

        # Extract alpha/mask if available
        if 'alpha' in result:
            rendered_alpha = result['alpha'].permute(1, 2, 0)  # [H, W, 1]
        else:
            # Compute DIFFERENTIABLE mask from color deviation using sigmoid
            bg_color_tensor = torch.tensor(bg_color, device=rendered_image.device).view(1, 1, 3)
            color_diff = torch.abs(rendered_image - bg_color_tensor).sum(dim=-1, keepdim=True)
            # Use sigmoid to create soft mask (differentiable)
            threshold = 0.01
            sharpness = 1000.0
            rendered_alpha = torch.sigmoid((color_diff - threshold) * sharpness)

        rendered_alpha = torch.clamp(rendered_alpha, 0.0, 1.0)

        if return_mask:
            return rendered_image, rendered_alpha
        else:
            return rendered_image

    def train_phase1_phase2(self, instance_name: str):
        """
        Train Phase 1 + Phase 2 for a single instance

        Args:
            instance_name: e.g., "291_humanoid_038" or "066_character_132"
        """
        logger.info("=" * 80)
        logger.info(f"TRAINING INSTANCE: {instance_name}")
        logger.info("=" * 80)

        # Load metadata
        metadata_path = self.flux_edit_dir / instance_name / "metadata.json"
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {metadata_path}")
            return False

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        humanoid_name = metadata['humanoid_name']
        object_id = metadata['object_id']
        object_text = metadata['object_text']
        object_init_position = metadata.get('object_init_position', None)  # Optional: 1, 2, 3, 4 for corners

        logger.info(f"Humanoid: {humanoid_name}")
        logger.info(f"Object: {object_text} (ID: {object_id})")
        if object_init_position is not None:
            logger.info(f"Object initial position: {object_init_position}")

        # Create output directory for this instance
        instance_output_dir = self.output_base_dir / instance_name
        instance_output_dir.mkdir(exist_ok=True, parents=True)

        # Step 1: Load or generate humanoid gaussian
        # First check if metadata has humanoid_ply field (for pre-generated humanoids)
        if 'humanoid_ply' in metadata and metadata['humanoid_ply']:
            humanoid_gaussian_path = Path(metadata['humanoid_ply'])
            logger.info(f"Using humanoid gaussian path from metadata: {humanoid_gaussian_path}")
        else:
            # Default path based on humanoid_name
            humanoid_gaussian_path = self.mesh_gaussian_dir / "human_gaussian" / humanoid_name / f"{humanoid_name}_topology_gaussian.ply"

        if not humanoid_gaussian_path.exists():
            logger.info(f"Humanoid gaussian not found at {humanoid_gaussian_path}, attempting to generate...")

            # Extract humanoid ID from name (for numbered humanoids like "291_humanoid")
            import re
            match = re.match(r'(\d+)_', humanoid_name)
            if match:
                humanoid_id = int(match.group(1))
                logger.info(f"Extracted humanoid ID: {humanoid_id}")

                # Read filtered instances to get sha256
                filtered_instances_file = Path("/data4/zishuo/TRELLIS/filtered_instances_rendered.txt")
                if filtered_instances_file.exists():
                    with open(filtered_instances_file, 'r') as f:
                        filtered_instances = [line.strip() for line in f if line.strip()]

                    if humanoid_id <= len(filtered_instances):
                        humanoid_sha256 = filtered_instances[humanoid_id - 1]
                        logger.info(f"Found humanoid sha256: {humanoid_sha256[:16]}...")

                        # Generate gaussian with mesh guidance
                        humanoid_gaussian_path = self.generate_humanoid_gaussian_with_mesh_guidance(
                            sha256=humanoid_sha256,
                            humanoid_name=humanoid_name,
                            caption=metadata.get('humanoid_caption', '')
                        )

                        if humanoid_gaussian_path is None:
                            logger.error(f"Failed to generate humanoid gaussian")
                            return False
                    else:
                        logger.error(f"Humanoid ID {humanoid_id} out of range")
                        return False
                else:
                    logger.error(f"Filtered instances file not found: {filtered_instances_file}")
                    return False
            else:
                # Non-numeric humanoid name - try to generate from animal_data or human_data
                logger.info(f"Non-numeric humanoid name: {humanoid_name}")
                logger.info(f"Attempting to generate from animal_data or human_data directory...")

                # Try animal_data first, then human_data (relative to SCRIPT_DIR)
                search_dirs = [
                    SCRIPT_DIR / "animal_data",
                    SCRIPT_DIR / "human_data"
                ]

                image_path = None
                for search_dir in search_dirs:
                    if search_dir.exists():
                        candidate = search_dir / f"{humanoid_name}.png"
                        if candidate.exists():
                            image_path = candidate
                            logger.info(f"Found image in {search_dir.name}: {candidate.name}")
                            break

                if not image_path:
                    logger.error(f"Could not extract humanoid ID from name: {humanoid_name}")
                    logger.error(f"Image not found in animal_data or human_data directories")
                    logger.error(f"Expected path: {humanoid_gaussian_path}")
                    return False

                logger.info(f"Generating gaussian from image...")

                try:
                    # Generate using the same logic as batch_image_gaussian.py
                    image = Image.open(image_path).convert('RGB')

                    # Generate 3D using TRELLIS
                    pipeline = self.load_image_to_3d_pipeline()
                    logger.info(f"  Generating 3D from image...")
                    outputs = pipeline.run(
                        image=image,
                        seed=42,
                        formats=["gaussian", "mesh"],
                        sparse_structure_sampler_params={
                            "steps": 25,
                            "cfg_strength": 7.5,
                        },
                        slat_sampler_params={
                            "steps": 25,
                            "cfg_strength": 7.5,
                        },
                    )

                    mesh_trellis = outputs['mesh'][0]
                    gaussian_trellis = outputs['gaussian'][0]
                    logger.info(f"  Generated: {len(mesh_trellis.vertices)} vertices, {len(gaussian_trellis._xyz)} gaussians")

                    # Apply mesh guidance
                    logger.info(f"  Converting mesh to topology-aware Gaussian (64 views, 10k steps)...")
                    output_dir = humanoid_gaussian_path.parent
                    output_dir.mkdir(exist_ok=True, parents=True)
                    video_path = output_dir / f"{humanoid_name}_topology_gaussian.mp4"

                    from mesh_guidance import convert_mesh_to_topology_gaussian
                    gaussian_final, connectivity = convert_mesh_to_topology_gaussian(
                        mesh_trellis=mesh_trellis,
                        gaussian_trellis=gaussian_trellis,
                        device=str(self.device),
                        optimize=True,
                        num_steps=10000,
                        num_views=64,
                        resolution=512,
                        scale_factor=1.0/1.1,
                        render_video=True,
                        video_output_path=str(video_path),
                        video_num_frames=120,
                        video_resolution=512
                    )

                    logger.info(f"  Final Gaussian: {len(gaussian_final._xyz)} points with topology")
                    logger.info(f"  Topology edges: {sum(len(v) for v in connectivity.values()) // 2}")

                    # Save files
                    gaussian_final.save_ply(str(humanoid_gaussian_path))
                    logger.info(f"  Saved gaussian to {humanoid_gaussian_path}")

                    import trimesh
                    mesh_path = output_dir / f"{humanoid_name}_mesh.glb"
                    mesh_to_save = trimesh.Trimesh(
                        vertices=mesh_trellis.vertices.cpu().numpy(),
                        faces=mesh_trellis.faces.cpu().numpy()
                    )
                    mesh_to_save.export(str(mesh_path))
                    logger.info(f"  Saved mesh to {mesh_path}")

                    import pickle
                    connectivity_path = output_dir / f"{humanoid_name}_connectivity.pkl"
                    with open(connectivity_path, 'wb') as f:
                        pickle.dump(connectivity, f)
                    logger.info(f"  Saved connectivity to {connectivity_path}")

                    logger.info(f"  ✓ Successfully generated {humanoid_name}")

                except Exception as e:
                    logger.error(f"Failed to generate from image: {e}")
                    import traceback
                    traceback.print_exc()
                    return False

        # Load humanoid gaussian
        logger.info(f"Loading humanoid gaussian from {humanoid_gaussian_path}")
        humanoid_gaussian = self.load_gaussian_from_ply(humanoid_gaussian_path)
        logger.info(f"Loaded humanoid gaussian: {len(humanoid_gaussian._xyz)} points")

        # Load connectivity for ARAP loss
        # Determine the subdirectory (animal_data or human_gaussian) from the gaussian path
        gaussian_subdir = humanoid_gaussian_path.parent.parent.name
        connectivity_path = self.mesh_gaussian_dir / gaussian_subdir / humanoid_name / f"{humanoid_name}_connectivity.pkl"

        # Helper function to regenerate gaussian with correct connectivity
        def regenerate_humanoid_gaussian():
            """Regenerate humanoid gaussian with mesh guidance - delete old files and retrain"""
            nonlocal humanoid_gaussian  # Declare at the start of the function

            logger.warning(f"Regenerating humanoid gaussian with mesh guidance...")

            # Delete old incompatible files first
            logger.warning(f"Deleting old incompatible files to force regeneration...")
            # Use the correct subdirectory (animal_data or human_gaussian) from gaussian_subdir
            mesh_path = humanoid_gaussian_path.parent / f"{humanoid_name}_mesh.glb"
            video_path = humanoid_gaussian_path.parent / f"{humanoid_name}_topology_gaussian.mp4"
            for file_path in [humanoid_gaussian_path, mesh_path, connectivity_path, video_path]:
                if file_path.exists():
                    logger.info(f"  Deleting: {file_path.name}")
                    file_path.unlink()

            import re
            match = re.match(r'(\d+)_', humanoid_name)
            if match:
                # Numeric humanoid - can auto-regenerate
                humanoid_id = int(match.group(1))
                filtered_instances_file = Path("/data4/zishuo/TRELLIS/filtered_instances_rendered.txt")

                if filtered_instances_file.exists():
                    with open(filtered_instances_file, 'r') as f:
                        filtered_instances = [line.strip() for line in f if line.strip()]

                    if humanoid_id <= len(filtered_instances):
                        humanoid_sha256 = filtered_instances[humanoid_id - 1]
                        logger.info(f"Found humanoid sha256: {humanoid_sha256[:16]}...")

                        # Regenerate with mesh guidance
                        new_ply_path = self.generate_humanoid_gaussian_with_mesh_guidance(
                            sha256=humanoid_sha256,
                            humanoid_name=humanoid_name,
                            caption=metadata.get('humanoid_caption', '')
                        )

                        # Reload gaussian and connectivity
                        if new_ply_path and new_ply_path.exists():
                            humanoid_gaussian = self.load_gaussian_from_ply(new_ply_path)
                            logger.info(f"Reloaded humanoid gaussian: {len(humanoid_gaussian._xyz)} points")
                        else:
                            logger.error(f"Failed to regenerate gaussian PLY")
                            return False

                        if connectivity_path.exists():
                            import pickle
                            with open(connectivity_path, 'rb') as f:
                                self.humanoid_connectivity = pickle.load(f)

                            # Verify match after regeneration
                            max_idx = max(self.humanoid_connectivity.keys())
                            num_pts = len(humanoid_gaussian._xyz)
                            if max_idx >= num_pts:
                                logger.error(f"Regeneration failed! Still mismatched: max_idx={max_idx}, num_pts={num_pts}")
                                return False

                            logger.info(f"Reloaded connectivity: {len(self.humanoid_connectivity)} vertices")
                            logger.info(f"Verification passed: max_idx={max_idx} < num_pts={num_pts}")
                            return True
                        else:
                            logger.error(f"Failed to regenerate connectivity")
                            return False
                    else:
                        logger.error(f"Humanoid ID {humanoid_id} out of range")
                        return False
                else:
                    logger.error(f"Filtered instances file not found")
                    return False
            else:
                # Non-numeric humanoid (e.g., "zombie samurai", "207_character", "cat_1")
                # Try to find image in animal_data or human_data directory and regenerate
                logger.warning(f"Non-numeric humanoid name: {humanoid_name}")
                logger.warning(f"Attempting to regenerate from animal_data or human_data directory...")

                # Look for image in animal_data first, then human_data (relative to SCRIPT_DIR)
                search_dirs = [
                    SCRIPT_DIR / "animal_data",
                    SCRIPT_DIR / "human_data"
                ]

                image_path = None
                for search_dir in search_dirs:
                    if search_dir.exists():
                        candidate = search_dir / f"{humanoid_name}.png"
                        if candidate.exists():
                            image_path = candidate
                            logger.info(f"Found image in {search_dir.name}: {candidate.name}")
                            break

                if not image_path:
                    logger.error(f"=" * 80)
                    logger.error(f"CORRUPTED FILES FOR NON-NUMERIC HUMANOID: {humanoid_name}")
                    logger.error(f"=" * 80)
                    logger.error(f"This humanoid cannot be auto-regenerated.")
                    logger.error(f"Deleted corrupted files at: {humanoid_gaussian_path.parent}")
                    logger.error(f"Image not found in animal_data or human_data directories")
                    logger.error(f"")
                    logger.error(f"Please manually regenerate this humanoid or skip this instance.")
                    logger.error(f"=" * 80)
                    return False

                logger.info(f"Regenerating from image...")

                try:
                    # Load image
                    image = Image.open(image_path).convert('RGB')

                    # Generate 3D using TRELLIS
                    pipeline = self.load_image_to_3d_pipeline()
                    logger.info(f"  Generating 3D from image...")
                    outputs = pipeline.run(
                        image=image,
                        seed=42,
                        formats=["gaussian", "mesh"],
                        sparse_structure_sampler_params={
                            "steps": 25,
                            "cfg_strength": 7.5,
                        },
                        slat_sampler_params={
                            "steps": 25,
                            "cfg_strength": 7.5,
                        },
                    )

                    mesh_trellis = outputs['mesh'][0]
                    gaussian_trellis = outputs['gaussian'][0]
                    logger.info(f"  Generated: {len(mesh_trellis.vertices)} vertices, {len(gaussian_trellis._xyz)} gaussians")

                    # Apply mesh guidance to create topology-aware gaussian
                    logger.info(f"  Converting mesh to topology-aware Gaussian (64 views, 10k steps)...")
                    video_path = humanoid_gaussian_path.parent / f"{humanoid_name}_topology_gaussian.mp4"

                    from mesh_guidance import convert_mesh_to_topology_gaussian
                    gaussian_final, connectivity_new = convert_mesh_to_topology_gaussian(
                        mesh_trellis=mesh_trellis,
                        gaussian_trellis=gaussian_trellis,
                        device=str(self.device),
                        optimize=True,
                        num_steps=10000,
                        num_views=64,
                        resolution=512,
                        scale_factor=1.0/1.1,
                        render_video=True,
                        video_output_path=str(video_path),
                        video_num_frames=120,
                        video_resolution=512
                    )

                    logger.info(f"  Final Gaussian: {len(gaussian_final._xyz)} points with topology")
                    logger.info(f"  Topology edges: {sum(len(v) for v in connectivity_new.values()) // 2}")

                    # Save gaussian
                    gaussian_final.save_ply(str(humanoid_gaussian_path))
                    logger.info(f"  Saved gaussian to {humanoid_gaussian_path}")

                    # Save mesh
                    import trimesh
                    mesh_path = humanoid_gaussian_path.parent / f"{humanoid_name}_mesh.glb"
                    mesh_to_save = trimesh.Trimesh(
                        vertices=mesh_trellis.vertices.cpu().numpy(),
                        faces=mesh_trellis.faces.cpu().numpy()
                    )
                    mesh_to_save.export(str(mesh_path))
                    logger.info(f"  Saved mesh to {mesh_path}")

                    # Save connectivity
                    import pickle
                    with open(connectivity_path, 'wb') as f:
                        pickle.dump(connectivity_new, f)
                    logger.info(f"  Saved connectivity to {connectivity_path}")

                    # Reload gaussian and connectivity
                    humanoid_gaussian = self.load_gaussian_from_ply(humanoid_gaussian_path)
                    self.humanoid_connectivity = connectivity_new

                    logger.info(f"  ✓ Successfully regenerated {humanoid_name}")
                    logger.info(f"  Verification: {len(humanoid_gaussian._xyz)} points, {len(connectivity_new)} vertices")
                    return True

                except Exception as e:
                    logger.error(f"Failed to regenerate from image: {e}")
                    import traceback
                    traceback.print_exc()
                    return False

        if connectivity_path.exists():
            import pickle
            with open(connectivity_path, 'rb') as f:
                self.humanoid_connectivity = pickle.load(f)

            # Check if connectivity matches gaussian points (check both keys and neighbor indices)
            num_gaussian_points = len(humanoid_gaussian._xyz)
            max_key_index = max(self.humanoid_connectivity.keys()) if self.humanoid_connectivity else -1
            max_neighbor_index = -1

            # Check all neighbor indices
            for i, neighbors in self.humanoid_connectivity.items():
                for j in neighbors.keys():
                    max_neighbor_index = max(max_neighbor_index, j)

            max_connectivity_index = max(max_key_index, max_neighbor_index)

            logger.info(f"Connectivity index check: max_key={max_key_index}, max_neighbor={max_neighbor_index}, num_points={num_gaussian_points}")

            if max_connectivity_index >= num_gaussian_points:
                logger.warning(f"Connectivity index mismatch! Max index={max_connectivity_index}, but gaussian has only {num_gaussian_points} points")
                logger.warning(f"This will cause CUDA index out of bounds errors. Regenerating...")

                # Regenerate instead of fallback to K-NN
                if not regenerate_humanoid_gaussian():
                    logger.error(f"Failed to regenerate gaussian, aborting training")
                    return False
            else:
                logger.info(f"Loaded connectivity: {len(self.humanoid_connectivity)} vertices, "
                           f"{sum(len(v) for v in self.humanoid_connectivity.values()) // 2} edges")
                logger.info(f"Connectivity valid: max_index={max_connectivity_index} < num_points={num_gaussian_points}")
        else:
            logger.warning(f"Connectivity not found at {connectivity_path}")

            # Regenerate instead of fallback to K-NN
            if not regenerate_humanoid_gaussian():
                logger.error(f"Failed to generate connectivity, aborting training")
                return False

        # Step 2: Generate object2 gaussian from object_data image (regenerate each time)
        logger.info(f"Generating object2 gaussian from object_data image...")
        try:
            object2_gaussian = self.generate_object2_gaussian_from_image(object_text)
            logger.info(f"Generated object2 gaussian: {len(object2_gaussian._xyz)} points")
        except (FileNotFoundError, Exception) as e:
            logger.error(f"Failed to generate object2 gaussian: {e}")
            return False

        # Step 3: Load humanoid_segmented and mask for Phase 1
        humanoid_segmented_path = self.flux_edit_dir / instance_name / "humanoid_segmented.png"
        humanoid_mask_path = self.flux_edit_dir / instance_name / "humanoid_mask.png"

        if not humanoid_segmented_path.exists():
            logger.error(f"Humanoid segmented image not found: {humanoid_segmented_path}")
            return False

        # Load humanoid_segmented (RGB with white background)
        humanoid_segmented = Image.open(humanoid_segmented_path).convert('RGB')
        # Resize to 518x518 for VGG-T
        humanoid_segmented = humanoid_segmented.resize((self.render_resolution, self.render_resolution), Image.LANCZOS)
        self.humanoid_gt_rgb = torch.from_numpy(np.array(humanoid_segmented)).float() / 255.0
        self.humanoid_gt_rgb = self.humanoid_gt_rgb.to(self.device)

        # Load humanoid_mask (binary mask)
        if humanoid_mask_path.exists():
            humanoid_mask = Image.open(humanoid_mask_path).convert('L')
            # Resize to 518x518
            humanoid_mask = humanoid_mask.resize((self.render_resolution, self.render_resolution), Image.LANCZOS)
            self.humanoid_gt_mask = torch.from_numpy(np.array(humanoid_mask)).float() / 255.0
            self.humanoid_gt_mask = self.humanoid_gt_mask.unsqueeze(-1).to(self.device)  # [H, W, 1]
            logger.info(f"Loaded humanoid mask from {humanoid_mask_path}")
        else:
            # Fallback: create mask from humanoid_segmented (non-white pixels)
            logger.warning(f"Humanoid mask not found at {humanoid_mask_path}, creating from segmented image")
            self.humanoid_gt_mask = (self.humanoid_gt_rgb.sum(dim=-1, keepdim=True) < 2.9).float()

        logger.info(f"Loaded humanoid segmented image and mask for Phase 2 (resized to {self.render_resolution}x{self.render_resolution})")
        logger.info(f"  humanoid_gt_rgb shape: {self.humanoid_gt_rgb.shape}")
        logger.info(f"  humanoid_gt_mask shape: {self.humanoid_gt_mask.shape}")

        # Step 4: Load object2 segmented image and mask for Phase 1
        object2_segmented_path = self.flux_edit_dir / instance_name / "object2_segmented.png"
        object2_mask_path = self.flux_edit_dir / instance_name / "object2_mask.png"

        if not object2_segmented_path.exists():
            logger.error(f"Object2 segmented image not found: {object2_segmented_path}")
            return False

        # Load object2_segmented (RGB with white background)
        object2_segmented = Image.open(object2_segmented_path).convert('RGB')
        # Resize to 518x518 for VGG-T
        object2_segmented = object2_segmented.resize((self.render_resolution, self.render_resolution), Image.LANCZOS)
        self.object2_gt_rgb = torch.from_numpy(np.array(object2_segmented)).float() / 255.0
        self.object2_gt_rgb = self.object2_gt_rgb.to(self.device)

        # Load object2_mask (binary mask)
        if object2_mask_path.exists():
            object2_mask = Image.open(object2_mask_path).convert('L')
            # Resize to 518x518
            object2_mask = object2_mask.resize((self.render_resolution, self.render_resolution), Image.LANCZOS)
            self.object2_gt_mask = torch.from_numpy(np.array(object2_mask)).float() / 255.0
            self.object2_gt_mask = self.object2_gt_mask.unsqueeze(-1).to(self.device)  # [H, W, 1]
            logger.info(f"Loaded object2 mask from {object2_mask_path}")
        else:
            # Fallback: create mask from object2_segmented (non-white pixels)
            logger.warning(f"Object2 mask not found at {object2_mask_path}, creating from segmented image")
            self.object2_gt_mask = (self.object2_gt_rgb.sum(dim=-1, keepdim=True) < 2.9).float()

        logger.info(f"Loaded object2 segmented image and mask for Phase 1 (resized to {self.render_resolution}x{self.render_resolution})")
        logger.info(f"  object2_gt_rgb shape: {self.object2_gt_rgb.shape}")
        logger.info(f"  object2_gt_mask shape: {self.object2_gt_mask.shape}")

        # Step 4.5: Auto-detect object2 position from mask if not specified
        if object_init_position is None:
            logger.info("=" * 80)
            logger.info("AUTO-DETECTING OBJECT2 INITIAL POSITION FROM MASK")
            logger.info("=" * 80)

            # Analyze object2 mask to determine which quadrant it's in
            mask_np = self.object2_gt_mask.cpu().numpy().squeeze()  # [H, W]
            H, W = mask_np.shape

            # Divide image into 4 quadrants
            half_h, half_w = H // 2, W // 2

            # Calculate mask coverage in each quadrant
            # Quadrant 1: left-top (x < W/2, y < H/2)
            q1_mask = mask_np[:half_h, :half_w]
            q1_coverage = q1_mask.mean()

            # Quadrant 2: right-top (x >= W/2, y < H/2)
            q2_mask = mask_np[:half_h, half_w:]
            q2_coverage = q2_mask.mean()

            # Quadrant 3: left-bottom (x < W/2, y >= H/2)
            q3_mask = mask_np[half_h:, :half_w]
            q3_coverage = q3_mask.mean()

            # Quadrant 4: right-bottom (x >= W/2, y >= H/2)
            q4_mask = mask_np[half_h:, half_w:]
            q4_coverage = q4_mask.mean()

            # Find quadrant with maximum coverage
            coverages = [q1_coverage, q2_coverage, q3_coverage, q4_coverage]
            max_quadrant = coverages.index(max(coverages)) + 1  # 1-indexed

            logger.info(f"  Quadrant coverages:")
            logger.info(f"    Q1 (left-top):     {q1_coverage*100:.2f}%")
            logger.info(f"    Q2 (right-top):    {q2_coverage*100:.2f}%")
            logger.info(f"    Q3 (left-bottom):  {q3_coverage*100:.2f}%")
            logger.info(f"    Q4 (right-bottom): {q4_coverage*100:.2f}%")
            logger.info(f"  Detected object2 in Quadrant {max_quadrant}")

            object_init_position = max_quadrant

        # Step 5: Initialize trainable parameters
        # Phase 1: Train humanoid gaussian parameters
        n_humanoid = humanoid_gaussian._xyz.shape[0]

        self.obj1_xyz_init = humanoid_gaussian._xyz.clone().detach()
        self.obj1_delta_xyz = torch.nn.Parameter(torch.zeros_like(humanoid_gaussian._xyz))
        self.obj1_delta_rotation = torch.nn.Parameter(torch.zeros_like(humanoid_gaussian._rotation))

        # Store original gaussians
        self.humanoid_gaussian_original = humanoid_gaussian
        self.object2_gaussian_original = object2_gaussian

        # Phase 1: Train object2 pose parameters
        # Initialize translation based on object_init_position (1-4 for corners)
        if object_init_position == 1:
            # Left-top: x=-0.5, z=0.5
            init_translation = [-0.5, 0.0, 0.5]
            logger.info(f"Initializing object2 at Quadrant 1 (left-top): {init_translation}")
        elif object_init_position == 2:
            # Right-top: x=0.5, z=0.5
            init_translation = [0.5, 0.0, 0.5]
            logger.info(f"Initializing object2 at Quadrant 2 (right-top): {init_translation}")
        elif object_init_position == 3:
            # Left-bottom: x=-0.5, z=-0.5
            init_translation = [-0.5, 0.0, -0.5]
            logger.info(f"Initializing object2 at Quadrant 3 (left-bottom): {init_translation}")
        elif object_init_position == 4:
            # Right-bottom: x=0.5, z=-0.5
            init_translation = [0.5, 0.0, -0.5]
            logger.info(f"Initializing object2 at Quadrant 4 (right-bottom): {init_translation}")
        else:
            # Default: center
            init_translation = [0.0, 0.0, 0.0]
            logger.info(f"Initializing object2 at center (default): {init_translation}")

        self.obj2_translation = torch.nn.Parameter(torch.tensor(init_translation, device=self.device))
        self.obj2_quaternion = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device))
        self.obj2_scale = torch.nn.Parameter(torch.tensor([1.0], device=self.device))

        logger.info(f"Initialized trainable parameters:")
        logger.info(f"  Humanoid points: {n_humanoid}")
        logger.info(f"  Object2 points: {len(object2_gaussian._xyz)}")

        # Step 6: Load MVAdapter object masks for Phase 1 multi-view guidance
        # MVAdapter angles and their corresponding TRELLIS angles
        # MVAdapter angle -> TRELLIS angle mapping: TRELLIS = 225 - MVAdapter
        # 0° is excluded as per user request
        mvadapter_angles = [45, 90, 180, 270, 315]
        self.mvadapter_object_masks = {}
        self.mvadapter_to_trellis_mapping = {}

        mvadapter_dir = self.flux_edit_dir / instance_name / "mvadapter_object"
        if mvadapter_dir.exists() and mvadapter_dir.is_dir():
            # Try to load masks, skip if directory is empty or files are missing
            for mvadapter_angle in mvadapter_angles:
                mask_path = mvadapter_dir / f"mask_{mvadapter_angle:03d}.png"

                if mask_path.exists():
                    try:
                        # Load mask
                        mask_img = Image.open(mask_path).convert('L')
                        mask_img = mask_img.resize((self.render_resolution, self.render_resolution), Image.LANCZOS)
                        mask_tensor = torch.from_numpy(np.array(mask_img)).float() / 255.0
                        mask_tensor = mask_tensor.unsqueeze(-1).to(self.device)  # [H, W, 1]

                        # Calculate corresponding TRELLIS angle
                        trellis_angle = (225 - mvadapter_angle) % 360

                        self.mvadapter_object_masks[trellis_angle] = mask_tensor
                        self.mvadapter_to_trellis_mapping[mvadapter_angle] = trellis_angle
                    except Exception as e:
                        # Skip this mask if loading fails
                        logger.debug(f"  Skipped mask at {mask_path}: {e}")
                        continue

            # Only log if we actually loaded some masks
            if self.mvadapter_object_masks:
                logger.info("=" * 80)
                logger.info("LOADING MVADAPTER OBJECT MASKS FOR PHASE 1")
                logger.info("=" * 80)
                for mvadapter_angle, trellis_angle in sorted(self.mvadapter_to_trellis_mapping.items()):
                    mask_coverage = self.mvadapter_object_masks[trellis_angle].mean().item()
                    logger.info(f"  MVAdapter {mvadapter_angle:03d}° -> TRELLIS {trellis_angle:03d}°, mask coverage: {mask_coverage:.4f}")
                logger.info(f"Loaded {len(self.mvadapter_object_masks)} MVAdapter object masks")
            else:
                # Directory exists but no valid masks found - silently proceed
                self.mvadapter_object_masks = {}
        else:
            # Directory doesn't exist - silently proceed
            self.mvadapter_object_masks = {}

        # Step 7: Training loop
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Phase 1 (0-{self.phase1_end-1}): Train object2 pose (RGB + Mask MSE + IOU + MVAdapter MV masks)")
        if self.mvadapter_object_masks:
            logger.info(f"  - Multi-view mask guidance from MVAdapter object masks: {list(self.mvadapter_object_masks.keys())}°")
        logger.info(f"Phase 2 ({self.phase1_end}-{self.phase2_end-1}): Train humanoid gaussian")
        logger.info(f"  - Multi-view mask guidance from TRELLIS-generated reference (0, 90, 180, 270°)")
        logger.info(f"  - RGB loss at 225° + Main view mask + Multi-view mask + ARAP")

        # Setup optimizer for Phase 1 (object2 pose)
        optimizer = torch.optim.Adam([
            {'params': [self.obj2_translation, self.obj2_quaternion, self.obj2_scale], 'lr': self.learning_rate_pose},
        ], betas=(0.9, 0.99), eps=1e-15)

        # Training loop
        for step in range(self.phase2_end):
            # Phase transition
            if step == self.phase1_end:
                logger.info("=" * 60)
                logger.info(f"=== PHASE 2 START (step {step}) ===")
                logger.info("Switching optimizer to humanoid gaussian parameters...")
                logger.info("=" * 60)

                # Save Phase 1 result (object2)
                phase1_gaussian = self.compose_object2_gaussian()
                phase1_path = instance_output_dir / "phase1_object2.ply"
                phase1_gaussian.save_ply(str(phase1_path))
                logger.info(f"Saved Phase 1 result to {phase1_path}")

                # Generate reference gaussian from humanoid_segmented for multi-view mask guidance
                logger.info("=" * 60)
                logger.info("Generating reference gaussian from humanoid_segmented.png for mask guidance...")
                logger.info("=" * 60)

                # Load humanoid_segmented image
                humanoid_segmented_path = self.flux_edit_dir / instance_name / "humanoid_segmented.png"
                humanoid_segmented_img = Image.open(humanoid_segmented_path).convert('RGB')

                # Generate 3D gaussian using TRELLIS
                pipeline = self.load_image_to_3d_pipeline()
                logger.info(f"Running TRELLIS image-to-3D on humanoid_segmented.png...")
                outputs = pipeline.run(
                    image=humanoid_segmented_img,
                    seed=42,
                    formats=["gaussian"],
                    sparse_structure_sampler_params={
                        "steps": 25,
                        "cfg_strength": 7.5,
                    },
                    slat_sampler_params={
                        "steps": 25,
                        "cfg_strength": 7.5,
                    },
                )

                reference_gaussian = outputs['gaussian'][0]
                logger.info(f"Generated reference gaussian: {len(reference_gaussian._xyz)} points")

                # Render and align multi-view references to original humanoid renders
                logger.info("Rendering multi-view references and aligning to original humanoid...")
                self.reference_angles = [0.0, 90.0, 180.0, 270.0]
                self.reference_masks = {}
                # self.reference_images = {}  # Commented out: not using RGB guidance

                # Create reference render directory
                ref_render_dir = instance_output_dir / "phase2_reference_renders"
                ref_render_dir.mkdir(exist_ok=True, parents=True)

                for angle in self.reference_angles:
                    azimuth = np.radians(angle)

                    # 1. Render ORIGINAL humanoid gaussian at this angle (alignment target)
                    target_rgb, target_mask = self.render_gaussian_view(
                        self.humanoid_gaussian_original,
                        azimuth=azimuth,
                        elevation=self.reference_elevation,
                        radius=self.reference_radius,
                        return_mask=True,
                        bg_color=(1.0, 1.0, 1.0)  # White background
                    )

                    # Save target render for debugging
                    target_save = (target_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(target_save).save(ref_render_dir / f"target_humanoid_{int(angle):03d}.png")

                    # 2. Render reference gaussian at this angle
                    ref_rgb, ref_mask = self.render_gaussian_view(
                        reference_gaussian,
                        azimuth=azimuth,
                        elevation=self.reference_elevation,
                        radius=self.reference_radius,
                        return_mask=True,
                        bg_color=(1.0, 1.0, 1.0)  # White background
                    )

                    # Save raw reference render for debugging
                    ref_raw_save = (ref_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(ref_raw_save).save(ref_render_dir / f"reference_rgb_raw_{int(angle):03d}.png")

                    # 3. Align reference render to target humanoid render (like align_mvadapter_to_humanoid_render_bbox)
                    aligned_rgb, aligned_mask = self.align_reference_to_target(
                        ref_rgb,
                        ref_mask,
                        target_rgb,
                        target_mask,
                        angle=angle  # Pass angle for horizontal alignment decision
                    )

                    # Store aligned reference
                    self.reference_masks[angle] = aligned_mask.detach()
                    # self.reference_images[angle] = aligned_rgb.detach()  # Commented out: not using RGB guidance

                    # Save aligned renders
                    aligned_save = (aligned_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(aligned_save).save(ref_render_dir / f"reference_rgb_{int(angle):03d}.png")

                    mask_save = (aligned_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_save.squeeze(-1), mode='L').save(ref_render_dir / f"reference_mask_{int(angle):03d}.png")

                    logger.info(f"  Angle {angle:03.0f}°: aligned reference to target, mask coverage={aligned_mask.mean().item():.4f}")

                logger.info(f"Reference masks generated and saved to {ref_render_dir}")

                # Switch optimizer to humanoid
                optimizer = torch.optim.Adam([
                    {'params': [self.obj1_delta_xyz, self.obj1_delta_rotation],
                     'lr': self.learning_rate_gaussian},
                ], betas=(0.9, 0.99), eps=1e-15)

            optimizer.zero_grad()

            # Phase-specific rendering and loss computation
            if step < self.phase1_end:
                # Phase 1: Train object2 pose only (single view at 225°)
                current_gaussian = self.compose_object2_gaussian()

                # Render at 225 degrees with WHITE background
                azimuth = np.radians(225.0)
                rendered_rgb, rendered_mask = self.render_gaussian_view(
                    current_gaussian,
                    azimuth=azimuth,
                    elevation=self.reference_elevation,
                    radius=self.reference_radius,
                    return_mask=True,
                    bg_color=self.bg_color  # White background
                )

                # GT image (object2 segmented)
                gt_rgb = self.object2_gt_rgb
                gt_mask = self.object2_gt_mask

                # Resize if needed
                if gt_rgb.shape[:2] != rendered_rgb.shape[:2]:
                    gt_rgb = F.interpolate(
                        gt_rgb.permute(2, 0, 1).unsqueeze(0),
                        size=rendered_rgb.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)

                    gt_mask = F.interpolate(
                        gt_mask.permute(2, 0, 1).unsqueeze(0),
                        size=rendered_rgb.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)

                # Phase 1 (Object2): RGB + Mask MSE + IOU + MVAdapter multi-view mask guidance
                # 1. RGB Loss
                loss_rgb = F.mse_loss(rendered_rgb, gt_rgb)

                # 2. Mask MSE Loss - pixel-wise mask matching
                loss_mask_mse = F.mse_loss(rendered_mask, gt_mask)

                # 3. IOU Loss - spatial alignment for mask
                intersection = (rendered_mask * gt_mask).sum()
                union = (rendered_mask + gt_mask - rendered_mask * gt_mask).sum()
                iou = intersection / (union + 1e-6)
                loss_iou = 1.0 - iou

                # 4. MVAdapter multi-view mask guidance (only mask IOU loss)
                loss_mvadapter_multiview = torch.tensor(0.0, device=self.device)
                if self.mvadapter_object_masks:
                    num_mvadapter_views = len(self.mvadapter_object_masks)

                    for trellis_angle, mvadapter_mask in self.mvadapter_object_masks.items():
                        # Render object2 at this TRELLIS angle
                        azimuth_mv = np.radians(float(trellis_angle))
                        rendered_rgb_mv, rendered_mask_mv = self.render_gaussian_view(
                            current_gaussian,
                            azimuth=azimuth_mv,
                            elevation=self.reference_elevation,
                            radius=self.reference_radius,
                            return_mask=True,
                            bg_color=self.bg_color
                        )

                        # Compute mask IOU loss with MVAdapter mask
                        intersection_mv = (rendered_mask_mv * mvadapter_mask).sum()
                        union_mv = (rendered_mask_mv + mvadapter_mask - rendered_mask_mv * mvadapter_mask).sum()
                        iou_mv = intersection_mv / (union_mv + 1e-6)
                        loss_mvadapter_multiview += (1.0 - iou_mv)

                    # Average over views
                    loss_mvadapter_multiview = loss_mvadapter_multiview / num_mvadapter_views

                # Phase 1 (Object2): RGB + Mask MSE + IOU + MVAdapter MV masks
                # MVAdapter multi-view weight is 1.0 (same as weight_mask)
                total_loss = (loss_rgb * self.weight_rgb +
                             loss_mask_mse * self.weight_mask +
                             loss_iou * self.weight_mask)

                if self.mvadapter_object_masks:
                    total_loss += loss_mvadapter_multiview * 1.0  # Weight = 1.0

                # NO ARAP loss for Phase 1 (object2 pose training)
                # ARAP is only used in Phase 2 for humanoid gaussian training

                # Log every 200 steps
                if step % 200 == 0:
                    rendered_mask_coverage = rendered_mask.mean().item()
                    gt_mask_coverage = gt_mask.mean().item()

                    if self.mvadapter_object_masks:
                        logger.info(f"Phase 1 Step {step}: Loss={total_loss.item():.6f}, "
                                   f"RGB={loss_rgb.item():.6f}, Mask_MSE={loss_mask_mse.item():.6f}, "
                                   f"IOU={iou.item():.4f}, IOU_Loss={loss_iou.item():.6f}, "
                                   f"MVAdapter_MV={loss_mvadapter_multiview.item():.6f}")
                    else:
                        logger.info(f"Phase 1 Step {step}: Loss={total_loss.item():.6f}, "
                                   f"RGB={loss_rgb.item():.6f}, Mask_MSE={loss_mask_mse.item():.6f}, "
                                   f"IOU={iou.item():.4f}, IOU_Loss={loss_iou.item():.6f}")
                    logger.info(f"  Mask coverage - Rendered: {rendered_mask_coverage:.4f}, GT: {gt_mask_coverage:.4f}")

                    # Save renders
                    step_dir = instance_output_dir / f"phase1_step_{step:04d}"
                    step_dir.mkdir(exist_ok=True, parents=True)

                    # Save rendered RGB and mask at 225 degrees
                    img_save = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_save).save(step_dir / "render_object2_225.png")

                    mask_save = (rendered_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_save.squeeze(-1), mode='L').save(step_dir / "mask_object2_225.png")

                    # Save MVAdapter multi-view renders
                    if self.mvadapter_object_masks:
                        for trellis_angle, mvadapter_mask in self.mvadapter_object_masks.items():
                            azimuth_mv = np.radians(float(trellis_angle))
                            rendered_rgb_mv, rendered_mask_mv = self.render_gaussian_view(
                                current_gaussian,
                                azimuth=azimuth_mv,
                                elevation=self.reference_elevation,
                                radius=self.reference_radius,
                                return_mask=True,
                                bg_color=self.bg_color
                            )

                            # Save rendered and reference masks
                            img_mv_save = (rendered_rgb_mv.detach().cpu().numpy() * 255).astype(np.uint8)
                            Image.fromarray(img_mv_save).save(step_dir / f"render_mvadapter_{int(trellis_angle):03d}.png")

                            mask_mv_save = (rendered_mask_mv.detach().cpu().numpy() * 255).astype(np.uint8)
                            Image.fromarray(mask_mv_save.squeeze(-1), mode='L').save(step_dir / f"mask_mvadapter_{int(trellis_angle):03d}.png")

                        # Save MVAdapter reference masks (only once at step 0)
                        if step == 0:
                            for trellis_angle, mvadapter_mask in self.mvadapter_object_masks.items():
                                ref_mask_save = (mvadapter_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                                Image.fromarray(ref_mask_save.squeeze(-1), mode='L').save(step_dir / f"gt_mvadapter_mask_{int(trellis_angle):03d}.png")

                    # Save GT (only once at step 0)
                    if step == 0:
                        gt_rgb_save = (gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(gt_rgb_save).save(step_dir / "gt_object2_segmented.png")

                        gt_mask_save = (gt_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(gt_mask_save.squeeze(-1), mode='L').save(step_dir / "gt_object2_mask.png")

            else:
                # Phase 2: Train humanoid gaussian with multi-view mask guidance
                current_gaussian = self.compose_humanoid_gaussian()

                # Multi-view mask guidance loss (using aligned references from 0, 90, 180, 270 degrees)
                # No need to align during training - reference masks are already aligned to original humanoid render
                loss_mask_multiview = torch.tensor(0.0, device=self.device)
                num_views = len(self.reference_angles)

                for angle in self.reference_angles:
                    azimuth = np.radians(angle)

                    # Render current gaussian at this angle (same rendering as original humanoid)
                    rendered_rgb, rendered_mask = self.render_gaussian_view(
                        current_gaussian,
                        azimuth=azimuth,
                        elevation=self.reference_elevation,
                        radius=self.reference_radius,
                        return_mask=True,
                        bg_color=(1.0, 1.0, 1.0)  # White background
                    )

                    # Get aligned reference mask for this angle
                    ref_mask = self.reference_masks[angle]

                    # Compute mask IOU loss (both masks are in same space now)
                    intersection = (rendered_mask * ref_mask).sum()
                    union = (rendered_mask + ref_mask - rendered_mask * ref_mask).sum()
                    iou = intersection / (union + 1e-6)
                    loss_mask_multiview += (1.0 - iou)

                # Average over views
                loss_mask_multiview = loss_mask_multiview / num_views

                # Also compute loss at 225 degrees for RGB guidance (main view)
                azimuth_main = np.radians(225.0)
                rendered_rgb_main, rendered_mask_main = self.render_gaussian_view(
                    current_gaussian,
                    azimuth=azimuth_main,
                    elevation=self.reference_elevation,
                    radius=self.reference_radius,
                    return_mask=True,
                    bg_color=self.bg_color  # White background
                )

                # GT image (humanoid segmented with white background)
                gt_rgb = self.humanoid_gt_rgb
                gt_mask = self.humanoid_gt_mask

                # Resize if needed
                if gt_rgb.shape[:2] != rendered_rgb_main.shape[:2]:
                    gt_rgb = F.interpolate(
                        gt_rgb.permute(2, 0, 1).unsqueeze(0),
                        size=rendered_rgb_main.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)

                    gt_mask = F.interpolate(
                        gt_mask.permute(2, 0, 1).unsqueeze(0),
                        size=rendered_rgb_main.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)

                # Phase 2 (Humanoid): RGB loss at 225 degrees + Multi-view mask guidance
                # 1. RGB Loss at main view (225 degrees)
                loss_rgb = F.mse_loss(rendered_rgb_main, gt_rgb)

                # 2. Main view mask IOU loss
                intersection_main = (rendered_mask_main * gt_mask).sum()
                union_main = (rendered_mask_main + gt_mask - rendered_mask_main * gt_mask).sum()
                iou_main = intersection_main / (union_main + 1e-6)
                loss_mask_main = 1.0 - iou_main

                # Combine: RGB (main only) + mask (main + multi-view)
                total_loss = (loss_rgb * self.weight_rgb +
                             loss_mask_main * self.weight_mask +
                             loss_mask_multiview * self.weight_mask)

                # 4. ARAP loss - only for Phase 2 (humanoid training)
                try:
                    if self.humanoid_connectivity is not None:
                        # Use mesh topology connectivity
                        if not hasattr(self, '_arap_connectivity_cache'):
                            # Build connectivity cache on first use
                            logger.info("  Building ARAP connectivity cache from mesh topology...")
                            K = 3  # Use top-3 neighbors
                            n_humanoid = self.obj1_xyz_init.shape[0]

                            ii_list, jj_list, nn_list, weight_list = [], [], [], []
                            for i, neighbors in self.humanoid_connectivity.items():
                                # Skip if vertex index is out of range
                                if i >= n_humanoid:
                                    continue

                                # Get distances to all neighbors, filter valid ones
                                neighbor_items = sorted(neighbors.items(), key=lambda x: x[1])[:K]
                                valid_neighbor_idx = 0
                                for j, dist in neighbor_items:
                                    # Skip if neighbor index is out of range
                                    if j >= n_humanoid:
                                        continue

                                    ii_list.append(i)
                                    jj_list.append(j)
                                    nn_list.append(valid_neighbor_idx)
                                    weight_list.append(1.0 / (dist + 1e-6))
                                    valid_neighbor_idx += 1

                            # Verify all indices are valid before creating tensors
                            max_ii = max(ii_list) if ii_list else -1
                            max_jj = max(jj_list) if jj_list else -1
                            max_nn = max(nn_list) if nn_list else -1

                            logger.info(f"  Built connectivity lists: {len(ii_list)} edges")
                            logger.info(f"  Pre-check: max_ii={max_ii}, max_jj={max_jj}, max_nn={max_nn}, n_humanoid={n_humanoid}, K={K}")

                            if max_ii >= n_humanoid or max_jj >= n_humanoid or max_nn >= K:
                                logger.error(f"Invalid indices detected after filtering!")
                                self.humanoid_connectivity = None
                            else:
                                # Build weight tensor using for loop (same as final_loss.py)
                                logger.info(f"  Building weight tensor [{n_humanoid}, {K}]...")
                                weight_tensor = torch.zeros(n_humanoid, K, dtype=torch.float32).to(self.device)

                                # Use for loop for safe element-wise assignment
                                for idx in range(len(ii_list)):
                                    weight_tensor[ii_list[idx], nn_list[idx]] = weight_list[idx]

                                # Normalize weights per vertex
                                weight_sum = weight_tensor.sum(dim=1, keepdim=True).clamp(min=1e-6)
                                weight = weight_tensor / weight_sum

                                # Convert lists to GPU tensors
                                ii = torch.tensor(ii_list, dtype=torch.long).to(self.device)
                                jj = torch.tensor(jj_list, dtype=torch.long).to(self.device)
                                nn = torch.tensor(nn_list, dtype=torch.long).to(self.device)

                                self._arap_connectivity_cache = (ii, jj, nn, weight, K)
                                logger.info(f"  ARAP cache built successfully: {len(ii)} edges, K={K}")

                        # Create mock generated_parts for LossComputer
                        generated_parts = [{
                            'connectivity': self.humanoid_connectivity
                        }]

                        # Verify dimensions before ARAP computation
                        current_n_points = self.obj1_xyz_init.shape[0]
                        if hasattr(self, '_arap_connectivity_cache'):
                            cache_ii, cache_jj, _, _, _ = self._arap_connectivity_cache
                            if cache_ii.max() >= current_n_points or cache_jj.max() >= current_n_points:
                                logger.error(f"  ARAP cache indices out of bounds!")
                                logger.error(f"  max_ii={cache_ii.max()}, max_jj={cache_jj.max()}, n_points={current_n_points}")
                                logger.error(f"  This indicates connectivity/gaussian mismatch. Deleting corrupted files...")

                                # Delete existing files to force regeneration
                                # Use the correct directory from humanoid_gaussian_path
                                humanoid_dir = humanoid_gaussian_path.parent
                                files_to_delete = [
                                    humanoid_dir / f"{humanoid_name}_topology_gaussian.ply",
                                    humanoid_dir / f"{humanoid_name}_mesh.glb",
                                    humanoid_dir / f"{humanoid_name}_connectivity.pkl",
                                    humanoid_dir / f"{humanoid_name}_topology_gaussian.mp4"
                                ]
                                for file_path in files_to_delete:
                                    if file_path.exists():
                                        logger.info(f"    Deleting: {file_path.name}")
                                        file_path.unlink()

                                logger.error(f"  Deleted corrupted files. Please restart training for this instance.")
                                return False

                        loss_arap, _ = self.loss_computer.compute_arap_loss(
                            self.obj1_xyz_init,
                            self.obj1_delta_xyz,
                            self.humanoid_gaussian_original.aabb,
                            generated_parts,
                            connectivity_cache=getattr(self, '_arap_connectivity_cache', None)
                        )
                    else:
                        # Fallback to K-NN
                        loss_arap, _ = self.loss_computer.compute_arap_loss(
                            self.obj1_xyz_init,
                            self.obj1_delta_xyz,
                            self.humanoid_gaussian_original.aabb,
                            None,
                            connectivity_cache=getattr(self, '_arap_connectivity_cache', None)
                        )
                except (RuntimeError, torch.cuda.CudaError) as e:
                    logger.error(f"ARAP computation failed: {e}")
                    logger.error(f"This indicates corrupted mesh/connectivity. Regenerating humanoid gaussian...")

                    # Delete existing files to force regeneration
                    # Use the correct directory from humanoid_gaussian_path
                    humanoid_dir = humanoid_gaussian_path.parent
                    files_to_delete = [
                        humanoid_dir / f"{humanoid_name}_topology_gaussian.ply",
                        humanoid_dir / f"{humanoid_name}_mesh.glb",
                        humanoid_dir / f"{humanoid_name}_connectivity.pkl",
                        humanoid_dir / f"{humanoid_name}_topology_gaussian.mp4"
                    ]
                    for file_path in files_to_delete:
                        if file_path.exists():
                            logger.info(f"  Deleting: {file_path.name}")
                            file_path.unlink()

                    logger.error(f"Deleted corrupted files. Please restart training for this instance.")
                    return False

                # ARAP weight (fixed at 1.0)
                weight_arap_effective = 1.0
                total_loss += loss_arap * weight_arap_effective

                # Log every 200 steps
                if step % 200 == 0:
                    logger.info(f"Phase 2 Step {step}: Loss={total_loss.item():.6f}, "
                               f"RGB_main={loss_rgb.item():.6f}, "
                               f"IOU_main={iou_main.item():.4f}, Mask_MV={loss_mask_multiview.item():.6f}, "
                               f"ARAP={loss_arap.item():.6f}")

                    # Save renders
                    step_dir = instance_output_dir / f"phase2_step_{step:04d}"
                    step_dir.mkdir(exist_ok=True, parents=True)

                    # Save main view (225 degrees) RGB and mask
                    img_save = (rendered_rgb_main.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_save).save(step_dir / "render_humanoid_225.png")

                    mask_save = (rendered_mask_main.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_save.squeeze(-1), mode='L').save(step_dir / "mask_humanoid_225.png")

                    # Save multi-view renders for visualization
                    for angle in self.reference_angles:
                        azimuth = np.radians(angle)
                        rendered_rgb_mv, rendered_mask_mv = self.render_gaussian_view(
                            current_gaussian,
                            azimuth=azimuth,
                            elevation=self.reference_elevation,
                            radius=self.reference_radius,
                            return_mask=True,
                            bg_color=(1.0, 1.0, 1.0)
                        )

                        img_mv_save = (rendered_rgb_mv.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_mv_save).save(step_dir / f"render_mv_{int(angle):03d}.png")

                        mask_mv_save = (rendered_mask_mv.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(mask_mv_save.squeeze(-1), mode='L').save(step_dir / f"mask_mv_{int(angle):03d}.png")

                    # Save GT (only once at step 0 of phase 2)
                    if step == self.phase1_end:
                        gt_rgb_save = (gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(gt_rgb_save).save(step_dir / "gt_humanoid_segmented.png")

                        gt_mask_save = (gt_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(gt_mask_save.squeeze(-1), mode='L').save(step_dir / "gt_humanoid_mask.png")

            # Backward and step
            total_loss.backward()
            optimizer.step()

        # Step 7: Save final results
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)

        # Save Phase 2 result (humanoid only)
        phase2_gaussian = self.compose_humanoid_gaussian()
        phase2_path = instance_output_dir / "phase2_humanoid.ply"
        phase2_gaussian.save_ply(str(phase2_path))
        logger.info(f"Saved Phase 2 result to {phase2_path}")

        # Save combined result
        combined_gaussian = self.compose_combined_gaussian()
        combined_path = instance_output_dir / "combined_phase1_phase2.ply"
        combined_gaussian.save_ply(str(combined_path))
        logger.info(f"Saved combined result to {combined_path}")

        # Render combined result at multiple angles (for visualization only)
        logger.info("Rendering multi-angle views for visualization...")
        render_dir = instance_output_dir / "final_renders"
        render_dir.mkdir(exist_ok=True, parents=True)

        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            azimuth = np.radians(angle)
            rendered_rgb = self.render_gaussian_view(
                combined_gaussian,
                azimuth=azimuth,
                elevation=0.0,
                radius=self.reference_radius
            )
            img_save = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_save).save(render_dir / f"view_{angle:03d}.png")

        logger.info(f"Saved renders to {render_dir}")

        # Render 360-degree rotation video
        logger.info("Rendering 360-degree video...")
        video_frames = []
        num_frames = 120  # 4 seconds at 30fps
        for frame_idx in range(num_frames):
            # Rotate 360 degrees
            azimuth = np.radians(frame_idx * 360.0 / num_frames)
            rendered_rgb = self.render_gaussian_view(
                combined_gaussian,
                azimuth=azimuth,
                elevation=0.0,
                radius=self.reference_radius
            )
            frame = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            video_frames.append(frame)

        # Save video
        import imageio
        video_path = instance_output_dir / f"{instance_name}_combined.mp4"
        imageio.mimsave(str(video_path), video_frames, fps=30)
        logger.info(f"Saved video to {video_path}")

        logger.info(f"Instance {instance_name} training complete!")

        return True

    def _multiply_quaternions(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (w,x,y,z format)"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=-1)

    def compose_humanoid_gaussian(self) -> Gaussian:
        """Compose humanoid gaussian with trainable deltas"""
        gaussian = Gaussian(
            aabb=self.humanoid_gaussian_original.aabb.clone(),
            sh_degree=0,
            device=str(self.device)
        )

        # Apply deltas
        gaussian._xyz = self.obj1_xyz_init + self.obj1_delta_xyz
        gaussian._features_dc = self.humanoid_gaussian_original._features_dc.clone()
        gaussian._scaling = self.humanoid_gaussian_original._scaling.clone()
        gaussian._rotation = self.humanoid_gaussian_original._rotation + self.obj1_delta_rotation
        gaussian._opacity = self.humanoid_gaussian_original._opacity.clone()

        return gaussian

    def compose_object2_gaussian(self) -> Gaussian:
        """Compose object2 gaussian with trainable pose (with gradient flow)"""
        # Get object2's original Gaussian parameters (frozen)
        obj2_xyz_normalized = self.object2_gaussian_original._xyz.clone().detach()
        obj2_aabb = self.object2_gaussian_original.aabb
        obj2_xyz_local = obj2_xyz_normalized * obj2_aabb[3:] + obj2_aabb[:3]
        N_obj2 = obj2_xyz_local.shape[0]

        # Get frozen parameters
        rots_bias = torch.zeros((4), device=self.device)
        rots_bias[0] = 1
        obj2_rotations_full = self.object2_gaussian_original._rotation.clone().detach() + rots_bias[None, :]
        obj2_scalings = self.object2_gaussian_original._scaling.clone().detach()
        obj2_opacities = self.object2_gaussian_original.get_opacity.clone().detach()
        obj2_features_dc = self.object2_gaussian_original._features_dc.clone().detach()

        # Apply trainable global transformation
        quaternion = self.obj2_quaternion / (torch.norm(self.obj2_quaternion) + 1e-8)
        translation = self.obj2_translation
        scale = torch.clamp(self.obj2_scale, min=0.5)

        # 1. Scale
        scaled_xyz = obj2_xyz_local * scale

        # 2. Rotation (using differentiable quaternion to rotation matrix)
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        rot_matrix = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y])
        ])

        rotated_xyz = scaled_xyz @ rot_matrix.T

        # Rotate gaussian orientations (multiply quaternions)
        transformed_rotations_full = self._multiply_quaternions(
            quaternion.unsqueeze(0).expand(N_obj2, -1),
            obj2_rotations_full
        )

        # 3. Translation
        final_xyz = rotated_xyz + translation

        # Transform scaling (in log space)
        transformed_scalings = obj2_scalings + torch.log(scale)

        # Create gaussian
        gaussian = Gaussian(
            aabb=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            sh_degree=0,
            device=str(self.device)
        )

        # Normalize coordinates to AABB (CRITICAL: convert world coords back to normalized coords)
        obj2_aabb_normalized = gaussian.aabb
        obj2_means_normalized = (final_xyz - obj2_aabb_normalized[:3]) / obj2_aabb_normalized[3:]

        # Set parameters
        gaussian._xyz = obj2_means_normalized
        gaussian._features_dc = obj2_features_dc
        gaussian._scaling = transformed_scalings
        gaussian._rotation = transformed_rotations_full - rots_bias[None, :]
        gaussian._opacity = obj2_opacities

        return gaussian

    def compose_combined_gaussian(self) -> Gaussian:
        """Compose combined gaussian (humanoid + object2)"""
        humanoid_gaussian = self.compose_humanoid_gaussian()
        object2_gaussian = self.compose_object2_gaussian()

        # Combine
        combined = Gaussian(
            aabb=humanoid_gaussian.aabb.clone(),
            sh_degree=0,
            device=str(self.device)
        )

        combined._xyz = torch.cat([humanoid_gaussian._xyz, object2_gaussian._xyz], dim=0)
        combined._features_dc = torch.cat([humanoid_gaussian._features_dc, object2_gaussian._features_dc], dim=0)
        combined._scaling = torch.cat([humanoid_gaussian._scaling, object2_gaussian._scaling], dim=0)
        combined._rotation = torch.cat([humanoid_gaussian._rotation, object2_gaussian._rotation], dim=0)
        combined._opacity = torch.cat([humanoid_gaussian._opacity, object2_gaussian._opacity], dim=0)

        return combined

    def run(self):
        """Run batch reconstruction training"""
        logger.info("=" * 80)
        logger.info("BATCH RECONSTRUCTION TRAINING V3")
        logger.info("=" * 80)
        logger.info(f"Target instances: {self.target_instances}")

        # Track failures and skips
        failed_instances = []
        succeeded_instances = []
        skipped_instances = []

        for instance_name in self.target_instances:
            # Check if instance already completed (has combined output)
            instance_output_dir = self.output_base_dir / instance_name
            combined_output_path = instance_output_dir / "combined_phase1_phase2.ply"

            if combined_output_path.exists():
                logger.info(f"⊙ {instance_name} already completed, skipping...")
                skipped_instances.append(instance_name)
                continue

            try:
                success = self.train_phase1_phase2(instance_name)
                if success:
                    logger.info(f"✓ {instance_name} completed successfully")
                    succeeded_instances.append(instance_name)
                else:
                    logger.error(f"✗ {instance_name} failed")
                    failed_instances.append(instance_name)
            except Exception as e:
                logger.error(f"✗ {instance_name} failed with exception: {e}")

                # Check if CUDA error
                is_cuda_error = "cuda" in str(e).lower() or "device-side assert" in str(e).lower()

                if is_cuda_error:
                    logger.error("CUDA error detected - skipping to next instance")
                    logger.error("Note: CUDA context may be broken, subsequent instances may fail")
                    # Don't call any CUDA operations after CUDA error - they will crash
                else:
                    import traceback
                    traceback.print_exc()

                failed_instances.append(instance_name)

            # Clean up CUDA cache after each instance (skip if CUDA is broken)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    # CUDA context is broken, skip cleanup and continue
                    pass

        logger.info("=" * 80)
        logger.info("BATCH TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total instances: {len(self.target_instances)}")
        logger.info(f"Skipped (already done): {len(skipped_instances)}")
        logger.info(f"Succeeded (newly trained): {len(succeeded_instances)}")
        logger.info(f"Failed: {len(failed_instances)}")

        # Write failed instances to log file
        if failed_instances:
            log_dir = self.output_base_dir / "logs"
            log_dir.mkdir(exist_ok=True, parents=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_log_path = log_dir / f"failed_instances_{timestamp}.txt"

            with open(failed_log_path, 'w') as f:
                f.write(f"Failed instances ({len(failed_instances)}/{len(self.target_instances)}):\n")
                f.write("=" * 80 + "\n")
                for instance in failed_instances:
                    f.write(f"{instance}\n")

            logger.info(f"Failed instances logged to: {failed_log_path}")
            logger.info("")
            logger.info("If failures were due to CUDA errors, please restart training:")
            logger.info("  python batch_recon3.py --all")
            logger.info("Already completed instances will be automatically skipped.")
        else:
            logger.info("All instances completed successfully!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch reconstruction training V3 (single reference + Phase 2)')
    parser.add_argument('--flux_edit_dir', type=str,
                       default='/data4/zishuo/TRELLIS/flux_edit_multiview',
                       help='Directory containing flux_edit_multiview results')
    parser.add_argument('--mesh_gaussian_dir', type=str,
                       default='/data4/zishuo/mesh_gaussian',
                       help='Directory containing mesh_gaussian humanoids')
    parser.add_argument('--renders_cond_dir', type=str,
                       default='/data4/zishuo/TRELLIS/datasets/ObjaverseXL_sketchfab/renders_cond',
                       help='Directory containing renders_cond images')
    parser.add_argument('--output_dir', type=str,
                       default='/data4/zishuo/TRELLIS/batch_recon3_output',
                       help='Output directory for training results')
    parser.add_argument('--instances', nargs='+',
                       default=None,
                       help='List of instances to train (e.g., 291_humanoid_038 066_character_132)')
    parser.add_argument('--all', action='store_true',
                       help='Process all instances in flux_edit_multiview directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    # Determine which instances to process
    if args.all:
        # Scan flux_edit_multiview directory for all instances
        flux_edit_path = Path(args.flux_edit_dir)
        all_instances = []
        for item in sorted(flux_edit_path.iterdir()):
            if item.is_dir() and (item / "metadata.json").exists():
                all_instances.append(item.name)
        logger.info(f"Found {len(all_instances)} instances in {args.flux_edit_dir}")
        logger.info(f"Instances: {all_instances}")
        target_instances = all_instances
    elif args.instances:
        target_instances = args.instances
    else:
        # Default: process all instances
        flux_edit_path = Path(args.flux_edit_dir)
        all_instances = []
        for item in sorted(flux_edit_path.iterdir()):
            if item.is_dir() and (item / "metadata.json").exists():
                all_instances.append(item.name)
        logger.info(f"No instances specified, processing all {len(all_instances)} instances by default")
        logger.info(f"Instances: {all_instances}")
        logger.info(f"Use --instances to specify specific instances")
        target_instances = all_instances

    trainer = BatchReconstructionTrainerV3(
        flux_edit_dir=args.flux_edit_dir,
        mesh_gaussian_dir=args.mesh_gaussian_dir,
        renders_cond_dir=args.renders_cond_dir,
        output_base_dir=args.output_dir,
        target_instances=target_instances,
        device=args.device
    )

    trainer.run()


if __name__ == '__main__':
    main()
