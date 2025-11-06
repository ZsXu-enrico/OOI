#!/usr/bin/env python3
"""
Man Scene with Reconstruction + ARAP Training
Generates 'A young man sits on a chair and the background is white' and trains with reconstruction + ARAP
"""
# mesh guidance,final loss,process_image,mv_reference
import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from pathlib import Path
import torch
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
import imageio
from PIL import Image
import torch.nn.functional as F
from glob import glob

# Import TRELLIS pipeline
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from trellis.representations import Gaussian
from trellis.utils import render_utils

# Import FLUX Kontext for reference image generation
from diffusers import FluxKontextPipeline

# Import SDS guidance (Stable Diffusion XL)
from final_sds import create_sds_guidance

# Import centralized loss computation
from final_loss import LossComputer
# Import Image Processing utilities
from process_image import ImageProcessingMixin

# Add VGG-T to path for depth estimation
vggt_path = Path("/home/zxu298/3DTownHQ/vggt")
sys.path.insert(0, str(vggt_path))

# Import VGG-T for depth estimation
from vggt.models.vggt import VGGT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# ==================== ARAP Utilities ====================
# NOTE: ARAP utilities are now imported from mesh_guidance.py
# This provides:
# - Mesh-based topology extraction
# - ARAP loss computation with mesh connectivity
# - Gaussian alignment optimization
from mesh_guidance import (
    cal_arap_error,
    cal_connectivity_from_points,
    produce_edge_matrix_nfmt,
    estimate_rotation
)


# NOTE: ARAP functions (estimate_rotation, cal_arap_error) are now imported from mesh_guidance.py
# No duplicate definitions needed here
# ==================== End of ARAP Utilities ====================


class ManSceneReconARAPTrainer(ImageProcessingMixin):
    def __init__(self, output_dir: str = "man_recon_arap_output", object1: str = "man", object2: str = "chair",
                 scene_prompt: str = None,
                 object1_only_prompt: str = None,
                 sds_negative_prompt: str = 'ugly, blurry, low quality, deformed, noisy, artifacts',
                 target_prompt: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store object names
        self.object1_name = object1
        self.object2_name = object2

        # Initialize TRELLIS pipeline
        logger.info("Loading TRELLIS text-to-3D pipeline...")
        self.text_pipeline = TrellisTextTo3DPipeline.from_pretrained("TRELLIS-text-xlarge")
        self.text_pipeline.to(self.device)
        logger.info("TRELLIS text-to-3D pipeline loaded!")

        logger.info("Loading TRELLIS image-to-3D pipeline...")
        self.image_pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.image_pipeline.to(self.device)
        logger.info("TRELLIS image-to-3D pipeline loaded!")

        # Scene configuration - using provided object names
        self.objects = [
            {
                'id': 0,
                'prompt': f'a {object1}',
                'translation': np.array([0.0, 0.0, 0.0]),  # Centered
                'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
                'scale': 1.0
            },
            {
                'id': 1,
                'prompt': f'a {object2}',
                'translation': np.array([0.0, 0.0, 0.0]),  # Behind the first object
                'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
                'scale':0.8
            }
        ]

        # Prompts for SDS and reference image generation (from args)
        self.scene_prompt = scene_prompt
        self.object1_only_prompt = object1_only_prompt
        self.sds_negative_prompt = sds_negative_prompt
        self.target_prompt = target_prompt

        # SDS configuration for Phase 3
        self.sds_model_type = 'sdxl'  # 'sdxl' (Stable Diffusion XL)
        self.sds_guidance = None  # Will be initialized in initialize_sds_guidance()
        self.weight_sds = 0.001  # SDS loss weight (Phase 3)
        self.training_steps = 25000  # Total training steps (Phase 1: 8000, Phase 2: 2000, Phase 3: 15000)
        self.learning_rate_pose = 0.003  # Learning rate for pose (reduced for stable convergence)
        self.learning_rate_gaussian = 0.001  # Learning rate for gaussian params

        # Loss weights (following Animate3d's configuration)
        self.weight_arap = 1.0  # ARAP loss weight
        self.weight_rgb = 1.0  # RGB reconstruction weight (Animate3d: lambda_rgb = 1.0)
        self.weight_mask = 1.0  # Mask reconstruction weight (1:1 ratio with RGB)

        # Scale normalization for reconstruction loss
        self.use_scale_normalization = False  # Normalize foreground scale before comparing

        # Reference images configuration
        self.reference_images_dir = None  # Will be set if using reconstruction loss
        self.use_reconstruction_loss = False
        self.reference_views = []  # List of dicts with 'image', 'mask', 'azimuth', 'elevation', 'radius'
        self.multiview_images = []  # List of multiview images for Phase 3 only (will be loaded separately)
        self.bg_color = (1.0, 1.0, 1.0)  # White background

        # FLUX Kontext configuration for reference image generation
        # Local FLUX model path
        self.flux_model_path = "/data4/zishuo/FLUX.1-Kontext-dev"

        # Initialize centralized loss computer (handles LPIPS, IoU, ARAP, reconstruction loss)
        self.loss_computer = LossComputer(
            device=self.device,
            bg_color=list(self.bg_color),
            use_scale_normalization=self.use_scale_normalization
        )

        self.flux_pipe = None  # FLUX Kontext pipeline (lazy loading)
        self.vggt_model = None  # VGG-T model for depth estimation (lazy loading)
        self.weight_depth = 1.0  # Weight for VGG-T depth loss

        # Reference view angle (from TRELLIS generation, 150 degrees = 180 - 30)
        # Matching test_flux_edit_2.py rendering parameters
        self.reference_azimuth = 135.0  # degrees (TRELLIS coordinate system)
        self.reference_elevation = 0.0  # degrees
        self.reference_radius = 1.2  # Camera distance (unified to 0.8 for closer view)
        self.reference_fov = 60.0  # Field of view (matching test_flux_edit_2.py)

        # SAM2 and Grounding DINO for segmentation (lazy loading)
        self.sam2_predictor = None
        self.grounding_model = None
        self.ground_dino_processor = None

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

    def load_sam2(self):
        """Load SAM2 model for image segmentation"""
        if self.sam2_predictor is None:
            logger.info("Loading SAM2 model...")
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = "facebook/sam2.1-hiera-large"
            self.sam2_predictor = SAM2ImagePredictor.from_pretrained(sam2_model, device=self.device)
            logger.info("SAM2 model loaded!")
        return self.sam2_predictor

    def load_grounding_dino(self):
        """Load Grounding DINO model for object detection"""
        if self.grounding_model is None or self.ground_dino_processor is None:
            logger.info("Loading Grounding DINO model...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            ground_dino_model_path = "/data4/zishuo/grounding-dino-base"
            self.ground_dino_processor = AutoProcessor.from_pretrained(ground_dino_model_path)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(ground_dino_model_path).to(self.device)
            logger.info("Grounding DINO model loaded!")
        return self.grounding_model, self.ground_dino_processor

    def segment_reference_with_sam2(self, reference_image: Image.Image, tags: list):
        """Segment reference image into multiple objects using Grounding DINO + SAM2

        Args:
            reference_image: PIL Image to segment
            tags: List of text prompts for objects to detect (e.g., ["boy", "chair"])

        Returns:
            masks: Dictionary mapping tag -> binary mask (numpy array, [H, W])
        """
        logger.info("=== Segmenting reference image with SAM2 ===")

        # Lazy load models
        self.load_sam2()
        self.load_grounding_dino()

        # Convert PIL to numpy for SAM2
        image_np = np.array(reference_image.convert("RGB"))

        # Set image for SAM2
        self.sam2_predictor.set_image(image_np)

        masks_dict = {}

        for tag in tags:
            logger.info(f"Detecting and segmenting: {tag}")

            # Step 1: Use Grounding DINO to detect bounding boxes
            inputs = self.ground_dino_processor(
                images=reference_image,
                text=tag,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            # Post-process to get boxes
            results = self.ground_dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.25,
                target_sizes=[reference_image.size[::-1]]
            )

            if len(results[0]["boxes"]) == 0:
                logger.warning(f"No boxes detected for '{tag}', creating empty mask")
                masks_dict[tag] = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
                continue

            # Get detected boxes (use the first/highest score box)
            input_boxes = results[0]["boxes"].cpu().numpy()
            scores = results[0]["scores"].cpu().numpy()

            # Use the box with highest score
            best_idx = np.argmax(scores)
            best_box = input_boxes[best_idx:best_idx+1]

            logger.info(f"  Detected {len(input_boxes)} boxes for '{tag}', using best (score: {scores[best_idx]:.3f})")

            # Step 2: Use SAM2 to segment with the detected box
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, _sam_scores, _logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=best_box,
                    multimask_output=False,
                )

            # Get the mask (shape: [1, H, W])
            mask = masks[0]

            logger.info(f"  Segmentation complete for '{tag}' (coverage: {mask.mean()*100:.1f}%)")

            masks_dict[tag] = mask.astype(np.float32)

        return masks_dict

    def generate_reference_image_with_flux(self):
        """Generate reference image using FLUX Kontext with separately rendered objects

        Returns:
            reference_dir: Path to directory containing reference images
        """
        logger.info("=== Generating Reference Image with FLUX Kontext ===")

        reference_dir = self.output_dir / "reference_images"
        reference_dir.mkdir(exist_ok=True, parents=True)

        # Check if reference images already exist
        reference_path = reference_dir / "reference_edited.png"
        if reference_path.exists():
            logger.info(f"Found existing reference images in {reference_dir}, skipping generation")
            return str(reference_dir)

        # Camera parameters - SIDE VIEW
        azimuth = np.radians(self.reference_azimuth)
        elevation = 0.0
        radius = 1.5

        logger.info(f"Using SIDE VIEW: azimuth={np.degrees(azimuth):.1f}°, elevation={np.degrees(elevation):.1f}°")

        # Render objects separately (512x512 for FLUX input)
        logger.info(f"Rendering {self.object1_name} and {self.object2_name} separately...")
        rendered_images = []

        for obj in self.objects:
            logger.info(f"Rendering {obj['prompt']}...")

            # Apply scale if specified (for object2)
            gaussian_to_render = obj['gaussian']
            scale = obj.get('scale', 1.0)

            if scale != 1.0:
                logger.info(f"  Applying scale={scale} to {obj['prompt']} before rendering")
                original_xyz = gaussian_to_render._xyz.clone()

                # Get world coordinates, apply scale, set back
                xyz_world = gaussian_to_render.get_xyz
                xyz_world_scaled = xyz_world * scale
                gaussian_to_render.from_xyz(xyz_world_scaled)

            rendered = self.render_gaussian_view(
                gaussian_to_render,
                azimuth=azimuth,
                elevation=elevation,
                radius=radius,
                resolution=512,
                return_mask=False
            )

            # Restore original xyz if we modified it
            if scale != 1.0:
                gaussian_to_render._xyz = original_xyz

            # Handle RGBA output
            if rendered.shape[-1] == 4:
                rgb = rendered[..., :3]
                alpha = rendered[..., 3:4]
                rendered_rgb = rgb * alpha + 1.0 * (1 - alpha)
                rendered = rendered_rgb

            # Convert to PIL
            rendered_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
            rendered_pil = Image.fromarray(rendered_np)

            # Save individual render
            render_path = reference_dir / f"{obj['prompt'].replace(' ', '_')}_render.png"
            rendered_pil.save(render_path)
            logger.info(f"Saved {obj['prompt']} render to {render_path}")

            rendered_images.append(rendered_pil)

        # Concatenate images horizontally
        logger.info("Concatenating images horizontally for FLUX input...")
        obj1_img, obj2_img = rendered_images[0], rendered_images[1]

        total_width = obj1_img.width + obj2_img.width
        max_height = max(obj1_img.height, obj2_img.height)

        concatenated_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        y_offset_obj1 = (max_height - obj1_img.height) // 2
        y_offset_obj2 = (max_height - obj2_img.height) // 2
        concatenated_image.paste(obj1_img, (0, y_offset_obj1))
        concatenated_image.paste(obj2_img, (obj1_img.width, y_offset_obj2))

        # Save concatenated context image
        concat_path = reference_dir / "concatenated_context.png"
        concatenated_image.save(concat_path)
        logger.info(f"Saved concatenated context to {concat_path}")

        # Load FLUX Kontext
        logger.info("Loading FLUX Kontext pipeline...")
        flux_pipe = FluxKontextPipeline.from_pretrained(
            self.flux_model_path,
            torch_dtype=torch.bfloat16
        )
        flux_pipe.to(self.device)
        flux_pipe.enable_vae_slicing()
        flux_pipe.enable_vae_tiling()
        flux_pipe.enable_sequential_cpu_offload()
        logger.info("Pipeline loaded!")

        # Generate edited image
        logger.info(f"Generating composed image with prompt: '{self.target_prompt}'")

        with torch.no_grad():
            result = flux_pipe(
                prompt=self.target_prompt,
                image=concatenated_image,
                height=max_height,
                width=total_width,
                guidance_scale=2.5,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]

        # Save full FLUX output
        full_output_path = reference_dir / "flux_output_full.png"
        result.save(full_output_path)
        logger.info(f"Saved full FLUX output to {full_output_path}")

        # Crop left half as the final composed result
        result_width, result_height = result.size
        mid_x = result_width // 2
        left_half = result.crop((0, 0, mid_x, result_height))

        # Save reference image (left half)
        left_half.save(reference_path)
        logger.info(f"Saved reference image (left half) to {reference_path}")

        # Free FLUX pipeline before loading SAM2
        del flux_pipe
        torch.cuda.empty_cache()
        logger.info("FLUX pipeline freed")

        # Segment reference image with SAM2
        logger.info("Segmenting reference image with SAM2...")
        logger.info(f"Using SAM tags: object1='{self.object1_name}', object2='{self.object2_name}'")
        masks_dict = self.segment_reference_with_sam2(left_half, tags=[self.object1_name, self.object2_name])

        # Save segmented masks
        for tag, mask in masks_dict.items():
            mask_img = (mask * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_img, mode='L')
            mask_save_path = reference_dir / f"reference_edited_{tag}_mask.png"
            mask_pil.save(mask_save_path)
            logger.info(f"Saved {tag} mask to {mask_save_path}")

        # Create combined mask
        combined_mask = np.maximum(
            masks_dict.get(self.object1_name, np.zeros_like(list(masks_dict.values())[0])),
            masks_dict.get(self.object2_name, np.zeros_like(list(masks_dict.values())[0]))
        )
        combined_mask_img = (combined_mask * 255).astype(np.uint8)
        combined_mask_pil = Image.fromarray(combined_mask_img, mode='L')
        combined_mask_path = reference_dir / "reference_edited_mask.png"
        combined_mask_pil.save(combined_mask_path)
        logger.info(f"Saved combined mask to {combined_mask_path}")

        # Create separate RGB visualizations
        logger.info("Creating separate RGB visualizations...")
        result_np = np.array(left_half)
        obj1_mask = masks_dict.get(self.object1_name, np.zeros(result_np.shape[:2], dtype=np.float32))
        obj2_mask = masks_dict.get(self.object2_name, np.zeros(result_np.shape[:2], dtype=np.float32))

        # Object1 RGB only (white background)
        obj1_rgb = result_np.copy()
        obj1_mask_3ch = np.stack([obj1_mask, obj1_mask, obj1_mask], axis=-1)
        obj1_rgb = (obj1_rgb * obj1_mask_3ch + 255 * (1 - obj1_mask_3ch)).astype(np.uint8)
        obj1_rgb_pil = Image.fromarray(obj1_rgb)
        obj1_rgb_pil.save(reference_dir / f"reference_{self.object1_name}_only_rgb.png")
        logger.info(f"Saved {self.object1_name}-only RGB")

        # Object2 RGB only (white background)
        obj2_rgb = result_np.copy()
        obj2_mask_3ch = np.stack([obj2_mask, obj2_mask, obj2_mask], axis=-1)
        obj2_rgb = (obj2_rgb * obj2_mask_3ch + 255 * (1 - obj2_mask_3ch)).astype(np.uint8)
        obj2_rgb_pil = Image.fromarray(obj2_rgb)
        obj2_rgb_pil.save(reference_dir / f"reference_{self.object2_name}_only_rgb.png")
        logger.info(f"Saved {self.object2_name}-only RGB")

        logger.info(f"Reference image generation complete! Saved to {reference_dir}")
        return str(reference_dir)

    def load_reference_images(self, reference_dir, camera_params_list):
        """Load reference images from directory

        Args:
            reference_dir: Directory containing reference images
            camera_params_list: List of camera parameter dicts with 'azimuth', 'elevation', 'radius'

        Returns:
            reference_views: List of view dicts with image, mask, and camera params
        """
        logger.info(f"Loading reference images from {reference_dir}...")
        reference_dir = Path(reference_dir)

        reference_views = []

        for camera_params in camera_params_list:
            # Load reference image (combined scene)
            ref_image_path = reference_dir / "reference_edited.png"
            if not ref_image_path.exists():
                logger.warning(f"Reference image not found: {ref_image_path}")
                continue

            ref_image = Image.open(ref_image_path)
            ref_image_tensor = torch.from_numpy(np.array(ref_image)).float() / 255.0
            ref_image_tensor = ref_image_tensor.to(self.device)

            # Load combined mask
            ref_mask_path = reference_dir / "reference_edited_mask.png"
            ref_mask = Image.open(ref_mask_path)
            ref_mask_tensor = torch.from_numpy(np.array(ref_mask)).float() / 255.0
            ref_mask_tensor = ref_mask_tensor.unsqueeze(-1).to(self.device)

            # Load object1-only RGB
            obj1_rgb_path = reference_dir / f"reference_{self.object1_name}_only_rgb.png"
            obj1_rgb = Image.open(obj1_rgb_path)
            obj1_rgb_tensor = torch.from_numpy(np.array(obj1_rgb)).float() / 255.0
            obj1_rgb_tensor = obj1_rgb_tensor.to(self.device)

            # Load object1-only mask
            obj1_mask_path = reference_dir / f"reference_edited_{self.object1_name}_mask.png"
            obj1_mask = Image.open(obj1_mask_path)
            obj1_mask_tensor = torch.from_numpy(np.array(obj1_mask)).float() / 255.0
            obj1_mask_tensor = obj1_mask_tensor.unsqueeze(-1).to(self.device)

            # Load object2-only RGB
            obj2_rgb_path = reference_dir / f"reference_{self.object2_name}_only_rgb.png"
            obj2_rgb = Image.open(obj2_rgb_path)
            obj2_rgb_tensor = torch.from_numpy(np.array(obj2_rgb)).float() / 255.0
            obj2_rgb_tensor = obj2_rgb_tensor.to(self.device)

            # Load object2-only mask
            obj2_mask_path = reference_dir / f"reference_edited_{self.object2_name}_mask.png"
            obj2_mask = Image.open(obj2_mask_path)
            obj2_mask_tensor = torch.from_numpy(np.array(obj2_mask)).float() / 255.0
            obj2_mask_tensor = obj2_mask_tensor.unsqueeze(-1).to(self.device)

            # Store view with all components
            reference_views.append({
                'image': ref_image_tensor,
                'mask': ref_mask_tensor,
                f'{self.object1_name}_rgb': obj1_rgb_tensor,
                f'{self.object1_name}_mask': obj1_mask_tensor,
                f'{self.object2_name}_rgb': obj2_rgb_tensor,
                f'{self.object2_name}_mask': obj2_mask_tensor,
                'azimuth': camera_params['azimuth'],
                'elevation': camera_params['elevation'],
                'radius': camera_params['radius']
            })

            logger.info(f"Loaded reference view: azimuth={np.degrees(camera_params['azimuth']):.1f}°")

        logger.info(f"Loaded {len(reference_views)} reference view(s)")
        return reference_views

    def generate_objects(self, seed: int = 42):
        """Generate individual 3D objects using TRELLIS with mesh topology"""
        # Import mesh guidance module
        from mesh_guidance import convert_mesh_to_topology_gaussian

        logger.info("=== Generating 3D objects with mesh topology ===")

        self.generated_parts = []

        for obj in self.objects:
            logger.info(f"Generating: {obj['prompt']}")

            # Check if cached ply exists
            gaussian_path = self.output_dir / f"obj_{obj['id']:02d}_{obj['prompt'].replace(' ', '_')}_final.ply"
            mesh_path = self.output_dir / f"obj_{obj['id']:02d}_{obj['prompt'].replace(' ', '_')}_mesh.glb"

            if gaussian_path.exists() and mesh_path.exists():
                logger.info(f"Loading cached files from {gaussian_path}")
                # Create a Gaussian instance first, then load ply file
                # IMPORTANT: mesh_guidance uses sh_degree=0, not 3!
                gaussian_final = Gaussian(
                    aabb=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
                    sh_degree=0,  # mesh_guidance's quick_align_gaussian uses sh_degree=0
                    device='cuda'
                )
                # IMPORTANT: Use default transform to match save_ply's default transform
                gaussian_final.load_ply(str(gaussian_path))  # Use default transform, not None!

                # Load mesh
                import trimesh
                mesh_loaded = trimesh.load(str(mesh_path))

                # Handle Scene vs Trimesh object
                if isinstance(mesh_loaded, trimesh.Scene):
                    # GLB file contains a scene with multiple objects, extract first mesh
                    mesh_trimesh = list(mesh_loaded.geometry.values())[0]
                else:
                    mesh_trimesh = mesh_loaded

                # Simple mesh wrapper
                class SimpleMesh:
                    def __init__(self, vertices, faces):
                        self.vertices = torch.from_numpy(vertices).float()
                        self.faces = torch.from_numpy(faces).long()

                mesh_trellis = SimpleMesh(mesh_trimesh.vertices, mesh_trimesh.faces)
                gaussian_trellis = gaussian_final  # Keep the loaded gaussian as-is!

                # Extract connectivity from mesh topology (don't regenerate gaussian!)
                # Check if this is object1 (first object) by comparing with object1_name
                if self.object1_name.lower() in obj['prompt'].lower():
                    # Check for cached connectivity
                    connectivity_path = self.output_dir / f"obj_{obj['id']:02d}_{obj['prompt'].replace(' ', '_')}_connectivity.pkl"

                    if connectivity_path.exists():
                        logger.info(f"  Loading cached connectivity from {connectivity_path}")
                        import pickle
                        with open(connectivity_path, 'rb') as f:
                            connectivity = pickle.load(f)
                        logger.info(f"  Connectivity loaded: {len(connectivity)} vertices")
                    else:
                        logger.info("  Extracting connectivity from mesh topology (this may take a while for large meshes)...")
                        from mesh_guidance import MeshToGaussianConverter
                        converter = MeshToGaussianConverter(device=str(self.device))
                        # Only extract connectivity, don't regenerate gaussian
                        connectivity = converter.extract_mesh_connectivity(
                            vertices=mesh_trellis.vertices.to(self.device),
                            faces=mesh_trellis.faces.to(self.device)
                        )
                        logger.info(f"  Connectivity extracted: {len(connectivity)} vertices")

                        # Cache connectivity for next time
                        logger.info(f"  Saving connectivity to {connectivity_path}")
                        import pickle
                        with open(connectivity_path, 'wb') as f:
                            pickle.dump(connectivity, f)

                    # Keep using the loaded gaussian (gaussian_final)
                else:
                    connectivity = None

            else:
                logger.info("Generating from scratch...")

                # Determine object name from prompt
                object_name = None
                if obj['id'] == 0:
                    object_name = self.object1_name
                elif obj['id'] == 1:
                    object_name = self.object2_name

                # Try to load image from human_data or object_data directory
                use_image_to_3d = False
                image = None
                if object_name:
                    # Determine data directory based on object ID
                    if obj['id'] == 0:
                        # Object 1 (human) from human_data/
                        data_dir = Path(__file__).parent / "human_data"
                    else:
                        # Object 2 (object) from object_data/
                        data_dir = Path(__file__).parent / "object_data"

                    # Try multiple image formats
                    for ext in ['.png', '.jpg', '.jpeg']:
                        image_path = data_dir / f"{object_name}{ext}"
                        if image_path.exists():
                            logger.info(f"Found image at {image_path}, using image-to-3D pipeline")
                            image = Image.open(image_path)
                            use_image_to_3d = True
                            break

                if use_image_to_3d and image is not None:
                    # Use image-to-3D pipeline
                    out = self.image_pipeline.run(
                        image=image,
                        seed=seed + obj['id'],
                        formats=["gaussian", "mesh"],
                        sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
                        slat_sampler_params={"steps": 25, "cfg_strength": 7.5},
                    )
                else:
                    # Use text-to-3D for objects without images
                    logger.info(f"No image found in {'human_data/' if obj['id'] == 0 else 'object_data/'} for '{object_name}', using text-to-3D pipeline")
                    out = self.text_pipeline.run(
                        prompt=obj['prompt'],
                        seed=seed + obj['id'],
                        formats=["gaussian", "mesh"],
                        sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
                        slat_sampler_params={"steps": 25, "cfg_strength": 7.5},
                    )

                # Extract TRELLIS outputs
                gaussian_trellis = out["gaussian"][0]
                mesh_trellis = out["mesh"][0]

                logger.info(f"  TRELLIS generated: {len(gaussian_trellis._xyz)} Gaussians, "
                           f"{len(mesh_trellis.vertices) if hasattr(mesh_trellis, 'vertices') else 'N/A'} vertices")

                # Apply mesh guidance only for the first object (e.g., man), scale second object (e.g., chair)
                if obj['id'] == 0:
                    logger.info(f"  Converting mesh to topology-aware Gaussian (for {self.object1_name})...")
                    target_device = str(self.device)
                    gaussian_final, connectivity = convert_mesh_to_topology_gaussian(
                        mesh_trellis=mesh_trellis,
                        gaussian_trellis=gaussian_trellis,
                        device=target_device,
                        optimize=True,
                        num_steps=10000,
                        num_views=64,
                        resolution=512,
                        scale_factor=1.0/1.1,
                        render_video=True,
                        video_output_path=str(self.output_dir / f"{self.object1_name}_mesh_guidance.mp4")
                    )

                    logger.info(f"  Final Gaussian: {len(gaussian_final._xyz)} points with topology")
                    logger.info(f"  Topology edges: {sum(len(v) for v in connectivity.values()) // 2}")
                else:
                    logger.info(f"  Using original TRELLIS Gaussian (no mesh guidance for {self.object2_name})...")
                    logger.info(f"  Note: scale={obj['scale']} will be applied during FLUX rendering, not in 3D generation")
                    # Clone the gaussian without scaling (scale will be applied during rendering)
                    gaussian_final = Gaussian(
                        sh_degree=gaussian_trellis._sh_degree,
                        aabb=gaussian_trellis.aabb.clone(),
                        xyz=gaussian_trellis._xyz.clone(),  # 不缩放，保持原始大小
                        features=gaussian_trellis._features.clone(),
                        scaling=gaussian_trellis._scaling.clone(),
                        rotation=gaussian_trellis._rotation.clone(),
                        opacity=gaussian_trellis._opacity.clone()
                    )
                    connectivity = None
                    logger.info(f"  Gaussian saved at original size: {len(gaussian_final._xyz)} points")

                # Save both versions
                gaussian_final.save_ply(str(gaussian_path))

            # Save original TRELLIS gaussian for reference
            gaussian_trellis_path = self.output_dir / f"obj_{obj['id']:02d}_{obj['prompt'].replace(' ', '_')}_trellis.ply"
            gaussian_trellis.save_ply(str(gaussian_trellis_path))

            # Save mesh (MeshExtractResult doesn't have save method, use trimesh)
            mesh_path = self.output_dir / f"obj_{obj['id']:02d}_{obj['prompt'].replace(' ', '_')}_mesh.glb"
            import trimesh
            mesh_to_save = trimesh.Trimesh(
                vertices=mesh_trellis.vertices.cpu().numpy(),
                faces=mesh_trellis.faces.cpu().numpy()
            )
            mesh_to_save.export(str(mesh_path))

            # Store in object dict
            obj['gaussian'] = gaussian_final              # ← Use this for training (has topology)
            obj['gaussian_trellis'] = gaussian_trellis    # Keep original for reference
            obj['mesh'] = mesh_trellis                    # Original mesh
            obj['connectivity'] = connectivity            # ← Mesh topology for ARAP
            obj['gaussian_path'] = str(gaussian_path)
            obj['mesh_path'] = str(mesh_path)

            self.generated_parts.append(obj)

            logger.info(f"Saved to {gaussian_path}")
            logger.info(f"  - Final Gaussian (with topology): {gaussian_path}")
            logger.info(f"  - Mesh: {mesh_path}")

        # Free TRELLIS pipelines memory before loading FLUX
        logger.info("Freeing TRELLIS pipelines to save GPU memory...")
        del self.text_pipeline
        del self.image_pipeline
        torch.cuda.empty_cache()

    def initialize_sds_guidance(self):
        """
        Initialize SDS guidance for Phase 3 training
        Uses Stable Diffusion XL
        """
        logger.info(f"=== Initializing SDS Guidance ({self.sds_model_type.upper()}) ===")

        # Create SDS guidance using final_sds.py
        guidance_scale = 7.5

        # Use consistent t_range for both models: (0.02, 0.50) -> t range [20, 500]
        t_range = (0.02, 0.50)

        self.sds_guidance = create_sds_guidance(
            device=self.device,
            model_type=self.sds_model_type,
            guidance_scale=guidance_scale,
            t_range=t_range,
            use_fp16=True
        )

        # Encode text prompts
        logger.info(f"Encoding prompts for SDS:")
        logger.info(f"  Positive: {self.scene_prompt}")
        logger.info(f"  Negative: {self.sds_negative_prompt}")

        self.sds_guidance.get_text_embeds(
            prompt=self.scene_prompt,
            negative_prompt=self.sds_negative_prompt
        )
        logger.info(f"  Model: {self.sds_model_type}")
        logger.info(f"  Guidance scale: {guidance_scale}")
        logger.info(f"  Weight: {self.weight_sds}")

    def build_camera_matrix(self, azimuth: float, elevation: float, radius: float) -> torch.Tensor:
        """
        Build camera pose matrix from spherical coordinates
        Similnar to GradeADreamer's orbit_camera function

        Args:
            azimuth: Camera azimuth in radians
            elevation: Camera elevation in radians
            radius: Camera distance from origin

        Returns:
            camera_matrix: [4, 4] camera pose matrix (camera-to-world transform)
        """
        # Convert spherical to Cartesian coordinates
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = -radius * np.sin(elevation)
        z = radius * np.cos(elevation) * np.cos(azimuth)

        campos = np.array([x, y, z], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Build look-at matrix (OpenGL convention)
        forward = target - campos
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Build rotation matrix (camera-to-world)
        rot_matrix = np.stack([right, up, -forward], axis=0)  # [3, 3]

        # Build 4x4 transform matrix
        camera_matrix = np.eye(4, dtype=np.float32)
        camera_matrix[:3, :3] = rot_matrix.T  # Transpose for column-major
        camera_matrix[:3, 3] = campos

        return torch.from_numpy(camera_matrix).to(self.device)

    def setup_trainable_parameters(self):
        """Setup trainable parameters

        优化策略：
        - Obj1: 优化 delta (xyz, rotation)，保持初始状态作为anchor
        - Obj2: 优化全局 pose 参数 (translation, quaternion, scale)，Gaussian 固定
        """
        logger.info("=== Setting up trainable parameters ===")

        # Store original gaussians (用于初始化和ARAP)
        self.original_gaussians = []

        # === Obj1: Gaussian parameters - delta optimization ===
        obj1_gaussian = self.objects[0]['gaussian']

        # 保存初始状态（固定，作为anchor）
        self.obj1_xyz_init = obj1_gaussian._xyz.clone().detach()
        self.obj1_rotation_init = obj1_gaussian._rotation.clone().detach()
        self.obj1_scaling_init = obj1_gaussian._scaling.clone().detach()

        self.obj1_delta_xyz = torch.zeros_like(self.obj1_xyz_init, requires_grad=True)
        self.obj1_delta_rotation = torch.zeros_like(self.obj1_rotation_init, requires_grad=True)

        # Fixed parameters (no gradient)
        self.obj1_opacity = obj1_gaussian._opacity.clone().detach()
        self.obj1_aabb = obj1_gaussian.aabb  # 保存 AABB 用于坐标转换

        if obj1_gaussian._features_dc is not None:
            self.obj1_features_dc = obj1_gaussian._features_dc.clone().detach()
            if obj1_gaussian._features_rest is not None:
                self.obj1_features_rest = obj1_gaussian._features_rest.clone().detach()
            else:
                self.obj1_features_rest = None
        else:
            self.obj1_features_dc = None
            self.obj1_features_rest = None

        self.original_gaussians.append(obj1_gaussian)

        # === Obj2: Pose parameters - trainable ===
        obj2_gaussian = self.objects[1]['gaussian']

        # Chair pose parameters (trainable in stage 1, frozen in stage 2)
        self.obj2_translation = torch.tensor(
            self.objects[1]['translation'],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        self.obj2_quaternion = torch.tensor(
            self.objects[1]['quaternion'],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        self.obj2_scale = torch.tensor(
            [self.objects[1]['scale']],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        # Store chair's original Gaussian (frozen)
        self.original_gaussians.append(obj2_gaussian)

        logger.info("Optimization setup:")
        logger.info(f"  {self.object1_name.capitalize()} ({self.obj1_xyz_init.shape[0]} Gaussians): delta_xyz, delta_rotation (trainable)")
        logger.info(f"  {self.object1_name.capitalize()}: scaling, colors, opacity (fixed)")
        logger.info(f"  {self.object2_name.capitalize()} ({obj2_gaussian._xyz.shape[0]} Gaussians): Gaussian fixed, pose trainable")
        logger.info(f"  {self.object2_name.capitalize()} initial pose: t={self.objects[1]['translation']}, q={self.objects[1]['quaternion']}, s={self.objects[1]['scale']}")

    def compose_gaussians_from_trainable_params(self) -> Gaussian:
        """Compose gaussians using trainable parameters

        Returns a Gaussian with:
        - Obj1: delta-based optimization (xyz, rotation), no global transform
        - Obj2: trainable pose (translation, quaternion, scale), fixed Gaussian
        """
        all_means = []
        all_quats = []
        all_scales = []
        all_opacities = []
        all_features = []

        # === Obj1 Gaussian (delta optimization) ===
        obj1_gaussian = self.original_gaussians[0]
        obj1_aabb = obj1_gaussian.aabb

        # Compute current state = init + delta
        obj1_xyz_current = self.obj1_xyz_init + self.obj1_delta_xyz  # [N, 3]
        obj1_rotation_current = self.obj1_rotation_init + self.obj1_delta_rotation  # [N, 4]

        # Use current parameters (already in AABB space)
        obj1_means_normalized = obj1_xyz_current
        obj1_means_world = obj1_means_normalized * obj1_aabb[3:] + obj1_aabb[:3]

        # Get rotations and scaling
        rots_bias = torch.zeros((4), device=obj1_means_world.device)
        rots_bias[0] = 1
        obj1_rotations_full = obj1_rotation_current + rots_bias[None, :]  # Add bias for full quaternion
        obj1_scalings = self.obj1_scaling_init  # [N, 3], fixed (not trainable)

        # Get opacity (fixed)
        obj1_opacities = self.obj1_opacity

        # Get SH features (fixed)
        if self.obj1_features_dc is not None:
            obj1_features = self.obj1_features_dc.clone()
            if self.obj1_features_rest is not None:
                obj1_features = torch.cat([obj1_features, self.obj1_features_rest], dim=2)
        else:
            obj1_features = obj1_gaussian.get_features.clone()

        # Collect man parameters (no global transform)
        all_means.append(obj1_means_world)
        all_quats.append(obj1_rotations_full - rots_bias[None, :])  # Back to internal format
        all_scales.append(obj1_scalings)
        all_opacities.append(obj1_opacities)
        all_features.append(obj1_features)

        # === Chair Gaussian (fixed Gaussian, trainable pose) ===
        obj2_gaussian = self.original_gaussians[1]

        # Get chair's original Gaussian parameters (frozen)
        chair_xyz_normalized = obj2_gaussian._xyz.clone().detach()
        chair_aabb = obj2_gaussian.aabb
        chair_xyz_local = chair_xyz_normalized * chair_aabb[3:] + chair_aabb[:3]
        N_chair = chair_xyz_local.shape[0]

        # Get chair's frozen parameters
        chair_rotations_full = obj2_gaussian._rotation.clone().detach() + rots_bias[None, :]
        chair_scalings = obj2_gaussian._scaling.clone().detach()
        chair_opacities = obj2_gaussian.get_opacity.clone().detach()

        # Get chair's SH features (frozen)
        if obj2_gaussian._features_dc is not None:
            chair_features = obj2_gaussian._features_dc.clone().detach()
            if obj2_gaussian._features_rest is not None:
                chair_features = torch.cat([chair_features, obj2_gaussian._features_rest], dim=2)
        else:
            chair_features = obj2_gaussian.get_features.clone().detach()

        # Apply trainable global transformation to chair
        # Normalize quaternion
        quaternion = self.obj2_quaternion / (torch.norm(self.obj2_quaternion) + 1e-8)
        translation = self.obj2_translation
        scale = torch.clamp(self.obj2_scale, min=0.5)  # Prevent shrinking too small

        # 1. Scale
        scaled_xyz = chair_xyz_local * scale

        # 2. Rotation
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        rot_matrix = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y])
        ])

        rotated_xyz = scaled_xyz @ rot_matrix.T

        # Rotate gaussian orientations
        transformed_rotations_full = self._multiply_quaternions(
            quaternion.unsqueeze(0).expand(N_chair, -1),
            chair_rotations_full
        )

        # 3. Translation
        final_chair_xyz = rotated_xyz + translation

        # Transform scaling
        transformed_chair_scalings = chair_scalings + torch.log(scale)

        # Collect chair parameters
        all_means.append(final_chair_xyz)
        all_quats.append(transformed_rotations_full - rots_bias[None, :])
        all_scales.append(transformed_chair_scalings)
        all_opacities.append(chair_opacities)
        all_features.append(chair_features)

        # === Concatenate obj1 + obj2 ===
        combined_means = torch.cat(all_means, dim=0)
        combined_quats = torch.cat(all_quats, dim=0)
        combined_scalings = torch.cat(all_scales, dim=0)
        combined_opacities = torch.cat(all_opacities, dim=0)
        combined_features = torch.cat(all_features, dim=0)

        # Create combined Gaussian object
        combined_gaussian = Gaussian(
            aabb=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            sh_degree=self.original_gaussians[0].sh_degree,
            device=combined_means.device
        )

        # Normalize coordinates to AABB
        combined_aabb = combined_gaussian.aabb
        combined_means_normalized = (combined_means - combined_aabb[:3]) / combined_aabb[3:]

        # Set parameters
        combined_gaussian._xyz = combined_means_normalized
        combined_gaussian._rotation = combined_quats
        combined_gaussian._scaling = combined_scalings
        combined_gaussian._opacity = combined_opacities

        # Set features
        if combined_features.dim() == 2:
            combined_features = combined_features.unsqueeze(1)

        if combined_features.shape[2] <= 3:
            combined_gaussian._features_dc = combined_features
            combined_gaussian._features_rest = None
        else:
            combined_gaussian._features_dc = combined_features[:, :, :3]
            combined_gaussian._features_rest = combined_features[:, :, 3:]

        return combined_gaussian

    def compose_obj1_gaussian_only(self) -> Gaussian:
        """Compose only man's gaussian (for Phase 1 training)

        Returns a Gaussian with ONLY obj1's trainable parameters (delta-based)
        """
        # === Obj1 Gaussian (delta optimization) ===
        obj1_gaussian = self.original_gaussians[0]
        obj1_aabb = obj1_gaussian.aabb

        # Compute current state = init + delta
        obj1_xyz_current = self.obj1_xyz_init + self.obj1_delta_xyz  # [N, 3]
        obj1_rotation_current = self.obj1_rotation_init + self.obj1_delta_rotation  # [N, 4]

        # Use current parameters (already in AABB space)
        obj1_means_normalized = obj1_xyz_current
        obj1_means_world = obj1_means_normalized * obj1_aabb[3:] + obj1_aabb[:3]

        # Get rotations and scaling
        rots_bias = torch.zeros((4), device=obj1_means_world.device)
        rots_bias[0] = 1
        obj1_rotations_full = obj1_rotation_current + rots_bias[None, :]  # Add bias for full quaternion
        obj1_scalings = self.obj1_scaling_init  # [N, 3], fixed (not trainable)

        # Get opacity (fixed)
        obj1_opacities = self.obj1_opacity

        # Get SH features (fixed)
        if self.obj1_features_dc is not None:
            obj1_features = self.obj1_features_dc.clone()
            if self.obj1_features_rest is not None:
                obj1_features = torch.cat([obj1_features, self.obj1_features_rest], dim=2)
        else:
            obj1_features = obj1_gaussian.get_features.clone()

        # Create man-only Gaussian object
        obj1_only_gaussian = Gaussian(
            aabb=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            sh_degree=self.original_gaussians[0].sh_degree,
            device=obj1_means_world.device
        )

        # Normalize coordinates to AABB
        obj1_aabb_normalized = obj1_only_gaussian.aabb
        obj1_means_normalized_final = (obj1_means_world - obj1_aabb_normalized[:3]) / obj1_aabb_normalized[3:]

        # Set parameters
        obj1_only_gaussian._xyz = obj1_means_normalized_final
        obj1_only_gaussian._rotation = obj1_rotations_full - rots_bias[None, :]  # Back to internal format
        obj1_only_gaussian._scaling = obj1_scalings
        obj1_only_gaussian._opacity = obj1_opacities

        # Set features
        if obj1_features.dim() == 2:
            obj1_features = obj1_features.unsqueeze(1)

        if obj1_features.shape[2] <= 3:
            obj1_only_gaussian._features_dc = obj1_features
            obj1_only_gaussian._features_rest = None
        else:
            obj1_only_gaussian._features_dc = obj1_features[:, :, :3]
            obj1_only_gaussian._features_rest = obj1_features[:, :, 3:]

        return obj1_only_gaussian

    def compose_obj2_gaussian_only(self) -> Gaussian:
        """Compose only chair's gaussian (for Phase 2 training)

        Returns a Gaussian with ONLY obj2's parameters (frozen Gaussian, trainable pose)
        """
        # === Chair Gaussian (fixed Gaussian, trainable pose) ===
        obj2_gaussian = self.original_gaussians[1]

        # Get chair's original Gaussian parameters (frozen)
        chair_xyz_normalized = obj2_gaussian._xyz.clone().detach()
        chair_aabb = obj2_gaussian.aabb
        chair_xyz_local = chair_xyz_normalized * chair_aabb[3:] + chair_aabb[:3]
        N_chair = chair_xyz_local.shape[0]

        # Get chair's frozen parameters
        rots_bias = torch.zeros((4), device=chair_xyz_local.device)
        rots_bias[0] = 1
        chair_rotations_full = obj2_gaussian._rotation.clone().detach() + rots_bias[None, :]
        chair_scalings = obj2_gaussian._scaling.clone().detach()
        chair_opacities = obj2_gaussian.get_opacity.clone().detach()

        # Get chair's SH features (frozen)
        if obj2_gaussian._features_dc is not None:
            chair_features = obj2_gaussian._features_dc.clone().detach()
            if obj2_gaussian._features_rest is not None:
                chair_features = torch.cat([chair_features, obj2_gaussian._features_rest], dim=2)
        else:
            chair_features = obj2_gaussian.get_features.clone().detach()

        # Apply trainable global transformation to chair
        # Normalize quaternion
        quaternion = self.obj2_quaternion / (torch.norm(self.obj2_quaternion) + 1e-8)
        translation = self.obj2_translation
        scale = torch.clamp(self.obj2_scale, min=0.5)  # Prevent shrinking too small

        # 1. Scale
        scaled_xyz = chair_xyz_local * scale

        # 2. Rotation
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        rot_matrix = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y])
        ])

        rotated_xyz = scaled_xyz @ rot_matrix.T

        # Rotate gaussian orientations
        transformed_rotations_full = self._multiply_quaternions(
            quaternion.unsqueeze(0).expand(N_chair, -1),
            chair_rotations_full
        )

        # 3. Translation
        final_chair_xyz = rotated_xyz + translation

        # Transform scaling
        transformed_chair_scalings = chair_scalings + torch.log(scale)

        # Create chair-only Gaussian object
        chair_only_gaussian = Gaussian(
            aabb=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            sh_degree=self.original_gaussians[1].sh_degree,
            device=final_chair_xyz.device
        )

        # Normalize coordinates to AABB
        chair_aabb_normalized = chair_only_gaussian.aabb
        chair_means_normalized = (final_chair_xyz - chair_aabb_normalized[:3]) / chair_aabb_normalized[3:]

        # Set parameters
        chair_only_gaussian._xyz = chair_means_normalized
        chair_only_gaussian._rotation = transformed_rotations_full - rots_bias[None, :]
        chair_only_gaussian._scaling = transformed_chair_scalings
        chair_only_gaussian._opacity = chair_opacities

        # Set features
        if chair_features.dim() == 2:
            chair_features = chair_features.unsqueeze(1)

        if chair_features.shape[2] <= 3:
            chair_only_gaussian._features_dc = chair_features
            chair_only_gaussian._features_rest = None
        else:
            chair_only_gaussian._features_dc = chair_features[:, :, :3]
            chair_only_gaussian._features_rest = chair_features[:, :, 3:]

        return chair_only_gaussian

    def _multiply_quaternions(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (w,x,y,z format)"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=-1)

    def render_gaussian_view(self, gaussian: Gaussian, azimuth: float, elevation: float,
                            radius: float, resolution: int = 720, return_mask: bool = False):
        """Render a single view of the gaussian with gradient support

        Args:
            gaussian: Gaussian object to render
            azimuth: Camera azimuth angle (radians)
            elevation: Camera elevation angle (radians)
            radius: Camera radius
            resolution: Rendering resolution (default 720 for training views)
            return_mask: If True, return (rgb, mask) tuple; otherwise return rgb only

        Returns:
            If return_mask=False: rendered_image [H, W, 3]
            If return_mask=True: (rendered_image [H, W, 3], rendered_mask [H, W, 1])
        """
        from trellis.renderers import GaussianRenderer
        from easydict import EasyDict as edict

        # Get extrinsics and intrinsics
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws=float(azimuth),
            pitchs=float(elevation),
            rs=float(radius),
            fovs=60.0
        )

        # Setup renderer
        renderer = GaussianRenderer()
        renderer.rendering_options = edict({
            'resolution': resolution,
            'near': 0.8,
            'far': 1.6,
            'bg_color': self.bg_color,  # White background (1.0, 1.0, 1.0)
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
            # This maintains gradient flow for mask loss
            bg_color = torch.tensor(self.bg_color, device=rendered_image.device).view(1, 1, 3)
            color_diff = torch.abs(rendered_image - bg_color).sum(dim=-1, keepdim=True)
            # Use sigmoid to create soft mask (differentiable):
            # sigmoid((color_diff - threshold) * sharpness)
            # sharpness=1000 makes it very steep (almost binary) but still differentiable
            threshold = 0.01
            sharpness = 1000.0  # Increased from 100 to make it more binary-like
            rendered_alpha = torch.sigmoid((color_diff - threshold) * sharpness)

        rendered_alpha = torch.clamp(rendered_alpha, 0.0, 1.0)

        if return_mask:
            return rendered_image, rendered_alpha
        else:
            # 返回 RGBA [H, W, 4]，方便后续预处理
            rendered_rgba = torch.cat([rendered_image, rendered_alpha], dim=-1)
            return rendered_rgba

    def save_multiview_images(self, recon_step_dir, angle_deg, rendered_rgb, gt_rgb, rendered_mask, gt_mask):
        """Save multiview rendered and GT images

        Args:
            recon_step_dir: Directory to save images
            angle_deg: Angle in degrees (MVAdapter coordinate system)
            rendered_rgb: Rendered RGB image [H, W, 3]
            gt_rgb: Ground truth RGB image [H, W, 3]
            rendered_mask: Rendered mask [H, W, 1]
            gt_mask: Ground truth mask [H, W, 1]
        """
        from PIL import Image

        angle_name = f"{int(angle_deg):03d}deg"

        # Save rendered RGB
        mv_rendered_np = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mv_rendered_np).save(recon_step_dir / f"mv_{angle_name}_rendered.png")

        # Save GT RGB
        mv_gt_np = (gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mv_gt_np).save(recon_step_dir / f"mv_{angle_name}_gt_rgb.png")

        # Save rendered mask
        mv_rendered_mask_np = (rendered_mask.detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mv_rendered_mask_np.squeeze(-1), mode='L').save(recon_step_dir / f"mv_{angle_name}_rendered_mask.png")

        # Save GT mask
        mv_gt_mask_np = (gt_mask.detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mv_gt_mask_np.squeeze(-1), mode='L').save(recon_step_dir / f"mv_{angle_name}_gt_mask.png")

    def load_phase2_checkpoint(self):
        """Load checkpoint from phase 2 end to skip to phase 3"""
        checkpoint_path = self.output_dir / "checkpoint_phase2_end.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load trainable parameters
        self.obj1_delta_xyz.data = checkpoint['obj1_delta_xyz'].to(self.device)
        self.obj1_delta_rotation.data = checkpoint['obj1_delta_rotation'].to(self.device)
        self.obj2_translation.data = checkpoint['obj2_translation'].to(self.device)
        self.obj2_quaternion.data = checkpoint['obj2_quaternion'].to(self.device)
        self.obj2_scale.data = checkpoint['obj2_scale'].to(self.device)

        logger.info(f"Loaded checkpoint from step {checkpoint['step']}")
        logger.info("Trainable parameters restored successfully")

        return checkpoint['step']

    def train_with_recon_arap(self, skip_to_phase3=False):
        """3-stage independent training with SAM2 segmentation

        Args:
            skip_to_phase3: If True, skip phase 1 and 2, load checkpoint and start from phase 3

        Phase 1 (0-7999): Train ONLY obj1's Gaussian, render ONLY obj1
          - Render: ONLY obj1's gaussian (no obj2)
          - Obj1: Gaussian params (xyz, rotation, scaling) - trainable
          - Loss: ARAP + Reconstruction (obj1 RGB + obj1_mask)
          - Obj2: NOT involved

        Phase 2 (8000-9999): Train ONLY obj2's pose, render ONLY obj2
          - Render: ONLY obj2's gaussian (no obj1)
          - Obj2: Pose params (translation, quaternion, scale) - trainable
          - Loss: IoU Loss + Reconstruction Loss (obj2 RGB + obj2_mask), ratio 1:1
          - Obj1: FROZEN

        Phase 3 (10000-24999): Fine-tune man's Gaussian (render obj1 + obj2 combined)
          - Render: obj1 + obj2 combined
          - Obj1: Gaussian params (xyz, rotation, scaling) - trainable
          - Loss: ARAP + Reconstruction (reference_edited RGB + reference_edited_mask)
          - Obj2: Pose FROZEN
        """
        if skip_to_phase3:
            logger.info("=== SKIPPING TO PHASE 3 ===")
            logger.info("Loading checkpoint from phase 2 end...")
            self.load_phase2_checkpoint()
            logger.info("Checkpoint loaded successfully")
        else:
            logger.info("=== Starting 3-Stage Independent Training ===")

        # Get number of gaussians
        n_man = self.original_gaussians[0]._xyz.shape[0]
        n_chair = self.original_gaussians[1]._xyz.shape[0]

        logger.info(f"{self.object1_name.capitalize()} gaussians: {n_man}")
        logger.info(f"{self.object2_name.capitalize()} gaussians: {n_chair}")
        logger.info("=" * 60)
        logger.info("PHASE 1 (0-7999): Train ONLY Obj1's Gaussian, Render ONLY Obj1 - Total 8000 steps")
        logger.info("  Render: ONLY obj1 (no obj2)")
        logger.info("  Prompt: Obj1-only (no obj2 mentioned)")
        logger.info(f"  {self.object1_name.capitalize()}: xyz, rotation, scaling (Gaussian params) - trainable")
        logger.info("  Stage 1 (0-3999): ARAP×1.0 + Reconstruction (ref@150° only)")
        logger.info("    Reconstruction: IoU×1.0 + RGB×1.0 + Mask×1.0")
        logger.info("    ARAP×1.0, NO SDS")
        logger.info("  Stage 2 (4000-7999): ARAP×2.0 + Reconstruction (ref@150° only)")
        logger.info("    Reconstruction: IoU×1.0 + RGB×1.0 + Mask×1.0")
        logger.info("    ARAP×2.0, NO SDS")
        logger.info("  Chair: NOT involved")
        logger.info("=" * 60)
        logger.info("PHASE 2 (8000-9999): Train ONLY Obj2's Pose, Render ONLY Obj2")
        logger.info("  Render: ONLY obj2 (no obj1)")
        logger.info("  Chair: translation, quaternion, scale (Pose params) - trainable")
        logger.info("    Loss: IoU Loss + Reconstruction Loss (obj2 RGB + obj2_mask)")
        logger.info("    Ratio: 1:1 (IoU : Reconstruction)")
        logger.info("  Man: FROZEN (keep Phase 1 result)")

        logger.info(f"Total training steps: {self.training_steps}")
        logger.info(f"Learning rate (gaussian): {self.learning_rate_gaussian}")
        logger.info(f"Learning rate (pose): {self.learning_rate_pose}")
        logger.info(f"Loss weights: ARAP={self.weight_arap}, RGB={self.weight_rgb}")
        if self.use_reconstruction_loss:
            logger.info(f"Number of reference views: {len(self.reference_views)}")

        # Determine training camera angle - use reference image angle if available
        if self.use_reconstruction_loss and len(self.reference_views) > 0:
            # Use the first reference view's camera parameters for training
            training_azimuth = self.reference_views[0]['azimuth']
            training_elevation = self.reference_views[0]['elevation']
            training_radius = self.reference_views[0]['radius']
            logger.info(f"Training camera angle from reference image: azimuth={np.degrees(training_azimuth):.1f}°, elevation={np.degrees(training_elevation):.1f}°, radius={training_radius:.2f}")
        else:
            # Default: 135 degree side view (reference azimuth)
            training_azimuth = np.radians(self.reference_azimuth)  # 135 degrees
            training_elevation = 0.0
            training_radius = 1.2
            logger.info(f"Training camera angle (default): azimuth={np.degrees(training_azimuth):.1f}°, elevation={np.degrees(training_elevation):.1f}°, radius={training_radius:.2f}")

        # Phase 1: 4000 steps (0-3999)
        # Stage 1 (0-1999): ARAP×2.0 + RGB + Mask IOU + Depth
        # Stage 2 (2000-3999): ARAP×4.0 + RGB + Mask IOU + Depth
        phase1_stage2_start = 2000  # When to increase ARAP weight
        phase1_end = 4000  # End of Phase 1 (man training)
        stage1_end = phase1_end  # For backward compatibility
        stage2_end = 6000  # End of Phase 2 (chair training)

        # Determine starting step and setup optimizer
        if skip_to_phase3:
            start_step = stage2_end  # Start from 6000 (Phase 3)
            # Setup optimizer for Phase 3 - man's delta parameters (lr=1e-4)
            optimizer = torch.optim.Adam([
                {'params': [self.obj1_delta_xyz, self.obj1_delta_rotation], 'lr': 1e-4},
            ], betas=(0.9, 0.99), eps=1e-15)
            # Initialize SDS guidance immediately
            self.initialize_sds_guidance()
            logger.info("=== Starting from Phase 3 (step 6000) ===")
        else:
            start_step = 0  # Normal training from beginning
            # Setup optimizer for Phase 1 - man's delta parameters (xyz and rotation only)
            optimizer = torch.optim.Adam([
                {'params': [self.obj1_delta_xyz, self.obj1_delta_rotation], 'lr': self.learning_rate_gaussian},
            ], betas=(0.9, 0.99), eps=1e-15)

        # Training loop
        for step in range(start_step, self.training_steps):
            # === Phase 1 Stage Transition ===
            if step == phase1_stage2_start:
                logger.info("=" * 60)
                logger.info(f"=== PHASE 1 STAGE 2 (step {step}) ===")
                logger.info("Increasing ARAP weight: 2.0 → 4.0")
                logger.info("=" * 60)

            # === Phase 2 Transition: Switch optimizer to chair parameters ===
            if step == phase1_end:
                logger.info("=" * 60)
                logger.info(f"=== PHASE 2 START (step {step}) ===")
                logger.info("Switching optimizer to chair pose parameters...")
                logger.info("=" * 60)
                optimizer = torch.optim.Adam([
                    {'params': [self.obj2_translation, self.obj2_quaternion, self.obj2_scale], 'lr': self.learning_rate_pose},
                ], betas=(0.9, 0.99), eps=1e-15)

            # === Phase 3 Transition: Switch optimizer back to man parameters ===
            if step == stage2_end:
                logger.info("=" * 60)
                logger.info(f"=== PHASE 3 START (step {step}) ===")
                logger.info("Switching optimizer back to man delta parameters...")
                logger.info("Initializing SDS guidance...")
                logger.info("=" * 60)

                # Save combined gaussian (man + chair) and checkpoint before starting Phase 3
                with torch.no_grad():
                    combined_gaussian_phase2 = self.compose_gaussians_from_trainable_params()
                    combined_ply_path = self.output_dir / "combined_phase2_end.ply"
                    combined_gaussian_phase2.save_ply(str(combined_ply_path))
                    logger.info(f"Saved combined gaussian (phase 2 end) to: {combined_ply_path}")

                    # Save checkpoint with all trainable parameters
                    checkpoint_path = self.output_dir / "checkpoint_phase2_end.pt"
                    checkpoint = {
                        'obj1_delta_xyz': self.obj1_delta_xyz.detach().cpu(),
                        'obj1_delta_rotation': self.obj1_delta_rotation.detach().cpu(),
                        'obj2_translation': self.obj2_translation.detach().cpu(),
                        'obj2_quaternion': self.obj2_quaternion.detach().cpu(),
                        'obj2_scale': self.obj2_scale.detach().cpu(),
                        'step': step,
                    }
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint (phase 2 end) to: {checkpoint_path}")

                # Phase 3: Use smaller learning rate (1e-4) for fine-tuning
                optimizer = torch.optim.Adam([
                    {'params': [self.obj1_delta_xyz, self.obj1_delta_rotation], 'lr': 1e-4},
                ], betas=(0.9, 0.99), eps=1e-15)
                # Initialize SDS guidance for Phase 3
                if self.sds_guidance is None:
                    self.initialize_sds_guidance()

            optimizer.zero_grad()

            # Compose gaussians based on current phase
            if step < phase1_end:
                # Phase 1: Render ONLY obj1's gaussian
                current_gaussian = self.compose_obj1_gaussian_only()
            elif step < stage2_end:
                # Phase 2: Render ONLY obj2's gaussian
                current_gaussian = self.compose_obj2_gaussian_only()
            else:
                # Phase 3: Render combined obj1 + obj2
                current_gaussian = self.compose_gaussians_from_trainable_params()

            # Compute ARAP loss (for both stages with different weights)
            obj1_gaussian = self.original_gaussians[0]
            loss_arap, self._arap_connectivity_cache = self.loss_computer.compute_arap_loss(
                self.obj1_xyz_init,
                self.obj1_delta_xyz,
                obj1_gaussian.aabb,
                self.generated_parts,
                connectivity_cache=getattr(self, '_arap_connectivity_cache', None)
            )

            # ARAP weight: 2.0 for Stage 1 (0-1999), 4.0 for Stage 2 (2000-3999)
            weight_arap_effective = 2.0 if step < phase1_stage2_start else 4.0

            # Optional: Delta regularization (prevent excessive deformation)
            # Penalize large delta values to keep deformation reasonable
            loss_delta_reg = (self.obj1_delta_xyz.norm(dim=-1).mean() +
                             0.1 * self.obj1_delta_rotation.norm(dim=-1).mean())
            weight_delta_reg = 0.001  # Small weight, just to prevent explosion

            # === Phase-specific Loss Computation ===

            # === PHASE 1: Single-view training with Depth supervision ===
            # Stage 1 (0-1999): ARAP×2.0 + RGB + Mask IOU + Depth
            # Stage 2 (2000-3999): ARAP×4.0 + RGB + Mask IOU + Depth
            if step < phase1_end:
                # Save images every 200 steps
                save_images = (step % 200 == 0)
                recon_step_dir = None
                if save_images:
                    stage_name = "Stage1" if step < phase1_stage2_start else "Stage2"
                    logger.info(f"Step {step} (Phase1 {stage_name}): Computing losses")
                    recon_step_dir = self.output_dir / f"reconstruction_step_{step:04d}_phase1"
                    recon_step_dir.mkdir(exist_ok=True, parents=True)

                # Initialize reconstruction losses
                loss_rgb_recon = torch.tensor(0.0, device=self.device)
                loss_iou = torch.tensor(0.0, device=self.device)
                loss_depth = torch.tensor(0.0, device=self.device)

                # Compute reconstruction loss ONLY for reference view (single-view training)
                if self.use_reconstruction_loss and len(self.reference_views) > 0:
                    ref_view = self.reference_views[0]

                    # Render the view from current gaussian (man only)
                    rendered_rgb_full, rendered_mask_recon = self.render_gaussian_view(
                        current_gaussian,
                        azimuth=ref_view['azimuth'],
                        elevation=ref_view['elevation'],
                        radius=ref_view['radius'],
                        return_mask=True
                    )

                    # PHASE 1: Use man-only GT
                    rendered_rgb_recon = rendered_rgb_full
                    gt_rgb = ref_view[f'{self.object1_name}_rgb']
                    gt_mask = ref_view[f'{self.object1_name}_mask']

                    # Resize if needed
                    if gt_rgb.shape[:2] != rendered_rgb_recon.shape[:2]:
                        gt_rgb = F.interpolate(
                            gt_rgb.permute(2, 0, 1).unsqueeze(0),
                            size=rendered_rgb_recon.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).permute(1, 2, 0)

                        gt_mask = F.interpolate(
                            gt_mask.permute(2, 0, 1).unsqueeze(0),
                            size=rendered_mask_recon.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).permute(1, 2, 0)

                    # 1. RGB Loss (MSE instead of L1+LPIPS)
                    loss_rgb_recon = F.mse_loss(rendered_rgb_recon, gt_rgb)

                    # 2. Mask IOU Loss (ENABLED for Phase 1)
                    intersection = (rendered_mask_recon * gt_mask).sum()
                    union = (rendered_mask_recon + gt_mask - rendered_mask_recon * gt_mask).sum()
                    iou = intersection / (union + 1e-6)
                    loss_iou = 1.0 - iou

                    # 3. VGG-T Depth Loss (NEW!)
                    # Precompute GT depth once at step 0
                    if step == 0:
                        logger.info("Precomputing GT depth with VGG-T...")
                        self.gt_depth = self.predict_depth_vggt(gt_rgb, enable_grad=False)
                        logger.info(f"GT depth range: [{self.gt_depth.min():.4f}, {self.gt_depth.max():.4f}]")

                    # Predict depth for rendered image (with gradient!)
                    rendered_depth = self.predict_depth_vggt(rendered_rgb_recon, enable_grad=True)

                    # Resize GT depth if needed
                    gt_depth = self.gt_depth
                    if gt_depth.shape != rendered_depth.shape:
                        gt_depth = F.interpolate(
                            gt_depth.unsqueeze(0).unsqueeze(0),
                            size=rendered_depth.shape,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()

                    loss_depth = F.mse_loss(rendered_depth, gt_depth)

                    # Save images (only every 200 steps)
                    if save_images and recon_step_dir is not None:
                        # Save rendered RGB
                        rendered_full_np = (rendered_rgb_full.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(rendered_full_np).save(recon_step_dir / "reference_view_rendered.png")

                        # Save GT RGB
                        gt_np = (gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(gt_np).save(recon_step_dir / "reference_view_gt_rgb.png")

                        # Save rendered mask
                        rendered_mask_np = (rendered_mask_recon.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(rendered_mask_np.squeeze(-1), mode='L').save(recon_step_dir / "reference_view_rendered_mask.png")

                        # Save GT mask
                        gt_mask_np = (gt_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(gt_mask_np.squeeze(-1), mode='L').save(recon_step_dir / "reference_view_gt_mask.png")

                        # Save rendered depth (normalized for visualization)
                        depth_vis = rendered_depth.detach().cpu().numpy()
                        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-6)
                        depth_vis = (depth_vis * 255).astype(np.uint8)
                        Image.fromarray(depth_vis, mode='L').save(recon_step_dir / "depth_render.png")

                        # Save GT depth (only once at step 0)
                        if step == 0:
                            gt_depth_vis = self.gt_depth.detach().cpu().numpy()
                            gt_depth_vis = (gt_depth_vis - gt_depth_vis.min()) / (gt_depth_vis.max() - gt_depth_vis.min() + 1e-6)
                            gt_depth_vis = (gt_depth_vis * 255).astype(np.uint8)
                            Image.fromarray(gt_depth_vis, mode='L').save(recon_step_dir / "depth_gt.png")

                # Total loss: ARAP + RGB + Mask IOU + Depth
                # All weights set to 1.0 (following batch_recon3.py)
                loss = (weight_arap_effective * loss_arap +
                        self.weight_rgb * loss_rgb_recon +
                        self.weight_mask * loss_iou +
                        self.weight_depth * loss_depth)

                # Log every 200 steps
                if step % 200 == 0:
                    logger.info(f"Phase 1 Step {step}: Loss={loss.item():.6f}, "
                                f"RGB={loss_rgb_recon.item():.6f}, IOU={iou.item():.4f}, "
                                f"Depth={loss_depth.item():.6f}, ARAP={loss_arap.item():.6f}")

            # === PHASE 2: Simple MSE loss (chair only) ===
            elif step < stage2_end:
                # Get GT (chair only from reference)
                gt_rgb = self.reference_views[0][f'{self.object2_name}_rgb']
                gt_mask = self.reference_views[0][f'{self.object2_name}_mask']

                # Render chair (background is already white globally)
                rendered_rgb, rendered_mask = self.render_gaussian_view(
                    current_gaussian,
                    azimuth=training_azimuth,
                    elevation=training_elevation,
                    radius=training_radius,
                    return_mask=True
                )

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
                        size=rendered_mask.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)

                # Phase 2: IoU Loss + Reconstruction Loss (RGB + Mask supervision)

                # 1. IoU Loss - spatial alignment for mask
                iou = self.loss_computer.compute_iou(rendered_mask, gt_mask)
                loss_iou = 1.0 - iou

                # 2. Reconstruction Loss - RGB and mask supervision
                loss_rgb_recon, loss_mask_recon = self.loss_computer.compute_reconstruction_loss(
                    rendered_rgb,
                    rendered_mask,
                    gt_rgb,
                    gt_mask,
                    use_mse=True  # Phase 2 uses MSE for RGB
                )

                # Combined loss with 1:1 ratio
                # IOU loss for spatial alignment + Reconstruction loss for RGB/mask quality
                weight_iou = 1.0      # IoU loss weight
                weight_recon = 1.0    # Reconstruction loss weight (RGB + Mask combined)

                # Total reconstruction loss combines RGB and mask
                loss_recon_total = self.weight_rgb * loss_rgb_recon + self.weight_mask * loss_mask_recon

                loss = weight_iou * loss_iou + weight_recon * loss_recon_total

                # Save images every 200 steps
                if step % 200 == 0:
                    recon_step_dir = self.output_dir / f"reconstruction_step_{step:04d}_phase2"
                    recon_step_dir.mkdir(exist_ok=True, parents=True)

                    # Save rendered RGB
                    rendered_np = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(rendered_np).save(recon_step_dir / "rendered_chair_rgb.png")

                    # Save GT RGB
                    gt_np = (gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(gt_np).save(recon_step_dir / "gt_chair_rgb.png")

                    # Save rendered mask
                    mask_np = (rendered_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_np.squeeze(-1), mode='L').save(recon_step_dir / "rendered_chair_mask.png")

                    # Save GT mask
                    gt_mask_np = (gt_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(gt_mask_np.squeeze(-1), mode='L').save(recon_step_dir / "gt_chair_mask.png")

            # === PHASE 3: MVAdapter SDS + ARAP (combined scene) ===
            else:
                # Initialize SDS guidance at the start of Phase 3 (step == stage2_end == 10000)
                if step == stage2_end and self.sds_guidance is None:
                    logger.info("=" * 60)
                    logger.info(f"=== PHASE 3 START (step {step}) ===")
                    logger.info("Initializing SDS guidance now...")
                    logger.info("=" * 60)
                    self.initialize_sds_guidance()

                # Reconstruction loss weight for Phase 3 (constant throughout)
                weight_recon_multiplier = 1.0  # All weights set to 1.0

                # Randomly sample 4 views from multiview images to save GPU memory
                num_multiviews = len(self.multiview_images)

                # Debug: Print multiview status
                if step == stage2_end + 1 or (num_multiviews == 0 and step % 100 == 0):
                    logger.warning(f"DEBUG: self.multiview_images length = {num_multiviews}")
                    if num_multiviews > 0:
                        sample_angles = [mv['angle_name'] for mv in self.multiview_images[:3]]
                        logger.info(f"  Sample angles loaded: {sample_angles}")

                if num_multiviews > 4:
                    # Randomly select 4 indices
                    selected_indices = torch.randperm(num_multiviews)[:4].tolist()
                else:
                    # Use all views if less than or equal to 4
                    selected_indices = list(range(num_multiviews))

                # Save images every 200 steps
                save_images = (step % 200 == 0)
                recon_step_dir = None
                if save_images:
                    logger.info(f"Step {step} (phase3 - Combined): Computing SDS and reconstruction losses")
                    logger.info(f"  Selected {len(selected_indices)} views from {num_multiviews} total: {selected_indices}")
                    recon_step_dir = self.output_dir / f"reconstruction_step_{step:04d}_phase3"
                    recon_step_dir.mkdir(exist_ok=True, parents=True)


                # Compute reconstruction loss for reference + multiview images
                # Reference image weight = 1.0, multiview weights = 1/num_multiviews
                loss_rgb_recon = torch.tensor(0.0, device=self.device)
                loss_mask_recon = torch.tensor(0.0, device=self.device)
                loss_iou = torch.tensor(0.0, device=self.device)

                if self.use_reconstruction_loss and len(self.reference_views) > 0:
                    # === 1. Reference image reconstruction (weight = 1.0) ===
                    ref_view = self.reference_views[0]

                    # Render the view from current gaussian (combined man+chair)
                    rendered_rgb_full, rendered_mask_recon = self.render_gaussian_view(
                        current_gaussian,
                        azimuth=ref_view['azimuth'],
                        elevation=ref_view['elevation'],
                        radius=ref_view['radius'],
                        return_mask=True
                    )

                    # PHASE 3: Use full rendered RGB directly (no SAM2 segmentation)
                    rendered_rgb_recon = rendered_rgb_full

                    # GT RGB (combined scene with gray background)
                    gt_rgb = ref_view['image']

                    # Resize GT RGB if needed BEFORE computing mask
                    if gt_rgb.shape[:2] != rendered_rgb_recon.shape[:2]:
                        gt_rgb = F.interpolate(
                            gt_rgb.permute(2, 0, 1).unsqueeze(0),
                            size=rendered_rgb_recon.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).permute(1, 2, 0)

                    # Compute GT mask from white background RGB (background=0, foreground=1)
                    gt_mask = self.compute_mask_from_gray_bg_rgb(
                        gt_rgb,
                        bg_color=self.bg_color,  # (1.0, 1.0, 1.0)
                        threshold=0.05
                    )

                    # Compute IoU loss for spatial alignment
                    iou = self.loss_computer.compute_iou(rendered_mask_recon, gt_mask)
                    loss_iou = 1.0 - iou

                    # Compute reconstruction loss (RGB + Mask)
                    ref_loss_rgb, ref_loss_mask = self.loss_computer.compute_reconstruction_loss(
                        rendered_rgb_recon,
                        rendered_mask_recon,
                        gt_rgb,
                        gt_mask,
                        use_mse=False
                    )

                    # Apply reference weight (1.0) for both RGB and Mask
                    loss_rgb_recon = 1.0 * ref_loss_rgb
                    loss_mask_recon = 1.0 * ref_loss_mask

                    # Save reference view reconstruction images (only every 200 steps)
                    if save_images and recon_step_dir is not None:
                        # Save rendered full RGB (obj1 + obj2)
                        rendered_full_np = (rendered_rgb_full.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(rendered_full_np).save(recon_step_dir / "reference_view_rendered_full.png")

                        # Save rendered RGB (for reconstruction)
                        rendered_recon_np = (rendered_rgb_recon.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(rendered_recon_np).save(recon_step_dir / "reference_view_rendered_recon.png")

                        # Save GT RGB (combined scene, gray background)
                        gt_np = (gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(gt_np).save(recon_step_dir / "reference_view_gt_rgb.png")

                        # Save rendered mask (computed from rendered RGB, bg=0/black, fg=1/white)
                        rendered_mask_np = (rendered_mask_recon.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(rendered_mask_np.squeeze(-1), mode='L').save(recon_step_dir / "reference_view_rendered_mask.png")

                        # Save GT mask (computed from GT RGB with gray bg, bg=0/black, fg=1/white)
                        gt_mask_np = (gt_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(gt_mask_np.squeeze(-1), mode='L').save(recon_step_dir / "reference_view_gt_mask.png")

                    # === 2. Multiview reconstruction (weight = 1/num_selected for each) ===
                    if len(selected_indices) > 0:
                        num_selected = len(selected_indices)
                        mv_weight = 1.0 / num_selected

                        for mv_idx in selected_indices:
                            mv_view = self.multiview_images[mv_idx]
                            # Render from multiview angle
                            mv_rendered_rgb_full, mv_rendered_mask = self.render_gaussian_view(
                                current_gaussian,
                                azimuth=mv_view['azimuth'],
                                elevation=mv_view['elevation'],
                                radius=mv_view['radius'],
                                return_mask=True
                            )

                            # PHASE 3: Use full rendered RGB directly (no SAM2 segmentation)
                            mv_rendered_rgb = mv_rendered_rgb_full

                            # GT RGB (already has gray background from loading)
                            mv_gt_rgb = mv_view['image']

                            # Resize GT RGB if needed BEFORE computing mask
                            if mv_gt_rgb.shape[:2] != mv_rendered_rgb.shape[:2]:
                                mv_gt_rgb = F.interpolate(
                                    mv_gt_rgb.permute(2, 0, 1).unsqueeze(0),
                                    size=mv_rendered_rgb.shape[:2],
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0).permute(1, 2, 0)

                            # Compute GT mask from white background RGB (background=0, foreground=1)
                            mv_gt_mask = self.compute_mask_from_gray_bg_rgb(
                                mv_gt_rgb,
                                bg_color=self.bg_color,  # (1.0, 1.0, 1.0)
                                threshold=0.05
                            )

                            # Compute multiview reconstruction loss (RGB only, no mask loss)
                            mv_loss_rgb = self.loss_computer.compute_reconstruction_loss_rgb_only(
                                mv_rendered_rgb,
                                mv_gt_rgb,
                                use_mse=False
                            )

                            # Add weighted multiview loss (RGB only)
                            loss_rgb_recon = loss_rgb_recon + mv_weight * mv_loss_rgb

                            # Save multiview reconstruction images (only every 200 steps)
                            if save_images and recon_step_dir is not None:
                                angle_name = mv_view['angle_name']
                                # Save rendered full RGB (obj1 + obj2)
                                mv_rendered_full_np = (mv_rendered_rgb_full.detach().cpu().numpy() * 255).astype(np.uint8)
                                Image.fromarray(mv_rendered_full_np).save(recon_step_dir / f"mv_{angle_name}_rendered_full.png")

                                # Save rendered obj1-only RGB (gray background)
                                mv_rendered_obj1_np = (mv_rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                                Image.fromarray(mv_rendered_obj1_np).save(recon_step_dir / f"mv_{angle_name}_rendered_{self.object1_name}_only.png")

                                # Save GT RGB (obj1 only, gray background)
                                mv_gt_np = (mv_gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                                Image.fromarray(mv_gt_np).save(recon_step_dir / f"mv_{angle_name}_gt_rgb.png")

                                # Save rendered mask (computed from rendered RGB, bg=0/black, fg=1/white)
                                mv_rendered_mask_np = (mv_rendered_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                                Image.fromarray(mv_rendered_mask_np.squeeze(-1), mode='L').save(recon_step_dir / f"mv_{angle_name}_rendered_mask.png")

                                # Save GT mask (computed from GT RGB with gray bg, bg=0/black, fg=1/white)
                                mv_gt_mask_np = (mv_gt_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                                Image.fromarray(mv_gt_mask_np.squeeze(-1), mode='L').save(recon_step_dir / f"mv_{angle_name}_gt_mask.png")

                # === Compute SDS Loss ===
                loss_sds = torch.tensor(0.0, device=self.device)

                # Calculate step_ratio for timestep annealing (following DreamGaussian)
                # Phase 3: steps 10000-25000, step_ratio goes from 0 -> 1
                phase3_start = stage2_end  # 10000
                phase3_total_steps = self.training_steps - phase3_start  # 15000
                step_ratio = (step - phase3_start) / phase3_total_steps  # 0.0 -> 1.0

                # SDXL: Render from random azimuth
                if self.sds_model_type == 'sdxl':
                    # SDXL: Render from random azimuth
                    azimuth = np.random.uniform(0, 2 * np.pi)
                    elevation = 0.0
                    radius = 1.5

                    # Render view
                    rendered_rgb, _ = self.render_gaussian_view(
                        current_gaussian,
                        azimuth=azimuth,
                        elevation=elevation,
                        radius=radius,
                        resolution=512,
                        return_mask=True
                    )

                    # Convert to [1, 3, H, W] format
                    if rendered_rgb.ndim == 3:  # [H, W, 3]
                        rendered_rgb = rendered_rgb.permute(2, 0, 1).unsqueeze(0)

                    # Validate rendered RGB before SDS - FAIL FAST
                    if torch.isnan(rendered_rgb).any():
                        raise ValueError(f"NaN detected in rendered_rgb before SDXL SDS! "
                                       f"shape: {rendered_rgb.shape}, min: {rendered_rgb.min()}, max: {rendered_rgb.max()}")

                    if torch.isinf(rendered_rgb).any():
                        raise ValueError(f"Inf detected in rendered_rgb before SDXL SDS!")

                    if rendered_rgb.min() < -0.1 or rendered_rgb.max() > 1.1:
                        raise ValueError(f"rendered_rgb values outside valid range: "
                                       f"min={rendered_rgb.min()}, max={rendered_rgb.max()}")

                    # Clamp to [0, 1] for safety
                    rendered_rgb = torch.clamp(rendered_rgb, 0.0, 1.0)

                    # Compute SDS loss with timestep annealing (and optionally get SDS images for saving)
                    if save_images:
                        loss_sds, sds_images, timestep = self.sds_guidance.train_step(rendered_rgb, step, azimuth=azimuth, return_sds_images=True, step_ratio=step_ratio)
                    else:
                        loss_sds = self.sds_guidance.train_step(rendered_rgb, step, azimuth=azimuth, step_ratio=step_ratio)

                    # Save SDXL rendered view and SDS predicted image
                    if save_images and recon_step_dir is not None:
                        # Convert azimuth to degrees for filename
                        azimuth_deg = int(np.degrees(azimuth))

                        # Save rendered RGB [1, 3, H, W] -> [H, W, 3]
                        rendered_rgb_hwc = rendered_rgb[0].permute(1, 2, 0)  # [H, W, 3]
                        rendered_np = (rendered_rgb_hwc.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(rendered_np).save(recon_step_dir / f"sdxl_az{azimuth_deg:03d}_rendered.png")

                        # Save SDS predicted RGB with timestep in filename [1, 3, H, W] -> [H, W, 3]
                        sds_rgb_hwc = sds_images[0].permute(1, 2, 0)  # [H, W, 3]
                        sds_np = (sds_rgb_hwc.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(sds_np).save(recon_step_dir / f"sdxl_az{azimuth_deg:03d}_t{int(timestep):04d}_sds_target.png")

                # Combined loss: ARAP + SDS only (reconstruction loss commented out)
                # Note: Reconstruction loss (RGB + Mask + IoU) is temporarily disabled
                # weight_mask_phase3 = 1.0
                # loss_recon_total = self.weight_rgb * loss_rgb_recon + weight_mask_phase3 * loss_mask_recon
                # loss = (weight_arap_effective * loss_arap +
                #        weight_recon_multiplier * (loss_iou + loss_recon_total) +
                #        self.weight_sds * loss_sds)

                # Simplified loss: only ARAP + SDS
                loss = weight_arap_effective * loss_arap + self.weight_sds * loss_sds

            # Backpropagate
            loss.backward()

            # Clip gradients based on phase
            if step < stage1_end:
                # Phase 1: Clip only man's delta parameters (xyz and rotation)
                torch.nn.utils.clip_grad_norm_(
                    [self.obj1_delta_xyz, self.obj1_delta_rotation],
                    max_norm=1.0
                )
            elif step < stage2_end:
                # Phase 2: Clip only chair's pose parameters
                torch.nn.utils.clip_grad_norm_(
                    [self.obj2_translation, self.obj2_quaternion, self.obj2_scale],
                    max_norm=1.0
                )
            else:
                # Phase 3: Clip only man's delta parameters (xyz and rotation)
                torch.nn.utils.clip_grad_norm_(
                    [self.obj1_delta_xyz, self.obj1_delta_rotation],
                    max_norm=1.0
                )

            optimizer.step()

            # Check parameter health after update - FAIL FAST
            if step >= stage2_end:  # Only check in Phase 3 (SDS phase)
                # Check delta parameters
                if torch.isnan(self.obj1_delta_xyz).any():
                    raise ValueError(f"NaN detected in obj1_delta_xyz at step {step}")
                if torch.isnan(self.obj1_delta_rotation).any():
                    raise ValueError(f"NaN detected in obj1_delta_rotation at step {step}")

                # Check if delta parameters are exploding
                max_delta_xyz = self.obj1_delta_xyz.abs().max().item()
                if max_delta_xyz > 10.0:
                    logger.warning(f"Large delta_xyz detected at step {step}: {max_delta_xyz:.4f}")

            # Logging - every 10 steps
            if step % 10 == 0:
                if step < phase1_end:
                    # Phase 1: Man Only
                    # Determine if using multiview
                    num_mv = len(self.obj1_multiview_images) if hasattr(self, 'obj1_multiview_images') else 0
                    mv_suffix = f"ref@150° + {num_mv} multiview" if num_mv > 0 else "ref@150° only"

                    if step < phase1_stage2_start:
                        stage_name = "Phase 1 Stage 1 (Man Only)"
                        phase_status = f"Recon×2.0 ({mv_suffix}), NO SDS, NO ARAP"
                    else:
                        stage_name = "Phase 1 Stage 2 (Man Only)"
                        phase_status = f"ARAP×{weight_arap_effective:.1f} + Recon×2.0 ({mv_suffix}), NO SDS"

                    logger.info(f"Step {step}/{self.training_steps} ({stage_name}) - {phase_status}")
                    logger.info(f"  Total Loss: {loss.item():.4f}")

                    # Phase 1 Stage 1 has NO SDS, NO ARAP
                    # Phase 1 Stage 2 has ARAP, NO SDS
                    if step >= phase1_stage2_start:
                        logger.info(f"  ARAP: {loss_arap.item():.4f} (weight: {weight_arap_effective:.1f}, weighted: {(weight_arap_effective * loss_arap).item():.4f})")

                    # Show delta statistics every 100 steps
                    if step % 100 == 0:
                        delta_xyz_norm = self.obj1_delta_xyz.norm(dim=-1).mean().item()
                        delta_xyz_max = self.obj1_delta_xyz.abs().max().item()
                        delta_rot_norm = self.obj1_delta_rotation.norm(dim=-1).mean().item()
                        logger.info(f"  Delta XYZ: mean_norm={delta_xyz_norm:.4f}, max={delta_xyz_max:.4f}")
                        logger.info(f"  Delta Rotation: mean_norm={delta_rot_norm:.4f}")

                    # Show reconstruction in both stages
                    if self.use_reconstruction_loss and len(self.reference_views) > 0 and step % 25 == 0:
                        # logger.info(f"  IoU: {loss_iou.item():.4f} (weight: 1.0, weighted: {(1.0 * loss_iou).item():.4f}) [{mv_suffix}]")  # Disabled
                        logger.info(f"  RGB Recon: {loss_rgb_recon.item():.4f} (weight: 1.0, weighted: {(1.0 * loss_rgb_recon).item():.4f}) [{mv_suffix}]")
                        # logger.info(f"  Mask Recon: {loss_mask_recon.item():.4f} (weight: 0.5, weighted: {(0.5 * loss_mask_recon).item():.4f}) [{mv_suffix}]")  # Disabled

                elif step < stage2_end:
                    stage_name = "Phase 2 (Chair IoU + Reconstruction)"
                    logger.info(f"Step {step}/{self.training_steps} ({stage_name})")
                    logger.info(f"  Total Loss: {loss.item():.4f}")
                    logger.info(f"  IoU: {iou.item():.4f}, IoU Loss: {loss_iou.item():.4f} (weighted: {(weight_iou * loss_iou).item():.4f})")
                    logger.info(f"  Reconstruction RGB: {loss_rgb_recon.item():.4f} (weighted: {(self.weight_rgb * loss_rgb_recon).item():.4f})")
                    logger.info(f"  Reconstruction Mask: {loss_mask_recon.item():.4f} (weighted: {(self.weight_mask * loss_mask_recon).item():.4f})")
                    logger.info(f"  Total Reconstruction: {loss_recon_total.item():.4f} (weighted: {(weight_recon * loss_recon_total).item():.4f})")

                    if step % 100 == 0:
                        logger.info(f"  {self.object2_name.capitalize()} pose: t={self.obj2_translation.detach().cpu().numpy()}, "
                                   f"s={self.obj2_scale.detach().cpu().item():.3f}")

                else:
                    stage_name = "Phase 3 (SDS + ARAP only)"
                    logger.info(f"Step {step}/{self.training_steps} ({stage_name})")
                    logger.info(f"  Total Loss: {loss.item():.4f}")
                    logger.info(f"  SDS: {loss_sds.item():.4f} (weighted: {(self.weight_sds * loss_sds).item():.4f})")
                    logger.info(f"  ARAP: {loss_arap.item():.4f} (weighted: {(weight_arap_effective * loss_arap).item():.4f})")

                    # Reconstruction loss logging disabled (loss commented out)
                    # if self.use_reconstruction_loss and len(self.reference_views) > 0 and step % 25 == 0:
                    #     num_selected = len(selected_indices)
                    #     num_total = len(self.multiview_images)
                    #     recon_info = f"ref(w=1.0) + {num_selected}/{num_total}×mv(w=1/{num_selected})" if num_selected > 0 else "ref only"
                    #     weighted_rgb = weight_recon_multiplier * self.weight_rgb * loss_rgb_recon
                    #     weighted_mask = weight_recon_multiplier * weight_mask_phase3 * loss_mask_recon
                    #     weighted_iou = weight_recon_multiplier * loss_iou
                    #     logger.info(f"  IoU: {loss_iou.item():.4f} (weighted: {weighted_iou.item():.4f})")
                    #     logger.info(f"  RGB Recon: {loss_rgb_recon.item():.4f} (weighted: {weighted_rgb.item():.4f}) [{recon_info}]")
                    #     logger.info(f"  Mask Recon: {loss_mask_recon.item():.4f} (weighted: {weighted_mask.item():.4f}) [ref only]")

        logger.info("Training complete!")

        # Return final composed gaussian
        with torch.no_grad():
            final_gaussian = self.compose_gaussians_from_trainable_params()
        return final_gaussian

    def save_results(self, trained_gaussian: Gaussian):
        """Save final results"""
        logger.info("=== Saving Results ===")

        # Save trained gaussian
        gaussian_path = self.output_dir / f"{self.object1_name}_recon_trained.ply"
        trained_gaussian.save_ply(str(gaussian_path))
        logger.info(f"Saved trained gaussian to {gaussian_path}")

        # Render final video
        logger.info("Rendering final video...")
        render_output = render_utils.render_video(
            trained_gaussian,
            resolution=1024,
            num_frames=120,
            r=3.0,
            bg_color=(1.0, 1.0, 1.0)
        )

        # Extract frames
        if isinstance(render_output, dict):
            frames = render_output.get("color", render_output.get("normal", None))
        else:
            frames = render_output

        if frames is not None:
            if isinstance(frames, list):
                frames_np = []
                for frame in frames:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    frames_np.append(frame)
                frames = np.stack(frames_np)
            elif isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()

            if frames.dtype in [np.float32, np.float64] or frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)

            video_path = self.output_dir / f"{self.object1_name}_recon.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            logger.info(f"Saved video to {video_path}")

        # Render and save multiview angles
        logger.info("Rendering multiview angles...")
        multiview_renders_dir = self.output_dir / "final_multiview_renders"
        multiview_renders_dir.mkdir(exist_ok=True, parents=True)

        # Define views to render: reference + multiview angles
        views_to_render = []

        # Add reference view (150°)
        if len(self.reference_views) > 0:
            ref_view = self.reference_views[0]
            views_to_render.append({
                'name': 'reference_150deg',
                'azimuth': ref_view['azimuth'],
                'elevation': ref_view['elevation'],
                'radius': ref_view['radius']
            })

        # Add multiview angles if available
        for mv_view in self.multiview_images:
            views_to_render.append({
                'name': f"multiview_{mv_view['angle_name']}deg",
                'azimuth': mv_view['azimuth'],
                'elevation': mv_view['elevation'],
                'radius': mv_view['radius']
            })

        # Render each view
        for view_info in views_to_render:
            logger.info(f"Rendering {view_info['name']} "
                       f"(azimuth={np.degrees(view_info['azimuth']):.1f}°, "
                       f"elevation={np.degrees(view_info['elevation']):.1f}°)...")

            # Render RGB and mask
            rendered_rgb, rendered_mask = self.render_gaussian_view(
                trained_gaussian,
                azimuth=view_info['azimuth'],
                elevation=view_info['elevation'],
                radius=view_info['radius'],
                resolution=720,
                return_mask=True
            )

            # Save RGB
            rgb_np = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            rgb_path = multiview_renders_dir / f"{view_info['name']}_rgb.png"
            Image.fromarray(rgb_np).save(rgb_path)

            # Save mask
            mask_np = (rendered_mask.detach().cpu().numpy() * 255).astype(np.uint8)
            mask_path = multiview_renders_dir / f"{view_info['name']}_mask.png"
            Image.fromarray(mask_np.squeeze(-1), mode='L').save(mask_path)

            logger.info(f"  Saved {view_info['name']} to {rgb_path.name} and {mask_path.name}")

        logger.info(f"Saved {len(views_to_render)} multiview renders to {multiview_renders_dir}")

    def run(self, reference_images_dir: str = None, camera_params: list = None, generate_reference: bool = True, skip_to_phase3: bool = False):
        """Execute the full pipeline

        Args:
            reference_images_dir: Optional directory containing reference images for reconstruction loss
            camera_params: Optional camera parameters for reference images
            generate_reference: If True, generate reference image using FLUX Kontext
            skip_to_phase3: If True, skip phase 1 and 2, load checkpoint and start from phase 3
        """
        # Auto-detect if we should skip to phase 3 based on existing files
        checkpoint_path = self.output_dir / "checkpoint_phase2_end.pt"
        if checkpoint_path.exists() and not skip_to_phase3:
            logger.info("=" * 60)
            logger.info("DETECTED: checkpoint_phase2_end.pt exists!")
            logger.info("Automatically enabling skip_to_phase3=True")
            logger.info("=" * 60)
            skip_to_phase3 = True

        if skip_to_phase3:
            logger.info("=== SKIP TO PHASE 3 MODE ===")
            logger.info("Will skip: reference generation, multiview generation")
            logger.info("Will load: existing objects, checkpoint from phase 2")

        logger.info("=== Man Scene with Reconstruction + ARAP Training ===")

        # Step 1: Generate objects (always needed to load existing gaussians)
        self.generate_objects()

        if not skip_to_phase3:
            # Step 2: Generate reference image with FLUX Kontext (if requested)
            # This generates the SIDE VIEW (150°)
            if generate_reference and reference_images_dir is None:
                reference_images_dir = self.generate_reference_image_with_flux()

            # Save reference_images_dir to instance variable
            if reference_images_dir is not None:
                self.reference_images_dir = reference_images_dir

            # Step 3: Load single reference image for reconstruction loss
            if reference_images_dir is not None:
                camera_params_list = [{
                    'azimuth': np.radians(self.reference_azimuth),
                    'elevation': np.radians(self.reference_elevation),
                    'radius': self.reference_radius
                }]
                self.reference_views = self.load_reference_images(
                    reference_images_dir,
                    camera_params_list
                )
                if len(self.reference_views) > 0:
                    logger.info(f"Loaded {len(self.reference_views)} reference view(s) for reconstruction loss")
                    self.use_reconstruction_loss = True
                else:
                    logger.warning("No reference views loaded")
            else:
                logger.info("No reference images directory provided, skipping reference view loading")

            # Step 4: Save initial gaussian
            initial_gaussian = self.objects[0]['gaussian']
            initial_path = self.output_dir / f"{self.object1_name}_initial.ply"
            initial_gaussian.save_ply(str(initial_path))
            logger.info(f"Saved initial gaussian to {initial_path}")
        else:
            logger.info("Skipping reference generation and initial gaussian save (skip_to_phase3=True)")

        # Step 5: Setup trainable parameters (always needed for loading checkpoint)
        self.setup_trainable_parameters()

        if not skip_to_phase3:
            # Step 6: SDS guidance will be initialized in Phase 3 (deferred loading)
            # Note: initialize_sds_guidance() will be called at step == 6000 in train_with_recon_arap()
            logger.info(f"=== SDS Guidance ({self.sds_model_type.upper()}) will be loaded at Phase 3 start (step 6000) ===")

            # Phase 1: Single-view training mode (no multiview)
            logger.info(f"=== Phase 1: Single-view training mode ===")
        else:
            logger.info("Skipping to Phase 3 (skip_to_phase3=True)")

        # Step 7: Train with 3-stage approach (Recon + ARAP + SDS)
        trained_gaussian = self.train_with_recon_arap(skip_to_phase3=skip_to_phase3)

        # Step 7: Save results
        self.save_results(trained_gaussian)

        logger.info(f"\n=== Complete! Results in {self.output_dir} ===")


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Man scene with Reconstruction + ARAP training (No DDS)')
    parser.add_argument('--object1', type=str, default='boy',
                       help='Name of first object (used for reading images from data/ and naming output)')
    parser.add_argument('--object2', type=str, default='chair',
                       help='Name of second object (used for reading images from data/ and naming output)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: mv_output/object1_object2)')
    parser.add_argument('--steps', type=int, default=25000,
                       help='Number of training steps (Phase 1: 8000 steps in 2 stages of 4000 each)')
    parser.add_argument('--lr_gaussian', type=float, default=0.001,
                       help='Learning rate for gaussian parameters')
    parser.add_argument('--arap_weight', type=float, default=100.0,
                       help='Weight for ARAP loss')
    parser.add_argument('--reference_images_dir', type=str, default=None,
                       help='Directory containing reference images for reconstruction loss')
    parser.add_argument('--rgb_weight', type=float, default=1.0,
                       help='Weight for RGB reconstruction loss')
    parser.add_argument('--mask_weight', type=float, default=1.0,
                       help='Weight for mask reconstruction loss (1:1 ratio with RGB)')
    parser.add_argument('--sds_weight', type=float, default=0.0001,
                       help='Weight for SDS loss (Phase 3)')
    parser.add_argument('--scene_prompt', type=str,
                       default="A young cartoon boy with short fluffy brown hair, large expressive eyes, and a big smile, "
                               "wearing a short-sleeved orange t-shirt, blue jeans, and brown shoes, sitting casually "
                               "on a simple modern gray chair. His hands are resting on the chair. "
                               "The background is white.",
                       help='Full scene prompt for SDS (Phase 3)')
    parser.add_argument('--object1_only_prompt', type=str,
                       default="A young cartoon boy with short fluffy brown hair, large expressive eyes, and a big smile, "
                               "wearing a short-sleeved orange t-shirt, blue jeans, and brown shoes, sitting casually. "
                               "His hands are resting. The background is white.",
                       help='Object1-only prompt (for Phase 1 SDS - without object2)')
    parser.add_argument('--sds_negative_prompt', type=str, default='ugly, blurry, low quality, deformed, noisy, artifacts',
                       help='Negative prompt for SDS guidance')
    parser.add_argument('--target_prompt', type=str,
                       default="Create a unified, cohesive image where the young man is "
                               "sitting naturally on the chair. Maintain the identity and characteristics of each subject while adjusting their "
                               "proportions, scale, and positioning. Do not change the color of the background.",
                       help='FLUX Kontext prompt for generating reference image')
    parser.add_argument('--sds_model_type', type=str, default='sdxl',
                       choices=['sdxl'],
                       help='SDS model type: sdxl (single-view)')
    parser.add_argument('--generate_reference', action='store_true', default=True,
                       help='Generate reference image using FLUX Kontext')
    parser.add_argument('--skip-to-phase3', action='store_true', default=False,
                       help='Skip Phase 1 and 2, load combined gaussian from combined_phase2_end.ply and start directly from Phase 3')
    args = parser.parse_args()

    # Set default output directory based on object names
    if args.output_dir is None:
        args.output_dir = f'mv_output/{args.object1}_{args.object2}'

    trainer = ManSceneReconARAPTrainer(
        output_dir=args.output_dir,
        object1=args.object1,
        object2=args.object2,
        scene_prompt=args.scene_prompt,
        object1_only_prompt=args.object1_only_prompt,
        sds_negative_prompt=args.sds_negative_prompt,
        target_prompt=args.target_prompt
    )
    trainer.training_steps = args.steps
    trainer.learning_rate_gaussian = args.lr_gaussian
    trainer.weight_arap = args.arap_weight
    trainer.weight_rgb = args.rgb_weight
    trainer.weight_mask = args.mask_weight
    trainer.weight_sds = args.sds_weight
    trainer.sds_model_type = args.sds_model_type
    trainer.run(reference_images_dir=args.reference_images_dir, generate_reference=args.generate_reference, skip_to_phase3=args.skip_to_phase3)


if __name__ == '__main__':
    main()