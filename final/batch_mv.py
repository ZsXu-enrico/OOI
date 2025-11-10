#!/usr/bin/env python3
"""
Batch Generate Multiview and 3D from FLUX-Edited Images
- Reads images from flux_edit folders
- Uses SAM2 + Grounding DINO to segment object2 from reference_edited.png
- Generates 3D using TRELLIS image-to-3D
- Renders multiview images
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
import argparse

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Add MV-Adapter to path
mvadapter_path = SCRIPT_DIR.parent / "MV-Adapter"
sys.path.insert(0, str(mvadapter_path))

# Import TRELLIS components
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian
from trellis.renderers import GaussianRenderer
from trellis.utils import render_utils
from easydict import EasyDict as edict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FluxEditMultiviewGenerator:
    def __init__(
        self,
        good_edit_dir: str = None,
        output_dir: str = None,
        object_data_dir: str = None,
        device: str = "cuda",
        num_views: int = 8,
        use_mvadapter: bool = True
    ):
        # 设置默认路径（相对于脚本目录）
        if good_edit_dir is None:
            good_edit_dir = str(SCRIPT_DIR / "flux_edit")
        if output_dir is None:
            output_dir = str(SCRIPT_DIR / "flux_edit_multiview")
        if object_data_dir is None:
            object_data_dir = str(SCRIPT_DIR / "object_data")

        self.good_edit_dir = Path(good_edit_dir)
        self.output_dir = Path(output_dir)
        self.object_data_dir = Path(object_data_dir)
        self.device = device
        self.num_views = num_views
        self.use_mvadapter = use_mvadapter

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Lazy loading
        self.sam2_predictor = None
        self.grounding_model = None
        self.ground_dino_processor = None
        self.pipeline = None
        self.renderer = None
        self.mvadapter_pipe = None

        # Load simplified humanoid keywords for SAM
        self.humanoid_keywords = self._load_humanoid_keywords()

        # Track SAM usage for periodic reload
        self.sam2_call_count = 0
        self.sam2_reload_interval = 50  # Reload SAM2 every 50 calls

        logger.info("Initializing FluxEditMultiviewGenerator...")
        logger.info(f"  Use MVAdapter: {use_mvadapter}")
        logger.info(f"  Loaded {len(self.humanoid_keywords)} humanoid keywords for SAM")

    def _load_humanoid_keywords(self):
        """Load simplified humanoid keywords from simple_humanoid_caption.txt"""
        humanoid_json_path = Path("/data4/zishuo/TRELLIS/reference_humanoids.json")
        simple_caption_path = Path("/data4/zishuo/TRELLIS/simple_humanoid_caption.txt")

        keywords_map = {}

        if not humanoid_json_path.exists() or not simple_caption_path.exists():
            logger.warning("Humanoid keyword files not found, will use default keywords")
            return keywords_map

        try:
            # Load humanoid list
            with open(humanoid_json_path, 'r') as f:
                humanoid_list = json.load(f)

            # Load simple captions
            with open(simple_caption_path, 'r') as f:
                simple_captions = [line.strip() for line in f if line.strip()]

            # Create mapping: folder_name -> simple_keyword
            for idx, humanoid in enumerate(humanoid_list):
                folder_name = humanoid['folder_name']
                if idx < len(simple_captions):
                    keywords_map[folder_name] = simple_captions[idx]

            logger.info(f"Loaded {len(keywords_map)} simplified humanoid keywords from simple_humanoid_caption.txt")

        except Exception as e:
            logger.warning(f"Failed to load humanoid keywords: {e}")

        return keywords_map

    def load_sam2(self):
        """Load SAM2 model for image segmentation"""
        logger.info("Loading SAM2 model...")
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_model = "facebook/sam2.1-hiera-large"
        sam2_predictor = SAM2ImagePredictor.from_pretrained(sam2_model, device=self.device)
        logger.info("SAM2 model loaded!")
        return sam2_predictor

    def load_grounding_dino(self):
        """Load Grounding DINO model for object detection"""
        logger.info("Loading Grounding DINO model...")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        # Use local model path (relative to SCRIPT_DIR)
        ground_dino_model_path = str(SCRIPT_DIR.parent / "pretrained_models" / "grounding-dino-base")
        ground_dino_processor = AutoProcessor.from_pretrained(ground_dino_model_path)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(ground_dino_model_path).to(self.device)
        logger.info("Grounding DINO model loaded from local path!")
        return grounding_model, ground_dino_processor

    def get_foreground_mask(self, image: Image.Image, white_threshold: int = 250):
        """Get foreground mask from image (non-white pixels)

        Args:
            image: PIL Image
            white_threshold: Threshold for detecting white background

        Returns:
            mask: Binary mask (numpy array, [H, W])
        """
        img_array = np.array(image.convert('RGB'))
        # Detect white background (255, 255, 255)
        is_white = (img_array[:, :, 0] > white_threshold) & \
                   (img_array[:, :, 1] > white_threshold) & \
                   (img_array[:, :, 2] > white_threshold)
        # Foreground: non-white pixels
        foreground_mask = (~is_white).astype(np.float32)
        return foreground_mask

    def segment_object_with_sam2(self, reference_image: Image.Image, tag: str, reference_box_size: float = None):
        """Segment object from reference image using Grounding DINO + SAM2

        Args:
            reference_image: PIL Image to segment
            tag: Text prompt for object to detect (e.g., "balloon")
            reference_box_size: Optional reference box size (for filtering). If provided,
                               will prefer boxes smaller than this size.

        Returns:
            tuple: (mask, score) where mask is Binary mask (numpy array, [H, W]) or None,
                   and score is the detection confidence score
        """
        logger.info(f"  Segmenting '{tag}' with SAM2...")

        # Periodic reload of SAM2 to prevent memory corruption
        self.sam2_call_count += 1
        if self.sam2_call_count >= self.sam2_reload_interval:
            logger.info(f"  Reloading SAM2 (call count: {self.sam2_call_count})...")
            if self.sam2_predictor is not None:
                del self.sam2_predictor
                self.sam2_predictor = None
            torch.cuda.empty_cache()
            self.sam2_call_count = 0

        # Lazy load models
        if self.sam2_predictor is None:
            self.sam2_predictor = self.load_sam2()
        if self.grounding_model is None or self.ground_dino_processor is None:
            self.grounding_model, self.ground_dino_processor = self.load_grounding_dino()

        # Convert PIL to numpy for SAM2
        image_np = np.array(reference_image.convert("RGB"))

        # Reset SAM2 internal state before setting new image
        try:
            self.sam2_predictor.reset_predictor()
        except:
            pass  # reset_predictor might not exist in all versions

        # Set image for SAM2 with error handling
        max_retries = 2
        for retry in range(max_retries):
            try:
                self.sam2_predictor.set_image(image_np)
                break  # Success
            except Exception as e:
                logger.warning(f"  SAM2 set_image failed (attempt {retry+1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    # Reload SAM2 and retry
                    logger.info(f"  Reloading SAM2 and retrying...")
                    del self.sam2_predictor
                    self.sam2_predictor = None
                    torch.cuda.empty_cache()
                    self.sam2_predictor = self.load_sam2()
                else:
                    logger.error(f"  SAM2 set_image failed after {max_retries} attempts, returning None")
                    return None, 0.0

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
            box_threshold=0.25,  # Lower threshold for better detection
            text_threshold=0.25,
            target_sizes=[reference_image.size[::-1]]  # (height, width)
        )

        if len(results[0]["boxes"]) == 0:
            logger.warning(f"  No boxes detected for '{tag}'")
            return None, 0.0

        # Get detected boxes
        input_boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()

        # Calculate box sizes (width * height)
        box_sizes = []
        for box in input_boxes:
            x1, y1, x2, y2 = box
            box_size = (x2 - x1) * (y2 - y1)
            box_sizes.append(box_size)
        box_sizes = np.array(box_sizes)

        # Select the best box
        if reference_box_size is not None and len(input_boxes) > 1:
            # Filter boxes smaller than reference box size
            smaller_boxes_mask = box_sizes < reference_box_size
            if smaller_boxes_mask.any():
                # Among smaller boxes, choose the one with highest score
                filtered_scores = scores.copy()
                filtered_scores[~smaller_boxes_mask] = -1  # Mark larger boxes
                best_idx = np.argmax(filtered_scores)
                logger.info(f"  Detected {len(input_boxes)} boxes, filtered to {smaller_boxes_mask.sum()} smaller boxes")
                logger.info(f"  Using box {best_idx} (score: {scores[best_idx]:.3f}, size: {box_sizes[best_idx]:.0f} vs reference: {reference_box_size:.0f})")
            else:
                # All boxes are larger, fall back to highest score
                best_idx = np.argmax(scores)
                logger.warning(f"  All {len(input_boxes)} boxes are larger than reference, using highest score")
                logger.info(f"  Using box {best_idx} (score: {scores[best_idx]:.3f}, size: {box_sizes[best_idx]:.0f} vs reference: {reference_box_size:.0f})")
        else:
            # No reference size, just use highest score
            best_idx = np.argmax(scores)
            logger.info(f"  Detected {len(input_boxes)} boxes for '{tag}', using best (score: {scores[best_idx]:.3f})")

        best_box = input_boxes[best_idx:best_idx+1]  # Keep shape [1, 4]
        best_score = scores[best_idx]

        # Step 2: Use SAM2 to segment with the detected box
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, _sam_scores, _logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=best_box,  # [1, 4] - xyxy format
                multimask_output=False,
            )

        # Get the mask (shape: [1, H, W])
        mask = masks[0]  # [H, W]

        logger.info(f"  Segmentation complete for '{tag}' (coverage: {mask.mean()*100:.1f}%)")

        return mask.astype(np.float32), float(best_score)

    def load_trellis_pipeline(self):
        """Load TRELLIS image-to-3D pipeline"""
        if self.pipeline is None:
            logger.info("Loading TRELLIS image-to-3D pipeline...")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-image-large"
            )
            self.pipeline.to(self.device)
            logger.info("Pipeline loaded!")

    def find_object_data_image(self, object_text: str):
        """Find matching PNG in object_data folder based on object_text
        Priority: 1) new/ 2) animal_object/object/ 3) object_data/

        Args:
            object_text: Object text description (e.g., "a soccer ball")

        Returns:
            PIL Image if found, None otherwise
        """
        # Extract keywords from object text (e.g., "a soccer ball" -> "soccer ball")
        words = object_text.lower().split()
        words = [w for w in words if w not in ['a', 'an', 'the']]
        keywords = ' '.join(words) if words else object_text.lower()

        # Define search directories in priority order
        search_dirs = [
            (self.object_data_new_dir, "new"),
            (self.object_data_animal_dir, "animal_object/object"),
            (self.object_data_dir, "object_data")
        ]

        # Try to find matching PNG in each directory
        for search_dir, dir_name in search_dirs:
            if not search_dir.exists():
                logger.debug(f"  {dir_name} directory not found: {search_dir}")
                continue

            for png_file in search_dir.glob("*.png"):
                file_stem = png_file.stem.lower()

                # Check bidirectional matching
                matched = False

                # Check if filename is in keywords
                if file_stem in keywords:
                    matched = True

                # Check if any keyword word is in filename or vice versa
                if not matched:
                    for kw in keywords.split():
                        if kw in file_stem or file_stem in kw:
                            matched = True
                            break

                if matched:
                    logger.info(f"  Found object_data image in {dir_name}: {png_file.name} for '{object_text}'")
                    return Image.open(png_file).convert('RGB')

        logger.info(f"  No matching object_data image found for '{object_text}' in any directory")
        return None

    def load_mvadapter_pipeline(self):
        """Load MVAdapter pipeline for multiview generation"""
        if self.mvadapter_pipe is None:
            logger.info("Loading MVAdapter pipeline...")
            from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
            from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
            from diffusers import AutoencoderKL, DDPMScheduler

            # Use local model paths (relative to SCRIPT_DIR)
            base_model = str(SCRIPT_DIR.parent / "pretrained_models" / "MVAdapter" / "stable-diffusion-xl-base-1.0")
            vae_model = str(SCRIPT_DIR.parent / "pretrained_models" / "MVAdapter" / "sdxl-vae-fp16-fix")
            adapter_path = str(SCRIPT_DIR.parent / "pretrained_models" / "MVAdapter" / "mv-adapter")

            # Prepare VAE
            vae = AutoencoderKL.from_pretrained(vae_model)

            # Load pipeline
            self.mvadapter_pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
                base_model,
                vae=vae,
                torch_dtype=torch.float16
            )

            # Setup scheduler
            self.mvadapter_pipe.scheduler = ShiftSNRScheduler.from_scheduler(
                self.mvadapter_pipe.scheduler,
                shift_mode="interpolated",
                shift_scale=8.0,
                scheduler_class=DDPMScheduler,
            )

            # Initialize and load adapter (6 views)
            self.mvadapter_pipe.init_custom_adapter(num_views=6)
            self.mvadapter_pipe.load_custom_adapter(
                adapter_path,
                weight_name="mvadapter_i2mv_sdxl.safetensors"
            )

            self.mvadapter_pipe.to(device=self.device, dtype=torch.float16)
            self.mvadapter_pipe.cond_encoder.to(device=self.device, dtype=torch.float16)
            self.mvadapter_pipe.enable_vae_slicing()
            logger.info("MVAdapter pipeline loaded!")

    def preprocess_image_for_mvadapter(self, image: Image.Image, height=768, width=768):
        """Preprocess image for MVAdapter (no resize, just gray background)"""
        # Convert to RGBA if needed
        if image.mode != "RGBA":
            img_array = np.array(image)
            # Detect white background (255, 255, 255)
            is_white = (img_array[:, :, 0] > 250) & (img_array[:, :, 1] > 250) & (img_array[:, :, 2] > 250)
            # Create alpha channel
            alpha = (~is_white).astype(np.uint8) * 255
            image = Image.fromarray(np.dstack([img_array, alpha]))

        # Resize image to target resolution (no cropping, no 90% scaling)
        image_resized = image.resize((width, height), Image.LANCZOS)
        image_array = np.array(image_resized).astype(np.float32) / 255.0

        # Composite with gray background (0.5)
        if image_array.shape[-1] == 4:
            # Has alpha channel
            image_final = image_array[:, :, :3] * image_array[:, :, 3:4] + (1 - image_array[:, :, 3:4]) * 0.5
        else:
            # No alpha, treat white as transparent
            img_rgb = image_array[:, :, :3]
            is_white = ((img_rgb[:, :, 0] > 0.98) & (img_rgb[:, :, 1] > 0.98) & (img_rgb[:, :, 2] > 0.98)).astype(np.float32)
            alpha_mask = (1.0 - is_white)[:, :, np.newaxis]
            image_final = img_rgb * alpha_mask + 0.5 * (1.0 - alpha_mask)

        image_final = (image_final * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(image_final)

    def generate_multiview_with_mvadapter(self, object2_image: Image.Image, object_prompt: str):
        """Generate multiview images using MVAdapter

        Args:
            object2_image: Segmented object2 image (white background)
            object_prompt: Text prompt for the object

        Returns:
            List of 6 PIL Images (multiview)
        """
        logger.info("  Generating multiview with MVAdapter...")

        # Load MVAdapter pipeline
        self.load_mvadapter_pipeline()

        # Preprocess image for MVAdapter
        ref_image_pil = self.preprocess_image_for_mvadapter(object2_image, height=768, width=768)

        # Define 6 target angles
        target_angles = [0, 45, 90, 180, 270, 315]
        num_views = len(target_angles)

        # Calculate MVAdapter camera angles (target_angle - 90)
        mvadapter_angles = [angle - 90 for angle in target_angles]

        # Generate camera conditioning
        from mvadapter.utils.mesh_utils import get_orthogonal_camera
        from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
        import math

        # Camera parameters
        reference_fov = 60.0
        reference_radius = 1.2

        fov_rad = math.radians(reference_fov)
        half_size = reference_radius * math.tan(fov_rad / 2)
        left, right, bottom, top = -half_size, half_size, -half_size, half_size

        cameras = get_orthogonal_camera(
            elevation_deg=[0] * num_views,
            distance=[reference_radius] * num_views,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            azimuth_deg=mvadapter_angles,
            device=self.device,
        )

        plucker_embeds = get_plucker_embeds_from_cameras_ortho(
            cameras.c2w, [1.1] * num_views, 768
        )
        control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

        # Generate with MVAdapter
        generator = torch.Generator(device=self.device).manual_seed(42)

        mvadapter_images = self.mvadapter_pipe(
            object_prompt,
            height=768,
            width=768,
            num_inference_steps=50,
            guidance_scale=3.0,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            reference_image=ref_image_pil,
            reference_conditioning_scale=1.0,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
            generator=generator,
        ).images

        logger.info(f"  Generated {len(mvadapter_images)} multiview images with MVAdapter")

        return mvadapter_images, target_angles

    def _setup_renderer(self):
        """Setup Gaussian renderer for multiview generation"""
        if self.renderer is None:
            self.renderer = GaussianRenderer()
            self.renderer.rendering_options = edict({
                'resolution': 512,
                'near': 0.8,
                'far': 1.6,
                'bg_color': torch.tensor([1.0, 1.0, 1.0]).to(self.device),
                'ssaa': 1
            })
            self.renderer.pipe = edict({
                'kernel_size': 0.1,
                'use_mip_gaussian': True,
                'convert_SHs_python': False,
                'compute_cov3D_python': False,
                'debug': False,
                'scale_modifier': 1.0
            })

    def render_multiview(
        self,
        gaussian: Gaussian,
        num_views: int = 8,
        elevation: float = 0.0,
        radius: float = 1.5
    ) -> list:
        """Render multiple views of a Gaussian

        Args:
            gaussian: Gaussian representation
            num_views: Number of views to render (evenly spaced around object)
            elevation: Elevation angle in radians
            radius: Camera distance

        Returns:
            List of PIL images
        """
        views = []
        azimuths = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

        for azimuth in azimuths:
            # Setup camera
            extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                yaws=azimuth,
                pitchs=elevation,
                rs=radius,
                fovs=60.0
            )

            # Render
            with torch.no_grad():
                result = self.renderer.render(gaussian, extrinsics, intrinsics)
                rendered_image = result['color'].permute(1, 2, 0)  # [H, W, 3]
                rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

            # Convert to PIL
            rendered_np = (rendered_image.cpu().numpy() * 255).astype(np.uint8)
            rendered_pil = Image.fromarray(rendered_np)
            views.append(rendered_pil)

        return views

    def process_folder(self, folder_path: Path):
        """Process a single folder from flux_edit

        Args:
            folder_path: Path to folder (e.g., flux_edit/021_humanoid_153)
        """
        folder_name = folder_path.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {folder_name}")
        logger.info(f"{'='*60}")

        # Check for required files
        reference_edited_path = folder_path / "reference_edited.png"
        metadata_path = folder_path / "metadata.json"

        if not reference_edited_path.exists():
            logger.warning(f"  reference_edited.png not found, skipping")
            return

        # Create output folder
        output_folder = self.output_dir / folder_name
        output_folder.mkdir(exist_ok=True, parents=True)

        # Check if already processed (check for mvadapter_full_reference instead)
        if self.use_mvadapter and (output_folder / "mvadapter_full_reference_grid.png").exists():
            logger.info(f"  Already processed, skipping")
            return

        # Copy humanoid_render.png, object_render.png, and reference_edited.png if they exist
        import shutil
        humanoid_render_src = folder_path / "humanoid_render.png"
        object_render_src = folder_path / "object_render.png"

        if humanoid_render_src.exists():
            humanoid_render_dst = output_folder / "humanoid_render.png"
            shutil.copy2(humanoid_render_src, humanoid_render_dst)
            logger.info(f"  Copied humanoid_render.png to output folder")

        if object_render_src.exists():
            object_render_dst = output_folder / "object_render.png"
            shutil.copy2(object_render_src, object_render_dst)
            logger.info(f"  Copied object_render.png to output folder")

        # Save reference_edited.png to output folder
        if reference_edited_path.exists():
            reference_edited_dst = output_folder / "reference_edited.png"
            shutil.copy2(reference_edited_path, reference_edited_dst)
            logger.info(f"  Copied reference_edited.png to output folder")

        # Load metadata if available
        metadata = {}
        object_text = None
        humanoid_name = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"  Humanoid: {metadata.get('humanoid_name', 'N/A')}")
            logger.info(f"  Object: {metadata.get('object_text', 'N/A')}")
            logger.info(f"  Action: {metadata.get('action_prompt', 'N/A')}")
            object_text = metadata.get('object_text', '')
            humanoid_name = metadata.get('humanoid_name', '')

        if not object_text:
            logger.warning(f"  No object_text in metadata, skipping")
            return

        # Extract object name from object_text (e.g., "a balloon" -> "balloon")
        # Remove articles
        object_name = object_text.lower()
        for article in ['a ', 'an ', 'the ']:
            if object_name.startswith(article):
                object_name = object_name[len(article):]
                break

        logger.info(f"  SAM tag for object2: '{object_name}'")

        # Load the flux-edited image
        logger.info(f"  Loading flux-edited image...")
        edited_image = Image.open(reference_edited_path).convert('RGB')
        edited_np = np.array(edited_image)

        # === First: Segment humanoid (object1) ===
        logger.info(f"  Segmenting humanoid (object1)...")
        self.load_trellis_pipeline()  # Load pipeline before SAM (to avoid OOM)

        # Get humanoid tag from simplified keywords
        humanoid_tag = self.humanoid_keywords.get(humanoid_name)

        if humanoid_tag:
            logger.info(f"  Using simplified humanoid keyword: '{humanoid_tag}'")
        else:
            # Fallback to simple tags based on folder name
            logger.warning(f"  No simplified keyword found for {humanoid_name}, using fallback")
            if "humanoid" in humanoid_name.lower():
                humanoid_tag = "person"
            elif "character" in humanoid_name.lower():
                humanoid_tag = "character"
            elif "robot" in humanoid_name.lower():
                humanoid_tag = "robot"
            else:
                humanoid_tag = "person"
            logger.info(f"  Using fallback humanoid tag: '{humanoid_tag}'")

        humanoid_mask, humanoid_score = self.segment_object_with_sam2(edited_image, humanoid_tag)

        # Calculate humanoid bounding box size for reference
        humanoid_box_size = None
        if humanoid_mask is not None and humanoid_mask.mean() >= 0.05:
            # Get bounding box of humanoid mask
            mask_y, mask_x = np.where(humanoid_mask > 0.5)
            if len(mask_y) > 0:
                y_min, y_max = mask_y.min(), mask_y.max()
                x_min, x_max = mask_x.min(), mask_x.max()
                humanoid_box_size = (x_max - x_min) * (y_max - y_min)
                logger.info(f"  Humanoid box size: {humanoid_box_size:.0f} pixels")

        # Check if humanoid mask is too small (likely detection error)
        if humanoid_mask is None or humanoid_mask.mean() < 0.05:  # < 5% coverage
            if humanoid_mask is not None:
                logger.warning(f"  Humanoid coverage too small ({humanoid_mask.mean()*100:.1f}%), likely detection error")
            else:
                logger.warning(f"  Failed to detect humanoid with tag '{humanoid_tag}'")

            logger.info(f"  Fallback: Will detect object2 first, then compute humanoid = foreground - object2")
            humanoid_mask_valid = False
        else:
            logger.info(f"  Humanoid detected successfully (coverage: {humanoid_mask.mean()*100:.1f}%)")
            humanoid_mask_valid = True

        # === Second: Segment object2 ===
        logger.info(f"  Segmenting object2 with tag '{object_name}'...")
        # Pass humanoid box size to prefer smaller boxes
        object2_mask, object2_score = self.segment_object_with_sam2(edited_image, object_name, reference_box_size=humanoid_box_size)
        segmentation_method = 'direct'
        gaussian_source = 'direct'  # Initialize gaussian_source

        # Get foreground mask (will be used in fallback scenarios)
        foreground_mask = self.get_foreground_mask(edited_image)

        # === Validation: Check if humanoid and object2 are detecting the same region ===
        if humanoid_mask is not None and object2_mask is not None:
            humanoid_coverage = humanoid_mask.mean()
            object2_coverage = object2_mask.mean()

            # Check if coverages are similar (within 5% difference suggests same region detected)
            coverage_diff = abs(humanoid_coverage - object2_coverage)
            if coverage_diff < 0.05:  # Similar coverage, likely detecting the same object
                logger.warning(f"  Humanoid and object2 have similar coverage (humanoid: {humanoid_coverage*100:.1f}%, object2: {object2_coverage*100:.1f}%)")
                logger.info(f"  Scores - humanoid: {humanoid_score:.3f}, object2: {object2_score:.3f}")

                # Decide which one is correct based on user's logic:
                # If humanoid_coverage <= object2_coverage AND humanoid_score > object2_score -> humanoid is correct
                # Otherwise -> object2 is correct
                if humanoid_coverage <= object2_coverage and humanoid_score > object2_score:
                    logger.info(f"  -> Humanoid detection appears correct (higher score). Recomputing object2 = foreground - humanoid")
                    object2_mask = np.maximum(foreground_mask - humanoid_mask, 0.0)
                    object2_mask = (object2_mask > 0.5).astype(np.float32)
                    segmentation_method = 'corrected_object2'
                    logger.info(f"  Recomputed object2 mask coverage: {object2_mask.mean()*100:.1f}%")
                else:
                    logger.info(f"  -> Object2 detection appears correct. Recomputing humanoid = foreground - object2")
                    humanoid_mask = np.maximum(foreground_mask - object2_mask, 0.0)
                    humanoid_mask = (humanoid_mask > 0.5).astype(np.float32)
                    humanoid_mask_valid = True
                    segmentation_method = 'corrected_humanoid'
                    logger.info(f"  Recomputed humanoid mask coverage: {humanoid_mask.mean()*100:.1f}%")

        # Fallback scenario 1: if object2 segmentation failed, use foreground - humanoid
        if object2_mask is None:
            segmentation_method = 'fallback_object2'
            gaussian_source = 'fallback_object2'
            logger.warning(f"  SAM2 failed to detect object2, using fallback method...")
            logger.info(f"  Fallback: Object2 = Foreground - Humanoid")

            logger.info(f"  Foreground mask coverage: {foreground_mask.mean()*100:.1f}%")

            # Use already segmented humanoid mask
            # Object2 = Foreground - Humanoid
            object2_mask = np.maximum(foreground_mask - humanoid_mask, 0.0)
            # Clean up small noise
            object2_mask = (object2_mask > 0.5).astype(np.float32)
            logger.info(f"  Fallback successful! Object2 mask coverage: {object2_mask.mean()*100:.1f}%")

        # Fallback scenario 2: if humanoid detection failed earlier, recompute humanoid = foreground - object2
        if not humanoid_mask_valid and object2_mask is not None:
            logger.info(f"  Recomputing humanoid mask: Humanoid = Foreground - Object2")
            logger.info(f"  Foreground mask coverage: {foreground_mask.mean()*100:.1f}%")

            # Humanoid = Foreground - Object2
            humanoid_mask = np.maximum(foreground_mask - object2_mask, 0.0)
            # Clean up small noise
            humanoid_mask = (humanoid_mask > 0.5).astype(np.float32)
            logger.info(f"  Recomputed humanoid mask coverage: {humanoid_mask.mean()*100:.1f}%")
            humanoid_mask_valid = True  # Now we have a valid humanoid mask

        # Check if object2_mask has any content
        if object2_mask.mean() < 0.01:
            logger.warning(f"  Object2 mask is too small or empty, skipping")
            return

        # === FINAL CHECK: Swap if humanoid mask is smaller than object2 mask ===
        humanoid_coverage = humanoid_mask.mean()
        object2_coverage = object2_mask.mean()

        if humanoid_coverage < object2_coverage:
            logger.warning(f"  Humanoid mask ({humanoid_coverage*100:.1f}%) is smaller than object2 mask ({object2_coverage*100:.1f}%)")
            logger.warning(f"  This suggests the masks are swapped. Swapping humanoid and object2 masks...")

            # Swap the masks
            humanoid_mask, object2_mask = object2_mask, humanoid_mask

            # Update segmentation method
            if segmentation_method == 'corrected_humanoid':
                segmentation_method = 'swapped_corrected_humanoid'
            elif segmentation_method == 'corrected_object2':
                segmentation_method = 'swapped_corrected_object2'
            elif segmentation_method == 'fallback_object2':
                segmentation_method = 'swapped_fallback_object2'
            else:
                segmentation_method = 'swapped_' + segmentation_method

            logger.info(f"  Masks swapped! New humanoid coverage: {humanoid_mask.mean()*100:.1f}%, object2 coverage: {object2_mask.mean()*100:.1f}%")
            logger.info(f"  Updated segmentation method: {segmentation_method}")

        # === NEW FLOW: Generate multiview from reference_edited, then segment ===
        logger.info("  === NEW FLOW: MVAdapter on reference_edited, then SAM segment ===")

        if self.use_mvadapter:
            # Generate multiview from the WHOLE reference_edited image
            logger.info(f"  Generating multiview from full reference_edited with MVAdapter...")

            # Use action prompt as the MVAdapter prompt (e.g., "a soldier is holding a map")
            action_prompt = metadata.get('action_prompt', f'{humanoid_tag} with {object_name}')
            logger.info(f"  MVAdapter prompt: '{action_prompt}'")

            mvadapter_full_images, mv_full_angles = self.generate_multiview_with_mvadapter(
                edited_image,  # Full reference_edited image
                action_prompt
            )

            # Save full MVAdapter multiview images
            mvadapter_full_dir = output_folder / "mvadapter_full_reference"
            mvadapter_full_dir.mkdir(exist_ok=True)

            for idx, (mv_img, angle) in enumerate(zip(mvadapter_full_images, mv_full_angles)):
                mv_path = mvadapter_full_dir / f"{angle:03d}_full.png"
                mv_img.save(mv_path)

            logger.info(f"  Saved {len(mvadapter_full_images)} full reference MVAdapter views")

            # Create grid for visualization
            grid_cols = 3
            grid_rows = (len(mvadapter_full_images) + grid_cols - 1) // grid_cols
            grid_width = 768 * grid_cols
            grid_height = 768 * grid_rows
            mvadapter_full_grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

            for idx, mv_img in enumerate(mvadapter_full_images):
                row = idx // grid_cols
                col = idx % grid_cols
                x = col * 768
                y = row * 768
                mvadapter_full_grid.paste(mv_img, (x, y))

            mvadapter_full_grid_path = output_folder / "mvadapter_full_reference_grid.png"
            mvadapter_full_grid.save(mvadapter_full_grid_path)
            logger.info(f"  Saved full reference MVAdapter grid")

            # Store MV images for later SAM processing
            logger.info(f"  Stored {len(mvadapter_full_images)} MV views for later SAM processing")

            # Clean up MVAdapter to free memory before SAM
            logger.info("  Cleaning up MVAdapter pipeline...")
            if hasattr(self, 'mvadapter_pipe') and self.mvadapter_pipe is not None:
                del self.mvadapter_pipe
                self.mvadapter_pipe = None
            torch.cuda.empty_cache()
            logger.info("  MVAdapter cleaned up")

        # === ORIGINAL FLOW: Direct segment from reference_edited (保留) ===
        logger.info("  === ORIGINAL FLOW: Direct segment from reference_edited ===")

        # Save humanoid mask and segmented image
        humanoid_mask_img = (humanoid_mask * 255).astype(np.uint8)
        humanoid_mask_pil = Image.fromarray(humanoid_mask_img, mode='L')
        humanoid_mask_path = output_folder / "humanoid_mask.png"
        humanoid_mask_pil.save(humanoid_mask_path)
        logger.info(f"  Saved humanoid mask to {humanoid_mask_path}")

        # Clean mask edges by thresholding to remove gray transition areas
        humanoid_mask_clean = (humanoid_mask > 0.5).astype(np.float32)
        humanoid_mask_3ch = np.stack([humanoid_mask_clean, humanoid_mask_clean, humanoid_mask_clean], axis=-1)
        humanoid_rgb = (edited_np * humanoid_mask_3ch + 255 * (1 - humanoid_mask_3ch)).astype(np.uint8)
        humanoid_rgb_pil = Image.fromarray(humanoid_rgb)
        humanoid_rgb_path = output_folder / "humanoid_segmented.png"
        humanoid_rgb_pil.save(humanoid_rgb_path)
        logger.info(f"  Saved segmented humanoid RGB to {humanoid_rgb_path}")

        # Save object2 mask and segmented image
        object2_mask_img = (object2_mask * 255).astype(np.uint8)
        object2_mask_pil = Image.fromarray(object2_mask_img, mode='L')
        object2_mask_path = output_folder / "object2_mask.png"
        object2_mask_pil.save(object2_mask_path)
        logger.info(f"  Saved object2 mask to {object2_mask_path}")

        # Clean mask edges by thresholding to remove gray transition areas
        object2_mask_clean = (object2_mask > 0.5).astype(np.float32)
        object2_mask_3ch = np.stack([object2_mask_clean, object2_mask_clean, object2_mask_clean], axis=-1)
        object2_rgb = (edited_np * object2_mask_3ch + 255 * (1 - object2_mask_3ch)).astype(np.uint8)
        object2_rgb_pil = Image.fromarray(object2_rgb)
        object2_rgb_path = output_folder / "object2_segmented.png"
        object2_rgb_pil.save(object2_rgb_path)
        logger.info(f"  Saved segmented object2 RGB to {object2_rgb_path}")

        # Skip 3D Gaussian generation
        logger.info(f"  Skipping TRELLIS 3D Gaussian generation")

        # === NOW DO SAM SEGMENTATION (After all MVAdapter generation is done) ===
        if self.use_mvadapter:
            logger.info("  === SAM Segmentation from MVAdapter views ===")

            # Reload MV images from disk
            mvadapter_full_dir = output_folder / "mvadapter_full_reference"
            mv_full_files = sorted(mvadapter_full_dir.glob("*_full.png"))

            if len(mv_full_files) > 0:
                # Segment from view 0
                mv_reference_image = Image.open(mv_full_files[0])
                angle_0 = int(mv_full_files[0].stem.split('_')[0])
                logger.info(f"  Segmenting humanoid and object from MV view {angle_0}°...")

                # Segment humanoid from MV
                mv_humanoid_mask, mv_humanoid_score = self.segment_object_with_sam2(
                    mv_reference_image,
                    humanoid_tag
                )

                if mv_humanoid_mask is not None:
                    # Save MV humanoid mask and segmented image
                    mv_humanoid_mask_img = (mv_humanoid_mask * 255).astype(np.uint8)
                    mv_humanoid_mask_pil = Image.fromarray(mv_humanoid_mask_img, mode='L')
                    mv_humanoid_mask_path = output_folder / "mv_humanoid_mask.png"
                    mv_humanoid_mask_pil.save(mv_humanoid_mask_path)

                    mv_reference_np = np.array(mv_reference_image)
                    mv_humanoid_mask_3ch = np.stack([mv_humanoid_mask, mv_humanoid_mask, mv_humanoid_mask], axis=-1)
                    mv_humanoid_rgb = (mv_reference_np * mv_humanoid_mask_3ch + 255 * (1 - mv_humanoid_mask_3ch)).astype(np.uint8)
                    mv_humanoid_rgb_pil = Image.fromarray(mv_humanoid_rgb)
                    mv_humanoid_rgb_path = output_folder / "mv_humanoid_reference.png"
                    mv_humanoid_rgb_pil.save(mv_humanoid_rgb_path)
                    logger.info(f"  Saved MV humanoid reference (coverage: {mv_humanoid_mask.mean()*100:.1f}%)")
                else:
                    logger.warning(f"  Failed to segment humanoid from MV view")

                # Segment object from MV view 0
                mv_object_mask, mv_object_score = self.segment_object_with_sam2(
                    mv_reference_image,
                    object_name
                )

                # Check for detection errors
                if mv_object_mask is not None:
                    object_coverage = mv_object_mask.mean()

                    # Check 1: Object coverage > 35% (too large)
                    if object_coverage > 0.35:
                        logger.warning(f"  Object coverage > 35% ({object_coverage*100:.1f}%), likely detection error")
                        logger.warning(f"  Skipping object mask for MV view {angle_0}°")
                        mv_object_mask = None

                    # Check 2: Object and humanoid have similar coverage
                    elif mv_humanoid_mask is not None:
                        humanoid_coverage = mv_humanoid_mask.mean()
                        coverage_diff = abs(object_coverage - humanoid_coverage)

                        # If coverages are within 5%, likely same object detected
                        if coverage_diff < 0.05:
                            logger.warning(f"  Object and humanoid have similar coverage (object: {object_coverage*100:.1f}%, humanoid: {humanoid_coverage*100:.1f}%), likely detection error")
                            logger.warning(f"  Skipping object mask for MV view {angle_0}°")
                            mv_object_mask = None

                if mv_object_mask is not None:
                    # Save MV object mask and segmented image
                    mv_object_mask_img = (mv_object_mask * 255).astype(np.uint8)
                    mv_object_mask_pil = Image.fromarray(mv_object_mask_img, mode='L')
                    mv_object_mask_path = output_folder / "mv_object_mask.png"
                    mv_object_mask_pil.save(mv_object_mask_path)

                    mv_reference_np = np.array(mv_reference_image)
                    mv_object_mask_3ch = np.stack([mv_object_mask, mv_object_mask, mv_object_mask], axis=-1)
                    mv_object_rgb = (mv_reference_np * mv_object_mask_3ch + 255 * (1 - mv_object_mask_3ch)).astype(np.uint8)
                    mv_object_rgb_pil = Image.fromarray(mv_object_rgb)
                    mv_object_rgb_path = output_folder / "mv_object_reference.png"
                    mv_object_rgb_pil.save(mv_object_rgb_path)
                    logger.info(f"  Saved MV object reference (coverage: {mv_object_mask.mean()*100:.1f}%)")
                else:
                    logger.warning(f"  Failed to segment object from MV view")

            # === Extract Object from ALL Full MVAdapter Views ===
            logger.info("  === Extracting object from all MV views ===")

            mvadapter_object_dir = output_folder / "mvadapter_object"
            mvadapter_object_dir.mkdir(exist_ok=True)

            mvadapter_object_images = []
            mvadapter_object_masks = []

            for mv_file in mv_full_files:
                mv_img = Image.open(mv_file)
                angle = int(mv_file.stem.split('_')[0])

                logger.info(f"  Processing object from MV view {angle}°...")

                # First segment humanoid from this view
                mv_view_humanoid_mask, mv_view_humanoid_score = self.segment_object_with_sam2(
                    mv_img,
                    humanoid_tag
                )

                # Then segment object from this view
                mv_view_object_mask, mv_view_object_score = self.segment_object_with_sam2(
                    mv_img,
                    object_name
                )

                # Skip this view if SAM fails
                if mv_view_object_mask is None or mv_view_object_mask.mean() < 0.01:
                    logger.warning(f"    Object SAM failed for view {angle}°, skipping this view")
                    continue

                # Check for detection errors
                object_coverage = mv_view_object_mask.mean()

                # Check 1: Object coverage > 35% (too large)
                if object_coverage > 0.35:
                    logger.warning(f"    Object coverage > 35% ({object_coverage*100:.1f}%) for view {angle}°, likely detection error")
                    logger.warning(f"    Skipping view {angle}°")
                    continue

                # Check 2: Object and humanoid have similar coverage
                if mv_view_humanoid_mask is not None:
                    humanoid_coverage = mv_view_humanoid_mask.mean()
                    coverage_diff = abs(object_coverage - humanoid_coverage)

                    # If coverages are within 5%, likely same object detected
                    if coverage_diff < 0.05:
                        logger.warning(f"    Object and humanoid have similar coverage (object: {object_coverage*100:.1f}%, humanoid: {humanoid_coverage*100:.1f}%), likely detection error")
                        logger.warning(f"    Skipping view {angle}°")
                        continue

                logger.info(f"    SAM successful (coverage: {object_coverage*100:.1f}%)")

                # Apply mask to extract object
                mv_img_np = np.array(mv_img)
                mv_view_object_mask_3ch = np.stack([mv_view_object_mask, mv_view_object_mask, mv_view_object_mask], axis=-1)
                mv_view_object_rgb = (mv_img_np * mv_view_object_mask_3ch + 255 * (1 - mv_view_object_mask_3ch)).astype(np.uint8)
                mv_view_object_rgb_pil = Image.fromarray(mv_view_object_rgb)

                # Save object image
                object_img_path = mvadapter_object_dir / f"{angle:03d}_object.png"
                mv_view_object_rgb_pil.save(object_img_path)

                # Save object mask
                object_mask_img = (mv_view_object_mask * 255).astype(np.uint8)
                object_mask_pil = Image.fromarray(object_mask_img, mode='L')
                object_mask_path = mvadapter_object_dir / f"{angle:03d}_object_mask.png"
                object_mask_pil.save(object_mask_path)

                mvadapter_object_images.append(mv_view_object_rgb_pil)
                mvadapter_object_masks.append(object_mask_pil)

            logger.info(f"  Saved {len(mvadapter_object_images)} object views and masks to {mvadapter_object_dir}")

            # Create object grid (RGB)
            if len(mvadapter_object_images) > 0:
                object_grid_cols = 3
                object_grid_rows = (len(mvadapter_object_images) + object_grid_cols - 1) // object_grid_cols
                object_grid_width = 768 * object_grid_cols
                object_grid_height = 768 * object_grid_rows
                mvadapter_object_grid = Image.new('RGB', (object_grid_width, object_grid_height), (255, 255, 255))

                for idx, mv_img in enumerate(mvadapter_object_images):
                    row = idx // object_grid_cols
                    col = idx % object_grid_cols
                    x = col * 768
                    y = row * 768
                    mvadapter_object_grid.paste(mv_img, (x, y))

                mvadapter_object_grid_path = output_folder / "mvadapter_object_grid.png"
                mvadapter_object_grid.save(mvadapter_object_grid_path)
                logger.info(f"  Saved object MVAdapter grid to {mvadapter_object_grid_path}")

                # Create object mask grid
                mvadapter_object_mask_grid = Image.new('L', (object_grid_width, object_grid_height), 255)

                for idx, mv_mask in enumerate(mvadapter_object_masks):
                    row = idx // object_grid_cols
                    col = idx % object_grid_cols
                    x = col * 768
                    y = row * 768
                    mvadapter_object_mask_grid.paste(mv_mask, (x, y))

                mvadapter_object_mask_grid_path = output_folder / "mvadapter_object_mask_grid.png"
                mvadapter_object_mask_grid.save(mvadapter_object_mask_grid_path)
                logger.info(f"  Saved object mask grid to {mvadapter_object_mask_grid_path}")

        # Skip Gaussian rendering
        logger.info(f"  Skipping Gaussian rendering")

        # Save updated metadata
        output_metadata = {
            **metadata,
            'sam_tag_object': object_name,
            'sam_tag_humanoid': humanoid_tag,
            'segmentation_method': segmentation_method,
            'mvadapter_full_reference_generated': self.use_mvadapter,
            'mv_reference_segmented': self.use_mvadapter,
        }
        with open(output_folder / "metadata.json", 'w') as f:
            json.dump(output_metadata, f, indent=2)

        logger.info(f"  ✓ Completed: {folder_name}")

    def process_all(self, folder_names=None):
        """Process all folders in flux_edit directory

        Args:
            folder_names: Optional list of specific folder names to process
        """
        logger.info("="*80)
        logger.info("FLUX-EDIT MULTIVIEW GENERATION")
        logger.info("="*80)

        # Get all folders in flux_edit
        if folder_names:
            # Process only specified folders
            folders = []
            for name in folder_names:
                folder_path = self.good_edit_dir / name
                if folder_path.exists() and folder_path.is_dir():
                    folders.append(folder_path)
                else:
                    logger.warning(f"Folder not found: {name}")
            folders = sorted(folders)
            logger.info(f"Processing {len(folders)} specified folders")
        else:
            # Process all folders
            folders = sorted([f for f in self.good_edit_dir.iterdir() if f.is_dir()])
            logger.info(f"Found {len(folders)} folders to process")

        # Process each folder
        for folder in tqdm(folders, desc="Processing folders"):
            try:
                self.process_folder(folder)
            except Exception as e:
                logger.error(f"Error processing {folder.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Cleanup
        torch.cuda.empty_cache()

        logger.info("\n" + "="*80)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate multiview and 3D from FLUX-edited images'
    )
    parser.add_argument(
        '--good_edit_dir',
        type=str,
        default='/data4/zishuo/TRELLIS/flux_edit',
        help='Path to flux_edit directory containing reference_edited.png files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data4/zishuo/TRELLIS/flux_edit_multiview',
        help='Output directory for multiview renders'
    )
    parser.add_argument(
        '--object_data_dir',
        type=str,
        default='/data4/zishuo/TRELLIS/object_data',
        help='Path to object_data directory containing PNG images for objects'
    )
    parser.add_argument(
        '--num_views',
        type=int,
        default=8,
        help='Number of views to render from Gaussian'
    )
    parser.add_argument(
        '--use_mvadapter',
        action='store_true',
        default=True,
        help='Use MVAdapter to generate multiview (default: True)'
    )
    parser.add_argument(
        '--no_mvadapter',
        action='store_false',
        dest='use_mvadapter',
        help='Do not use MVAdapter'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    parser.add_argument(
        '--folders',
        type=str,
        nargs='+',
        default=None,
        help='Specific folder names to process (e.g., cat_1_017 fox_1_013). If not specified, process all folders.'
    )

    args = parser.parse_args()

    generator = FluxEditMultiviewGenerator(
        good_edit_dir=args.good_edit_dir,
        output_dir=args.output_dir,
        object_data_dir=args.object_data_dir,
        device=args.device,
        num_views=args.num_views,
        use_mvadapter=args.use_mvadapter
    )

    generator.process_all(folder_names=args.folders)


if __name__ == '__main__':
    main()