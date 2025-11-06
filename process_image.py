#!/usr/bin/env python3
"""
Image Processing Utilities for TRELLIS Pipeline
Contains image processing functions extracted from sds_mv_mesh.py
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import logging
from PIL import Image
import torch.nn.functional as F
from glob import glob

logger = logging.getLogger(__name__)


# ==================== Image Processing Mixin Class ====================
# This class contains all image processing methods that can be inherited by the main trainer

class ImageProcessingMixin:
    """Mixin class containing image processing methods for TRELLIS training"""

    @staticmethod
    def compute_mask_from_gray_bg_rgb(rgb_tensor, bg_color=(1.0, 1.0, 1.0), threshold=0.05):
        """
        Compute mask from RGB image with white background (differentiable)
        Background = 0 (black), Foreground = 1 (white)

        Args:
            rgb_tensor: torch.Tensor [H, W, 3] in [0, 1]
            bg_color: Background color tuple (R, G, B) in [0, 1], default (1.0, 1.0, 1.0) for white
            threshold: Color difference threshold to consider as foreground

        Returns:
            mask: torch.Tensor [H, W, 1] in [0, 1] (background=0, foreground=1)
        """
        device = rgb_tensor.device
        bg = torch.tensor(bg_color, device=device).view(1, 1, 3)

        # Compute color difference from background
        color_diff = torch.abs(rgb_tensor - bg).sum(dim=-1, keepdim=True)  # [H, W, 1]

        # Use sigmoid for soft thresholding (differentiable)
        # sigmoid((color_diff - threshold) * sharpness)
        sharpness = 1000.0  # High sharpness for near-binary mask
        mask = torch.sigmoid((color_diff - threshold) * sharpness)

        return mask

    def load_reference_images(self, reference_images_dir: str, camera_params: list = None):
        """Load reference images for reconstruction loss

        Args:
            reference_images_dir: Directory containing reference images
            camera_params: List of dicts with 'azimuth', 'elevation', 'radius' for each view
                          If None, will use default camera positions
        """
        logger.info("=== Loading Reference Images ===")
        self.reference_images_dir = reference_images_dir
        self.use_reconstruction_loss = True

        # Get all image files, excluding intermediate files (obj1_render.png, obj1_mask.png)
        all_files = sorted(glob(os.path.join(reference_images_dir, "*.png")) +
                          glob(os.path.join(reference_images_dir, "*.jpg")))

        # Filter to ONLY load reference_edited.png as the main training image
        image_files = [f for f in all_files if os.path.basename(f) == "reference_edited.png"]

        if len(image_files) == 0:
            logger.warning(f"No reference images found in {reference_images_dir}")
            self.use_reconstruction_loss = False
            return

        logger.info(f"Found {len(image_files)} reference images (filtered from {len(all_files)} total files)")

        # Default camera parameters if not provided
        if camera_params is None:
            # Use 8 views around the center, starting from main view at 135 degrees
            # Main view matches the reference generation angle (azimuth = 135 degrees = 3π/4)
            # Then rotate 360 degrees with 8 evenly spaced views (45 degrees apart)
            main_azimuth = np.radians(self.reference_azimuth)  # 135 degrees - main viewing angle
            num_views = 8
            azimuth_step = 2 * np.pi / num_views  # 45 degrees

            camera_params = []
            for i in range(num_views):
                azimuth = main_azimuth + i * azimuth_step
                # Normalize to [0, 2π)
                azimuth = azimuth % (2 * np.pi)
                camera_params.append({
                    'azimuth': azimuth,
                    'elevation': 0.0,
                    'radius': 1.2
                })

            logger.info(f"Using {num_views} views around center, starting from {np.degrees(main_azimuth):.1f}° (main view)")

        # Ensure we have camera params for all images
        if len(camera_params) < len(image_files):
            logger.warning(f"Only {len(camera_params)} camera params provided for {len(image_files)} images. Using first {len(camera_params)} images.")
            image_files = image_files[:len(camera_params)]

        # Load each reference view
        for img_path, cam_params in zip(image_files, camera_params):
            # Load RGB image
            img = Image.open(img_path).convert("RGB")
            rgb = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]

            # Try to load corresponding mask files (separate for obj1 and obj2)
            img_name = Path(img_path).stem
            img_dir = Path(img_path).parent

            # Try to load obj1 mask
            obj1_mask_path = None
            for pattern in [f"{img_name}_obj1_mask.png", "reference_edited_obj1_mask.png"]:
                candidate = img_dir / pattern
                if candidate.exists():
                    obj1_mask_path = candidate
                    break

            if obj1_mask_path is not None:
                # Load obj1 mask from file
                obj1_mask_img = Image.open(obj1_mask_path).convert("L")
                obj1_alpha = np.array(obj1_mask_img).astype(np.float32) / 255.0  # [H, W]
                obj1_alpha = obj1_alpha[:, :, np.newaxis]  # [H, W, 1]
                logger.info(f"Loaded obj1 mask from {obj1_mask_path}")
            else:
                # No obj1 mask found, assume full opacity
                obj1_alpha = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
                logger.warning(f"No obj1 mask file found for {img_path}, assuming full opacity")

            # Try to load obj2 mask
            obj2_mask_path = None
            for pattern in [f"{img_name}_obj2_mask.png", "reference_edited_obj2_mask.png"]:
                candidate = img_dir / pattern
                if candidate.exists():
                    obj2_mask_path = candidate
                    break

            if obj2_mask_path is not None:
                # Load obj2 mask from file
                obj2_mask_img = Image.open(obj2_mask_path).convert("L")
                obj2_alpha = np.array(obj2_mask_img).astype(np.float32) / 255.0  # [H, W]
                obj2_alpha = obj2_alpha[:, :, np.newaxis]  # [H, W, 1]
                logger.info(f"Loaded obj2 mask from {obj2_mask_path}")
            else:
                # No obj2 mask found, assume empty (no obj2)
                obj2_alpha = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
                logger.warning(f"No chair mask file found for {img_path}, assuming no chair")

            # Also try to load combined mask (for compatibility)
            combined_mask_path = None
            for pattern in [f"{img_name}_mask.png", "reference_edited_mask.png"]:
                candidate = img_dir / pattern
                if candidate.exists():
                    combined_mask_path = candidate
                    break

            if combined_mask_path is not None:
                combined_mask_img = Image.open(combined_mask_path).convert("L")
                combined_alpha = np.array(combined_mask_img).astype(np.float32) / 255.0  # [H, W]
                combined_alpha = combined_alpha[:, :, np.newaxis]  # [H, W, 1]
            else:
                # Combine obj1 and obj2 masks
                combined_alpha = np.maximum(obj1_alpha, obj2_alpha)

            # Try to load obj1-only RGB
            obj1_rgb_path = None
            for pattern in [f"{img_name}_obj1_only_rgb.png", "reference_obj1_only_rgb.png"]:
                candidate = img_dir / pattern
                if candidate.exists():
                    obj1_rgb_path = candidate
                    break

            if obj1_rgb_path is not None:
                obj1_rgb_img = Image.open(obj1_rgb_path).convert("RGB")
                obj1_rgb = np.array(obj1_rgb_img).astype(np.float32) / 255.0  # [H, W, 3]
                # Fix background: ensure white (1.0) outside mask
                obj1_rgb[obj1_alpha[:, :, 0] == 0] = 1.0
                logger.info(f"Loaded obj1-only RGB from {obj1_rgb_path} (background corrected to white)")
            else:
                # If no separate obj1 RGB, use combined RGB (fallback)
                obj1_rgb = rgb.copy()
                logger.warning(f"No obj1-only RGB file found for {img_path}, using combined RGB")

            # Try to load obj2-only RGB
            obj2_rgb_path = None
            for pattern in [f"{img_name}_obj2_only_rgb.png", "reference_obj2_only_rgb.png"]:
                candidate = img_dir / pattern
                if candidate.exists():
                    obj2_rgb_path = candidate
                    break

            if obj2_rgb_path is not None:
                obj2_rgb_img = Image.open(obj2_rgb_path).convert("RGB")
                obj2_rgb = np.array(obj2_rgb_img).astype(np.float32) / 255.0  # [H, W, 3]
                # Keep WHITE background for obj2 (used in Phase 2 training)
                logger.info(f"Loaded obj2-only RGB from {obj2_rgb_path} (white background for Phase 2)")
            else:
                # If no separate obj2 RGB, use combined RGB (fallback)
                obj2_rgb = rgb.copy()
                logger.warning(f"No obj2-only RGB file found for {img_path}, using combined RGB")

            # Convert to torch tensors
            rgb_tensor = torch.from_numpy(rgb).to(self.device)
            obj1_rgb_tensor = torch.from_numpy(obj1_rgb).to(self.device)
            obj2_rgb_tensor = torch.from_numpy(obj2_rgb).to(self.device)
            obj1_mask_tensor = torch.from_numpy(obj1_alpha).to(self.device)
            obj2_mask_tensor = torch.from_numpy(obj2_alpha).to(self.device)
            combined_mask_tensor = torch.from_numpy(combined_alpha).to(self.device)

            self.reference_views.append({
                'image': rgb_tensor,  # [H, W, 3] combined RGB (reference_edited.png)
                'obj1_rgb': obj1_rgb_tensor,  # [H, W, 3] obj1-only RGB
                'obj2_rgb': obj2_rgb_tensor,  # [H, W, 3] obj2-only RGB
                'mask': combined_mask_tensor,   # [H, W, 1] combined mask (for full reconstruction)
                'obj1_mask': obj1_mask_tensor,    # [H, W, 1] obj1-only mask
                'obj2_mask': obj2_mask_tensor, # [H, W, 1] obj2-only mask
                'azimuth': cam_params['azimuth'],
                'elevation': cam_params['elevation'],
                'radius': cam_params['radius'],
                'path': img_path,
                'weight': 1.0  # Reference image weight
            })

            logger.info(f"Loaded: {os.path.basename(img_path)} - "
                       f"azimuth={np.degrees(cam_params['azimuth']):.1f}°, "
                       f"elevation={np.degrees(cam_params['elevation']):.1f}°")
            logger.info(f"  {self.object1_name.capitalize()} mask coverage: {obj1_alpha.mean()*100:.1f}%, "
                       f"{self.object2_name.capitalize()} mask coverage: {obj2_alpha.mean()*100:.1f}%")

        logger.info(f"Loaded {len(self.reference_views)} reference views for reconstruction")

    def load_obj1_multiview_images(self):
        """
        Load obj1-only multiview images for Phase 1 training

        Loads 6 angles: 0°, 45°, 90°, 180°, 270°, 315° (target angles in MVAdapter space)
        - 0°, 90°, 180°, 270°, 315° from MVAdapter generation
        - 45° uses reference image directly (reference_obj1_only_rgb.png)

        TRELLIS angle conversion: trellis_angle = 180 - target_angle

        Sets different weights:
        - 45°, 90°, 180°: weight = 1.0 (front and side views)
        - 0°, 270°, 315°: weight = 0.5 (back views)
        """
        logger.info("=== Loading Man-Only Multiview Images for Phase 1 ===")

        # Initialize obj1_multiview_images list
        self.obj1_multiview_images = []

        # Object1 multiview directory (using dynamic name)
        obj1_multiview_dir = self.output_dir / f"{self.object1_name}_multiview"
        if not obj1_multiview_dir.exists():
            logger.warning(f"{self.object1_name.capitalize()} multiview directory not found: {obj1_multiview_dir}")
            logger.warning(f"Please run generate_only_multiview_for_phase1() first")
            return

        # Define 6 target angles and their weights
        # Angles: 0°, 45°, 90°, 180°, 270°, 315° (in MVAdapter target space)
        # 45° will use reference image directly instead of MVAdapter output
        # Weights: 45°=1.0 (reference), others=0.2
        angle_configs = [
            {'target_angle': 0.0, 'weight': 0.2, 'use_reference': False},
            {'target_angle': 45.0, 'weight': 1.0, 'use_reference': True},  # Use reference image
            {'target_angle': 90.0, 'weight': 0.2, 'use_reference': False},
            {'target_angle': 180.0, 'weight': 0.2, 'use_reference': False},
            {'target_angle': 270.0, 'weight': 0.2, 'use_reference': False},
            {'target_angle': 315.0, 'weight': 0.2, 'use_reference': False},
        ]

        logger.info(f"Loading {len(angle_configs)} obj1-only multiview images")
        logger.info("Note: 45° uses reference image (TRELLIS angle 135°)")

        for config in angle_configs:
            target_angle = config['target_angle']
            weight = config['weight']
            use_reference = config['use_reference']

            # Special handling for 45° - use reference image
            if use_reference:
                logger.info(f"Loading 45° from reference image (target angle {target_angle}°, weight={weight})")

                # Load reference obj1-only RGB
                ref_path = Path(self.reference_images_dir) / "reference_obj1_only_rgb.png"
                if not ref_path.exists():
                    logger.warning(f"Reference obj1-only RGB not found: {ref_path}, skipping 45°")
                    continue

                rgb_img = Image.open(ref_path).convert("RGB")
                rgb_np = np.array(rgb_img).astype(np.float32) / 255.0
                rgb_tensor = torch.from_numpy(rgb_np).to(self.device)

                # Load reference obj1 mask
                mask_path = Path(self.reference_images_dir) / "reference_edited_obj1_mask.png"
                if not mask_path.exists():
                    logger.warning(f"Reference obj1 mask not found: {mask_path}, skipping 45°")
                    continue

                mask_img = Image.open(mask_path).convert("L")
                mask_np = np.array(mask_img).astype(np.float32) / 255.0
                mask_np = mask_np[:, :, np.newaxis]
                mask_tensor = torch.from_numpy(mask_np).to(self.device)

            else:
                # Load from MVAdapter output
                rgb_filename = f"{int(target_angle):03d}.png"
                mask_filename = f"{int(target_angle):03d}_mask.png"

                rgb_path = obj1_multiview_dir / rgb_filename
                mask_path = obj1_multiview_dir / mask_filename

                # Check if files exist
                if not rgb_path.exists():
                    logger.warning(f"Man multiview RGB not found: {rgb_path}, skipping")
                    continue
                if not mask_path.exists():
                    logger.warning(f"Man multiview mask not found: {mask_path}, skipping")
                    continue

                logger.info(f"Loading {rgb_filename} (target angle {target_angle}°, weight={weight})")

                # Load RGB image
                rgb_img = Image.open(rgb_path).convert("RGB")
                rgb_np = np.array(rgb_img).astype(np.float32) / 255.0  # [H, W, 3]
                rgb_tensor = torch.from_numpy(rgb_np).to(self.device)

                # Load mask image
                mask_img = Image.open(mask_path).convert("L")
                mask_np = np.array(mask_img).astype(np.float32) / 255.0  # [H, W], 0-1
                mask_np = mask_np[:, :, np.newaxis]  # [H, W, 1]
                mask_tensor = torch.from_numpy(mask_np).to(self.device)

            # Convert target angle to TRELLIS render angle
            # TRELLIS angle = 180 - target_angle
            trellis_render_angle = 180.0 - target_angle

            # Convert to radians for rendering
            azimuth_rad = np.radians(trellis_render_angle) % (2 * np.pi)

            # Store in obj1_multiview_images
            self.obj1_multiview_images.append({
                'image': rgb_tensor,         # [H, W, 3] obj1-only RGB with gray background
                'mask': mask_tensor,         # [H, W, 1] obj1 mask
                'azimuth': azimuth_rad,      # TRELLIS azimuth in radians
                'elevation': 0.0,            # Elevation (always 0)
                'radius': 1.2,               # Camera radius
                'target_angle': target_angle,  # Original target angle (for logging)
                'trellis_angle': trellis_render_angle,  # TRELLIS render angle
                'weight': weight,            # Reconstruction weight
                'angle_name': f"{int(target_angle):03d}",  # For logging
            })

            source = "reference" if use_reference else "MVAdapter"
            logger.info(f"  Loaded {source}: target={target_angle}°, trellis={trellis_render_angle}°, "
                       f"azimuth={np.degrees(azimuth_rad):.1f}°, weight={weight}, "
                       f"mask_coverage={mask_tensor.mean().item()*100:.1f}%")

        logger.info(f"Loaded {len(self.obj1_multiview_images)} obj1-only multiview images for Phase 1")

        # Summary
        if len(self.obj1_multiview_images) > 0:
            total_weight = sum(mv['weight'] for mv in self.obj1_multiview_images)
            logger.info(f"Total multiview weight: {total_weight:.1f}")
            logger.info("Angle weights:")
            for mv in self.obj1_multiview_images:
                logger.info(f"  - {mv['target_angle']:.0f}°: weight={mv['weight']}")

    def normalize_foreground_scale(self, image: torch.Tensor, mask: torch.Tensor,
                                  target_scale: float = None):
        """
        Normalize the scale of foreground objects in the image

        FIXED: Now correctly handles off-center foreground objects by:
        1. Computing foreground center position
        2. Scaling around the foreground center (not image center)
        3. Ensuring foreground is not cropped when off-center

        Args:
            image: [H, W, 3] RGB image
            mask: [H, W, 1] binary mask
            target_scale: Target foreground ratio. If None, will be computed

        Returns:
            normalized_image: [H, W, 3] scaled image
            normalized_mask: [H, W, 1] scaled mask
            scale_factor: The scale factor applied
        """
        H, W = image.shape[:2]

        # Find bounding box of foreground
        mask_binary = (mask > 0.5).squeeze(-1)  # [H, W]
        if not mask_binary.any():
            # No foreground, return original
            return image, mask, 1.0

        # Get foreground bounding box
        rows = torch.any(mask_binary, dim=1)
        cols = torch.any(mask_binary, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]

        # Compute current foreground size and center
        fg_height = (ymax - ymin + 1).float()
        fg_width = (xmax - xmin + 1).float()
        fg_size = max(fg_height, fg_width)
        fg_center_y = (ymin + ymax) / 2.0
        fg_center_x = (xmin + xmax) / 2.0

        # Compute target size if not provided
        if target_scale is None:
            # Target: foreground occupies 60% of image size
            target_scale = 0.6

        target_size = min(H, W) * target_scale
        scale_factor = target_size / fg_size

        # Only scale if difference is significant (>10%)
        if abs(scale_factor - 1.0) < 0.1:
            return image, mask, 1.0

        # Compute new size
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)

        # Resize image and mask
        image_BCHW = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        mask_BCHW = mask.permute(2, 0, 1).unsqueeze(0)    # [1, 1, H, W]

        image_resized = F.interpolate(image_BCHW, size=(new_H, new_W), mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask_BCHW, size=(new_H, new_W), mode='bilinear', align_corners=False)

        # Compute where the foreground center is in the resized image
        fg_center_y_resized = fg_center_y * scale_factor
        fg_center_x_resized = fg_center_x * scale_factor

        # Pad or crop to original size, keeping foreground centered
        if scale_factor > 1.0:
            # Scaled up - need to crop, but crop around the foreground center
            # We want the foreground center in resized image to map to the same position in final image
            crop_y = int(fg_center_y_resized - fg_center_y)
            crop_x = int(fg_center_x_resized - fg_center_x)

            # Clamp to valid range
            crop_y = max(0, min(crop_y, new_H - H))
            crop_x = max(0, min(crop_x, new_W - W))

            image_final = image_resized[:, :, crop_y:crop_y+H, crop_x:crop_x+W]
            mask_final = mask_resized[:, :, crop_y:crop_y+H, crop_x:crop_x+W]
        else:
            # Scaled down - need to pad, keeping foreground at the same position
            pad_y = int(fg_center_y - fg_center_y_resized)
            pad_x = int(fg_center_x - fg_center_x_resized)

            # Clamp to valid range
            pad_y = max(0, min(pad_y, H - new_H))
            pad_x = max(0, min(pad_x, W - new_W))

            bg_color = torch.tensor(self.bg_color, device=image.device).view(1, 3, 1, 1)
            image_final = torch.ones(1, 3, H, W, device=image.device) * bg_color
            mask_final = torch.zeros(1, 1, H, W, device=mask.device)
            image_final[:, :, pad_y:pad_y+new_H, pad_x:pad_x+new_W] = image_resized
            mask_final[:, :, pad_y:pad_y+new_H, pad_x:pad_x+new_W] = mask_resized

        # Convert back to [H, W, C] format
        image_normalized = image_final.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        mask_normalized = mask_final.squeeze(0).permute(1, 2, 0)    # [H, W, 1]

        return image_normalized, mask_normalized, scale_factor.item()

    def compute_iou(self, mask1: torch.Tensor, mask2: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Compute soft IoU (Intersection over Union) between two masks
        Uses continuous mask values instead of binarization to maintain gradients

        Args:
            mask1: [H, W, 1] or [H, W] first mask (continuous values in [0, 1])
            mask2: [H, W, 1] or [H, W] second mask (continuous values in [0, 1])
            threshold: unused (kept for API compatibility)

        Returns:
            iou: scalar tensor with IoU value in [0, 1]
        """
        # Ensure same shape
        if mask1.dim() == 3:
            mask1 = mask1.squeeze(-1)
        if mask2.dim() == 3:
            mask2 = mask2.squeeze(-1)

        # Soft IoU: use continuous values (differentiable)
        # intersection = sum of element-wise minimum
        # union = sum of element-wise maximum
        intersection = torch.min(mask1, mask2).sum()
        union = torch.max(mask1, mask2).sum()

        # Avoid division by zero
        iou = intersection / (union + 1e-8)

        return iou

    def _save_segmented_view(self, image: Image.Image, masks_dict: dict, output_dir: Path, view_name: str):
        """Helper function to save segmented objects from a view

        Args:
            image: PIL Image
            masks_dict: Dictionary with object masks (keys are object names)
            output_dir: Directory to save results
            view_name: Base name for saved files
        """
        image_np = np.array(image).astype(np.float32) / 255.0  # [H, W, 3]

        # Process object1 mask
        if self.object1_name in masks_dict and masks_dict[self.object1_name].sum() > 0:
            object1_mask = masks_dict[self.object1_name]  # [H, W] numpy
            logger.info(f"  {self.object1_name.capitalize()} mask coverage: {object1_mask.mean()*100:.1f}%")
        else:
            logger.warning(f"  No {self.object1_name} detected in {view_name}, using full image")
            object1_mask = np.ones((image.size[1], image.size[0]), dtype=np.float32)

        # Process object2 mask
        if self.object2_name in masks_dict and masks_dict[self.object2_name].sum() > 0:
            object2_mask = masks_dict[self.object2_name]  # [H, W] numpy
            logger.info(f"  {self.object2_name.capitalize()} mask coverage: {object2_mask.mean()*100:.1f}%")
        else:
            logger.warning(f"  No {self.object2_name} detected in {view_name}, using zeros")
            object2_mask = np.zeros((image.size[1], image.size[0]), dtype=np.float32)

        # Combined mask
        combined_mask = np.maximum(object1_mask, object2_mask)

        # Save object1-only RGB (white background where mask is 0)
        object1_rgb = image_np.copy()
        object1_rgb[object1_mask == 0] = 1.0  # White background matching bg_color
        object1_rgb_pil = Image.fromarray((object1_rgb * 255).astype(np.uint8))
        object1_rgb_pil.save(output_dir / f"{view_name}_{self.object1_name}_rgb.png")

        # Save object1 mask
        object1_mask_pil = Image.fromarray((object1_mask * 255).astype(np.uint8), mode='L')
        object1_mask_pil.save(output_dir / f"{view_name}_{self.object1_name}_mask.png")

        # Save object2-only RGB (white background where mask is 0 - for Phase 2 training)
        object2_rgb = image_np.copy()
        object2_rgb[object2_mask == 0] = 1.0  # White background for object2
        object2_rgb_pil = Image.fromarray((object2_rgb * 255).astype(np.uint8))
        object2_rgb_pil.save(output_dir / f"{view_name}_{self.object2_name}_rgb.png")

        # Save object2 mask
        object2_mask_pil = Image.fromarray((object2_mask * 255).astype(np.uint8), mode='L')
        object2_mask_pil.save(output_dir / f"{view_name}_{self.object2_name}_mask.png")

        # Save combined mask
        combined_mask_pil = Image.fromarray((combined_mask * 255).astype(np.uint8), mode='L')
        combined_mask_pil.save(output_dir / f"{view_name}_combined_mask.png")

        logger.info(f"  Saved segmented images for {view_name}")


# ==================== Unused Image Processing Functions (Commented) ====================

# @staticmethod
# def get_foreground_bbox(image, threshold=240):
#     """
#     Get bounding box of foreground (non-white) region in image
#
#     Args:
#         image: PIL Image or numpy array
#         threshold: pixel value threshold to consider as white (0-255)
#
#     Returns:
#         (ymin, ymax, xmin, xmax) or None if no foreground found
#     """
#     # Convert to numpy if needed
#     if isinstance(image, Image.Image):
#         img_array = np.array(image)
#     else:
#         img_array = image
#
#     # Create mask for non-white pixels
#     if img_array.shape[-1] == 4:
#         rgb = img_array[:, :, :3]
#         alpha = img_array[:, :, 3]
#         mask = ~((rgb[:, :, 0] >= threshold) &
#                  (rgb[:, :, 1] >= threshold) &
#                  (rgb[:, :, 2] >= threshold) &
#                  (alpha >= threshold))
#     else:
#         mask = ~((img_array[:, :, 0] >= threshold) &
#                  (img_array[:, :, 1] >= threshold) &
#                  (img_array[:, :, 2] >= threshold))
#
#     # Find bounding box
#     rows = np.any(mask, axis=1)
#     cols = np.any(mask, axis=0)
#
#     if not np.any(rows) or not np.any(cols):
#         return None
#
#     ymin, ymax = np.where(rows)[0][[0, -1]]
#     xmin, xmax = np.where(cols)[0][[0, -1]]
#
#     return (ymin, ymax, xmin, xmax)


# @staticmethod
# def align_multiview_to_reference(multiview_image, reference_image, target_size=(1024, 1024), threshold=240, align_mode='bottom'):
#     """
#     Align multiview image to reference image with better physical constraints
#
#     Supports multiple alignment modes:
#     - 'height': Match foreground height only (original behavior)
#     - 'bottom': Match height and align bottom edges (for standing objects)
#     - 'center': Match height and align centers (for centered objects)
#
#     Args:
#         multiview_image: PIL Image of multiview render
#         reference_image: PIL Image of reference
#         target_size: (width, height) to resize final image to
#         threshold: white background threshold
#         align_mode: 'height', 'bottom', or 'center'
#
#     Returns:
#         Aligned and resized PIL Image
#     """
#     # Note: This function uses get_foreground_bbox which is also commented out
#     pass


# def compute_center_of_mass(self, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
#     """
#     Compute soft center of mass (centroid) of a mask
#     Uses continuous mask values as weights to maintain gradients
#
#     Args:
#         mask: [H, W, 1] or [H, W] mask (continuous values in [0, 1])
#         threshold: unused (kept for API compatibility)
#
#     Returns:
#         center: [2] tensor with (y, x) normalized coordinates in [0, 1]
#     """
#     # Ensure 2D
#     if mask.dim() == 3:
#         mask = mask.squeeze(-1)
#
#     # Get dimensions
#     H, W = mask.shape
#
#     # Create coordinate grids
#     y_coords = torch.arange(H, dtype=torch.float32, device=mask.device)
#     x_coords = torch.arange(W, dtype=torch.float32, device=mask.device)
#
#     # Compute weighted sum using continuous mask values (differentiable)
#     total_mass = mask.sum() + 1e-8
#
#     y_center = (mask.sum(dim=1) * y_coords).sum() / total_mass
#     x_center = (mask.sum(dim=0) * x_coords).sum() / total_mass
#
#     # Normalize to [0, 1]
#     y_center_norm = y_center / H
#     x_center_norm = x_center / W
#
#     return torch.stack([y_center_norm, x_center_norm])


# def prepare_reference_image(self, reference_image_path: str):
#     """Process and segment reference image (150° view only) - NO multi-view generation
#
#     Args:
#         reference_image_path: Path to reference image (reference_edited.png)
#
#     Returns:
#         reference_dir: Directory containing processed reference image
#     """
#     logger.info("=== Processing Reference Image (Single View Only) ===")
#
#     # Create reference output directory
#     reference_dir = Path(reference_image_path).parent / "reference"
#     reference_dir.mkdir(exist_ok=True, parents=True)
#
#     # Load reference image
#     logger.info(f"Loading reference image: {reference_image_path}")
#     reference_image = Image.open(reference_image_path).convert("RGB")
#     original_size = reference_image.size
#     logger.info(f"Original size: {original_size}")
#
#     # Resize to 1024x1024 for consistency
#     reference_image = reference_image.resize((1024, 1024), Image.LANCZOS)
#     logger.info(f"Resized to: {reference_image.size}")
#
#     # Segment reference image (150°) with SAM2
#     logger.info("\n=== Segmenting reference image with SAM2 ===")
#     logger.info("Segmenting reference image (150° view)...")
#     reference_masks = self.mv_ref_generator.segment_reference_with_sam2(reference_image, tags=[self.object1_name, self.object2_name])
#
#     # Save segmented view
#     view_name = "view_150_reference"
#     self._save_segmented_view(reference_image, reference_masks, reference_dir, view_name)
#
#     # Also save the reference image itself
#     reference_image.save(reference_dir / f"{view_name}.png")
#     logger.info(f"Saved reference image to {view_name}.png")
#
#     logger.info("\n" + "="*60)
#     logger.info("Reference image processing complete!")
#     logger.info(f"Processed 1 view: 150° reference view")
#     logger.info(f"Saved to: {reference_dir}")
#     logger.info("="*60)
#
#     return str(reference_dir)
