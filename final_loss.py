"""
Final Loss Module - Centralized loss computation for SDS-MV-Mesh

This module contains all loss computation functions extracted from sds_mv_mesh.py
to provide a clean, reusable interface for computing various losses including:
- IoU loss
- Reconstruction loss (RGB + Mask)
- ARAP loss
- Perceptual loss (LPIPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from loguru import logger


class LossComputer:
    """
    Centralized loss computer for mesh optimization

    This class provides methods to compute various losses used in the optimization:
    - IoU loss for mask alignment
    - Reconstruction loss (L1 + LPIPS for RGB, MSE for mask)
    - ARAP loss for local deformation regularization
    """

    def __init__(self, device='cuda', bg_color=[1.0, 1.0, 1.0],
                 use_scale_normalization=False):
        """
        Initialize loss computer

        Args:
            device: Device to run computations on
            bg_color: Background color for padding operations [R, G, B] in [0, 1]
            use_scale_normalization: Whether to use scale normalization in reconstruction loss
        """
        self.device = device
        self.bg_color = bg_color
        self.use_scale_normalization = use_scale_normalization

        # Initialize LPIPS for perceptual loss
        logger.info("Loading LPIPS for perceptual loss...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)  # alex is faster than vgg
        logger.info("LPIPS loaded!")

        # DreamSim model (lazy loading - only load when needed)
        self.dreamsim_model = None
        self.dreamsim_preprocess = None

        # Cache for ARAP connectivity (computed once per mesh)
        self._arap_connectivity = None

    def compute_iou(self, mask1: torch.Tensor, mask2: torch.Tensor,
                   threshold: float = 0.5) -> torch.Tensor:
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

    def _load_dreamsim_model(self):
        """
        Lazy load DreamSim model (only load once when first needed)
        """
        if self.dreamsim_model is None:
            logger.info("Loading DreamSim model for perceptual similarity...")
            try:
                # Import DreamSim (path should be added to sys.path in main script)
                from dreamsim import dreamsim
                from torchvision import transforms

                # Load model (ensemble: dino_vitb16 + clip_vitb16 + open_clip_vitb16)
                # Cache directory: /data4/zishuo/dreamsim_models
                self.dreamsim_model, _ = dreamsim(
                    pretrained=True,
                    device=self.device,
                    cache_dir="/data4/zishuo/dreamsim_models",
                    dreamsim_type="ensemble"  # Best performance: 96.9% NIGHTS val
                )

                # Create preprocessing transform (resize to 224x224 as required by DreamSim)
                img_size = 224
                self.dreamsim_preprocess = transforms.Compose([
                    transforms.Resize((img_size, img_size),
                                    interpolation=transforms.InterpolationMode.BICUBIC),
                ])

                logger.info("DreamSim model loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load DreamSim model: {e}")
                raise

    def compute_dreamsim_loss(self, rendered_rgb: torch.Tensor, gt_rgb: torch.Tensor):
        """
        Compute DreamSim perceptual similarity loss

        DreamSim is a learned perceptual similarity metric based on human judgments.
        It returns distance (1 - cosine_similarity) which is differentiable.

        Args:
            rendered_rgb: [H, W, 3] rendered RGB image in range [0, 1]
            gt_rgb: [H, W, 3] ground truth RGB image in range [0, 1]

        Returns:
            loss: Scalar tensor with DreamSim perceptual distance
        """
        # Lazy load DreamSim model
        self._load_dreamsim_model()

        # Convert from [H, W, 3] to [1, 3, H, W] format
        rendered_chw = rendered_rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        gt_chw = gt_rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # Resize to 224x224 (DreamSim requirement)
        rendered_resized = self.dreamsim_preprocess(rendered_chw)  # [1, 3, 224, 224]
        gt_resized = self.dreamsim_preprocess(gt_chw)  # [1, 3, 224, 224]

        # Compute DreamSim distance (already returns distance, not similarity)
        # Distance = 1 - cosine_similarity, range [0, 2], lower is better
        distance = self.dreamsim_model(rendered_resized, gt_resized)

        # Return mean distance (should be scalar)
        return distance.mean()

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

        image_resized = F.interpolate(image_BCHW, size=(new_H, new_W),
                                     mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask_BCHW, size=(new_H, new_W),
                                    mode='bilinear', align_corners=False)

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

        return image_normalized, mask_normalized, scale_factor

    def compute_reconstruction_loss_rgb_only(self, rendered_rgb: torch.Tensor,
                                            gt_rgb: torch.Tensor,
                                            use_mse: bool = False):
        """
        Compute RGB-only reconstruction loss using L1 + LPIPS (no mask)

        Args:
            rendered_rgb: [H, W, 3] rendered RGB image in range [0, 1]
            gt_rgb: [H, W, 3] ground truth RGB
            use_mse: If True, use MSE for RGB loss; otherwise use L1 + LPIPS (default: False)

        Returns:
            loss_rgb: RGB reconstruction loss (L1 + LPIPS)
        """
        # Convert to [C, H, W] format for LPIPS
        rendered_chw = rendered_rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        gt_chw = gt_rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # L1 loss (pixel-wise difference)
        loss_l1 = F.l1_loss(rendered_rgb, gt_rgb)

        # LPIPS loss (perceptual similarity, needs range [-1, 1])
        rendered_lpips = rendered_chw * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        gt_lpips = gt_chw * 2.0 - 1.0
        loss_lpips = self.lpips_fn(rendered_lpips, gt_lpips).mean()

        # Combine L1 and LPIPS (DGE style: L1 + LPIPS)
        # L1: 0.2, LPIPS: 0.8 (LPIPS is more important for perceptual quality)
        loss_rgb = 0.2 * loss_l1 + 0.8 * loss_lpips

        return loss_rgb

    def compute_reconstruction_loss(self, rendered_rgb: torch.Tensor,
                                   rendered_mask: torch.Tensor,
                                   gt_rgb: torch.Tensor,
                                   gt_mask: torch.Tensor,
                                   use_mse: bool = False,
                                   use_mask_weighting: bool = False):
        """
        Compute reconstruction loss using L1 + LPIPS (DGE style)
        With scale normalization to handle foreground size mismatches

        Args:
            rendered_rgb: [H, W, 3] rendered RGB image in range [0, 1]
            rendered_mask: [H, W, 1] rendered mask/alpha in range [0, 1]
            gt_rgb: [H, W, 3] ground truth RGB (original, FLUX keeps white background)
            gt_mask: [H, W, 1] ground truth mask
            use_mse: If True, use MSE for RGB loss; otherwise use L1 + LPIPS (default: False)
            use_mask_weighting: If True, use gt_mask to weight RGB loss (only compute loss in foreground)

        Returns:
            loss_rgb: RGB reconstruction loss (L1 + LPIPS, following DGE)
            loss_mask: Mask reconstruction loss (MSE)
        """
        # Scale normalization: normalize rendered image to match GT foreground size
        if self.use_scale_normalization:
            # Compute target scale from GT
            gt_mask_binary = (gt_mask > 0.5).squeeze(-1)
            if gt_mask_binary.any():
                rows = torch.any(gt_mask_binary, dim=1)
                cols = torch.any(gt_mask_binary, dim=0)
                ymin, ymax = torch.where(rows)[0][[0, -1]]
                xmin, xmax = torch.where(cols)[0][[0, -1]]
                gt_fg_height = (ymax - ymin + 1).float()
                gt_fg_width = (xmax - xmin + 1).float()
                gt_fg_size = max(gt_fg_height, gt_fg_width)
                H, W = gt_mask.shape[:2]
                target_scale = (gt_fg_size / min(H, W)).item()

                # Normalize rendered image to match GT scale
                rendered_rgb, rendered_mask, _scale_factor = self.normalize_foreground_scale(
                    rendered_rgb, rendered_mask, target_scale=target_scale
                )

        # RGB reconstruction loss
        # L1 + LPIPS (following DGE implementation)
        # Convert to [C, H, W] format for LPIPS
        rendered_chw = rendered_rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        gt_chw = gt_rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # L1 loss (pixel-wise difference)
        loss_l1 = F.l1_loss(rendered_rgb, gt_rgb)

        # LPIPS loss (perceptual similarity, needs range [-1, 1])
        rendered_lpips = rendered_chw * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        gt_lpips = gt_chw * 2.0 - 1.0
        loss_lpips = self.lpips_fn(rendered_lpips, gt_lpips).mean()

        # Combine L1 and LPIPS (DGE style: L1 + LPIPS)
        # L1: 0.2, LPIPS: 0.8 (LPIPS is more important for perceptual quality)
        loss_rgb = 0.2 * loss_l1 + 0.8 * loss_lpips

        # Mask reconstruction loss (MSE is fine for binary mask)
        loss_mask = F.mse_loss(rendered_mask, gt_mask)

        return loss_rgb, loss_mask

    def compute_arap_loss(self, man_xyz_init, man_delta_xyz, man_aabb,
                         generated_parts, connectivity_cache=None):
        """
        Compute ARAP loss for local deformation using mesh topology

        IMPORTANT:
        - Only compute ARAP for the MAN's gaussian, NOT the chair!
        - ARAP should ONLY constrain LOCAL deformation (delta-based)
        - ARAP should NOT be affected by global pose changes (translation, quaternion, scale)
        - Uses MESH TOPOLOGY for accurate connectivity (not K-NN approximation)

        We compare:
        - Initial: man_xyz_init (fixed reference, in local space)
        - Current: man_xyz_init + man_delta_xyz (trainable deformation, in local space)

        Args:
            man_xyz_init: [N, 3] initial normalized xyz positions (fixed reference)
            man_delta_xyz: [N, 3] trainable delta xyz positions
            man_aabb: [6] axis-aligned bounding box [min_x, min_y, min_z, size_x, size_y, size_z]
            generated_parts: List of generated objects with connectivity info
            connectivity_cache: Optional tuple (ii, jj, nn, weight, K) to reuse connectivity

        Returns:
            arap_error: ARAP loss value
        """
        # Import ARAP utilities from mesh_guidance
        from mesh_guidance import cal_arap_error

        # Initial positions (fixed reference in local space)
        initial_xyz_normalized = man_xyz_init.clone().detach()  # [N, 3]
        initial_xyz = initial_xyz_normalized * man_aabb[3:] + man_aabb[:3]  # World coords

        # Current deformed positions (init + delta, trainable, in same local space)
        current_xyz_normalized = man_xyz_init + man_delta_xyz  # [N, 3] - trainable delta
        current_xyz = current_xyz_normalized * man_aabb[3:] + man_aabb[:3]  # World coords

        # Verify dimensions
        n_man = initial_xyz.shape[0]
        assert current_xyz.shape[0] == n_man, \
            f"Expected {n_man} man gaussians, got {current_xyz.shape[0]}"

        # Stack as sequence: [2, N, 3] - initial (t=0) and current (t=1)
        nodes_sequence = torch.stack([initial_xyz, current_xyz], dim=0)

        # Calculate connectivity using MESH TOPOLOGY (if available)
        if connectivity_cache is None:
            with torch.no_grad():
                # Check if we have mesh topology from generate_objects()
                man_obj = generated_parts[0]  # First object is the man
                if 'connectivity' in man_obj:
                    # Use mesh topology (RECOMMENDED - 3x more efficient)
                    logger.info("Using MESH TOPOLOGY for ARAP connectivity (accurate)")
                    connectivity_dict = man_obj['connectivity']
                    K = 3  # Use top-3 neighbors for efficiency

                    # Convert connectivity dict to index arrays
                    ii_list, jj_list, nn_list, weight_list = [], [], [], []
                    for i, neighbors in connectivity_dict.items():
                        # Get distances to all neighbors
                        neighbor_items = sorted(neighbors.items(),
                                              key=lambda x: x[1])[:K]  # Sort by distance, take top K
                        for n_idx, (j, dist) in enumerate(neighbor_items):
                            ii_list.append(i)
                            jj_list.append(j)
                            nn_list.append(n_idx)
                            weight_list.append(1.0 / (dist + 1e-6))  # Distance-based weight

                    ii = torch.tensor(ii_list, dtype=torch.long).cuda()
                    jj = torch.tensor(jj_list, dtype=torch.long).cuda()
                    nn = torch.tensor(nn_list, dtype=torch.long).cuda()

                    # Normalize weights per vertex
                    weight_tensor = torch.zeros(n_man, K).cuda()
                    for idx in range(len(ii_list)):
                        weight_tensor[ii_list[idx], nn_list[idx]] = weight_list[idx]
                    # Normalize
                    weight_sum = weight_tensor.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    weight = weight_tensor / weight_sum

                    logger.info(f"  ARAP using mesh topology: {len(ii)} edges, K={K} neighbors")

                else:
                    # Fallback to K-NN (if mesh topology not available)
                    logger.warning("Mesh topology not found, falling back to K-NN connectivity "
                                 "(less accurate)")
                    from mesh_guidance import cal_connectivity_from_points

                    radius = 0.1
                    K = 10

                    ii, jj, nn, weight = cal_connectivity_from_points(
                        nodes_sequence[0:1],  # Use initial frame for connectivity
                        radius=radius,
                        K=K
                    )

                connectivity_cache = (ii, jj, nn, weight, K)

        ii, jj, nn, weight, K = connectivity_cache

        # Calculate ARAP error using mesh_guidance module (measures local rigidity)
        arap_error = cal_arap_error(
            nodes_sequence,
            ii, jj, nn,
            K=K,
            weight=weight,
            sample_num=512
        )

        return arap_error, connectivity_cache

    def sample_random_views_around_reference(self, reference_azimuth, num_views=4,
                                            angle_range=90, distribution='uniform_random'):
        """
        Sample random camera views around the reference view

        Useful for SDXL reconstruction loss with multiple random views

        Args:
            reference_azimuth: Reference azimuth angle in degrees
            num_views: Number of views to sample
            angle_range: Angle range in degrees (±range around reference)
            distribution: 'uniform_random' or 'pure_random'

        Returns:
            azimuths: List of absolute azimuth angles in degrees
            delta_azimuths: List of relative angles to reference in degrees
        """
        import numpy as np

        if distribution == 'uniform_random':
            # Uniformly distributed + random perturbation
            base_deltas = np.linspace(-angle_range, angle_range, num_views)

            # Add random noise (30% of interval)
            interval = (2 * angle_range) / (num_views - 1) if num_views > 1 else angle_range
            noise_scale = interval * 0.3
            noise = np.random.uniform(-noise_scale, noise_scale, num_views)

            delta_azimuths = base_deltas + noise

        elif distribution == 'pure_random':
            # Pure random sampling
            delta_azimuths = np.random.uniform(-angle_range, angle_range, num_views)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Convert to absolute angles
        azimuths = [(reference_azimuth + delta) % 360 for delta in delta_azimuths]

        return azimuths, delta_azimuths