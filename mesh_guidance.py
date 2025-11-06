#!/usr/bin/env python3
"""
Mesh-Guided Gaussian Optimization
Converts TRELLIS mesh to topology-aware Gaussian and aligns it with TRELLIS Gaussian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pytorch3d.ops import knn_points, knn_gather
from trellis.representations import Gaussian

logger = logging.getLogger(__name__)


# ==================== SSIM Loss (from 3D Gaussian Splatting) ====================

def gaussian_kernel(size: int = 11, sigma: float = 1.5):
    """Create 2D Gaussian kernel"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = coords**2
    g = (-g / (2 * sigma**2)).exp()
    g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
         size_average: bool = True) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity Index) between two images
    Following 3D Gaussian Splatting paper implementation

    Args:
        img1, img2: [B, C, H, W] or [H, W, C] images
        window_size: Gaussian window size (default 11)
        size_average: if True, return mean SSIM; else return SSIM map

    Returns:
        SSIM value (scalar if size_average=True, else [B, C, H, W])
    """
    # Handle [H, W, C] input
    if img1.dim() == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        img2 = img2.permute(2, 0, 1).unsqueeze(0)

    C = img1.shape[1]

    # Create Gaussian window
    window = gaussian_kernel(window_size, 1.5).to(img1.device)
    window = window.expand(C, 1, window_size, window_size)

    # Constants for stability
    C1 = 0.01**2
    C2 = 0.03**2

    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=C) - mu1_mu2

    # SSIM formula
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

def d_ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute D-SSIM loss (1 - SSIM)
    Used in 3D Gaussian Splatting paper
    """
    return 1.0 - ssim(img1, img2, window_size=11, size_average=True)


# ==================== ARAP Utilities (from Animate3D) ====================

def produce_edge_matrix_nfmt(verts: torch.Tensor, edge_shape, ii, jj, nn, device="cuda") -> torch.Tensor:
    """
    Given a tensor of verts position, p (V x 3), produce a tensor E, where, for neighbour list J,
    E_in = p_i - p_(J[n])

    Args:
        verts: [Nv, 3] vertex positions
        edge_shape: (Nv, K, 3) shape of edge matrix
        ii: [Ne] source vertex indices
        jj: [Ne] target vertex indices
        nn: [Ne] neighbor order indices
        device: cuda or cpu

    Returns:
        E: [Nv, K, 3] edge vectors
    """
    E = torch.zeros(edge_shape).to(device)
    E[ii, nn] = verts[ii] - verts[jj]
    return E


def estimate_rotation(source: torch.Tensor, target: torch.Tensor, ii, jj, nn,
                      K: int = 10, weight: Optional[torch.Tensor] = None,
                      sample_idx: Optional[torch.Tensor] = None):
    """
    Estimate per-vertex rotation using SVD

    Args:
        source: [Nv, 3] source vertex positions
        target: [Nv, 3] target vertex positions
        ii, jj, nn: edge indices
        K: number of neighbors
        weight: [Nv, K] edge weights
        sample_idx: optional vertex sampling indices

    Returns:
        R: [Nv, 3, 3] rotation matrices
    """
    Nv = len(source)
    source_edge_mat = produce_edge_matrix_nfmt(source, (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    target_edge_mat = produce_edge_matrix_nfmt(target, (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]

    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
        logger.warning("Edge weight is None, using uniform weights")

    if sample_idx is not None:
        source_edge_mat = source_edge_mat[sample_idx]
        target_edge_mat = target_edge_mat[sample_idx]

    # Calculate covariance matrix in bulk
    D = torch.diag_embed(weight, dim1=1, dim2=2)  # [Nv, K, K]
    S = torch.bmm(source_edge_mat.permute(0, 2, 1), torch.bmm(D, target_edge_mat))  # [Nv, 3, 3]

    # In the case of no deflection, set S = 0, such that R = I
    unchanged_verts = torch.unique(torch.where((source_edge_mat == target_edge_mat).all(dim=1))[0])
    S[unchanged_verts] = 0

    # SVD
    U, sig, W = torch.svd(S)
    R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations

    # Handle determinant corrections (ensure proper rotations)
    entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()
    if len(entries_to_flip) > 0:
        Umod = U.clone()
        cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)
        Umod[entries_to_flip, :, cols_to_flip] *= -1
        R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))

    return R


def cal_arap_error(nodes_sequence: torch.Tensor, ii, jj, nn, K: int = 10,
                   weight: Optional[torch.Tensor] = None, sample_num: int = 512):
    """
    Calculate ARAP (As-Rigid-As-Possible) error

    Args:
        nodes_sequence: [Nt, Nv, 3] vertex positions across time
        ii, jj, nn: edge indices
        K: number of neighbors
        weight: [Nv, K] edge weights
        sample_num: number of vertices to sample for efficiency

    Returns:
        arap_error: scalar ARAP loss
    """
    Nt, Nv, _ = nodes_sequence.shape
    arap_error = 0

    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1

    source_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[0], (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]

    # Sample vertices for efficiency
    sample_idx = torch.arange(Nv).cuda()
    if Nv > sample_num:
        sample_idx = torch.from_numpy(np.random.choice(Nv, sample_num, replace=False)).long().cuda()
    else:
        source_edge_mat = source_edge_mat[sample_idx]
    weight = weight[sample_idx]

    for idx in range(1, Nt):
        with torch.no_grad():
            rotation = estimate_rotation(
                nodes_sequence[0], nodes_sequence[idx], ii, jj, nn,
                K=K, weight=weight, sample_idx=sample_idx
            )  # [Nv, 3, 3]

        # Compute energy
        target_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[idx], (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
        target_edge_mat = target_edge_mat[sample_idx]
        rot_rigid = torch.bmm(rotation, source_edge_mat[sample_idx].permute(0, 2, 1)).permute(0, 2, 1)  # [Nv, K, 3]
        stretch_vec = target_edge_mat - rot_rigid  # stretch vector
        stretch_norm = (torch.norm(stretch_vec, dim=2) ** 2)  # norm over (x,y,z) space
        arap_error += (weight * stretch_norm).sum()

    return arap_error


def cal_connectivity_from_points(points: torch.Tensor, radius: float = 0.1, K: int = 10,
                                 trajectory: Optional[torch.Tensor] = None,
                                 least_edge_num: int = 3, node_radius: Optional[torch.Tensor] = None,
                                 mode: str = 'nn', GraphK: int = 4, adaptive_weighting: bool = True):
    """
    Calculate K-NN connectivity from point cloud

    Args:
        points: [T, Nv, 3] or [1, Nv, 3] point positions
        radius: connection radius threshold
        K: number of neighbors
        trajectory: optional trajectory for connectivity
        least_edge_num: minimum guaranteed edges per vertex
        node_radius: optional per-node radius
        mode: 'nn' for nearest neighbor, 'floyd' for geodesic
        GraphK: K for graph construction in floyd mode
        adaptive_weighting: use distance-based adaptive weighting

    Returns:
        ii: [Ne] source vertex indices
        jj: [Ne] target vertex indices
        nn: [Ne] neighbor order indices
        weight: [Nv, K] edge weights
    """
    Nv = points.shape[1] if points is not None else trajectory.shape[0]

    if trajectory is None:
        # Query the first frame
        knn_res = knn_points(points[0:1], points[0:1], None, None, K=K+1)
        # Remove themselves
        nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]  # [Nv, K]

        # Query the rest frames (if multiple frames)
        if points.shape[0] > 1:
            rest_knn_pts = knn_gather(points[1:], nn_idx.unsqueeze(0).repeat(points.shape[0]-1, 1, 1))
            rest_nn_dist = ((rest_knn_pts - points[0:1][:, :, None]) ** 2).sum(-1)
            # Modify nn_dist: only keep if within radius in all frames
            nn_dist = torch.where(
                (rest_nn_dist < radius ** 2).all(0),
                nn_dist,
                torch.ones_like(nn_dist) * torch.inf
            )
    else:
        # Use trajectory for connectivity
        trajectory = trajectory.reshape([Nv, -1]) / trajectory.shape[1]  # Average distance
        knn_res = knn_points(trajectory[None], trajectory[None], None, None, K=K+1)
        nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]  # [Nv, K]

    # Make sure ranges are within the radius (but keep at least least_edge_num edges)
    nn_idx[:, least_edge_num:] = torch.where(
        nn_dist[:, least_edge_num:] < radius ** 2,
        nn_idx[:, least_edge_num:],
        -torch.ones_like(nn_idx[:, least_edge_num:])
    )
    nn_dist[:, least_edge_num:] = torch.where(
        nn_dist[:, least_edge_num:] < radius ** 2,
        nn_dist[:, least_edge_num:],
        torch.ones_like(nn_dist[:, least_edge_num:]) * torch.inf
    )

    # Calculate weights
    if adaptive_weighting:
        weight = torch.exp(-nn_dist / nn_dist.mean())
    elif node_radius is None:
        weight = torch.exp(-nn_dist)
    else:
        nn_radius = node_radius[nn_idx]
        weight = torch.exp(-nn_dist / (2 * nn_radius ** 2))
    weight = weight / weight.sum(dim=-1, keepdim=True)

    # Build edge lists
    ii = torch.arange(Nv)[:, None].cuda().long().expand(Nv, K).reshape([-1])
    jj = nn_idx.reshape([-1])
    nn = torch.arange(K)[None].cuda().long().expand(Nv, K).reshape([-1])
    mask = jj != -1
    ii, jj, nn = ii[mask], jj[mask], nn[mask]

    return ii, jj, nn, weight


# ==================== Mesh to Gaussian Conversion ====================

class MeshToGaussianConverter:
    """
    Converts mesh to Gaussian with topology information
    Similar to Animate3D's mesh2gaussian.py but integrated with TRELLIS
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def extract_mesh_connectivity(self, vertices: torch.Tensor, faces: torch.Tensor) -> Dict[int, Dict[int, float]]:
        """
        Extract connectivity information from mesh faces

        Args:
            vertices: [N, 3] vertex positions
            faces: [M, 3] face indices

        Returns:
            connectivity_dict: {vertex_id: {neighbor_id: distance, ...}, ...}
        """
        N = vertices.shape[0]
        M = faces.shape[0]

        logger.info(f"Extracting connectivity from {N} vertices and {M} faces...")

        # Build adjacency dictionary
        adj_dict = {i: {} for i in range(N)}

        for face in faces:
            v0, v1, v2 = face[0].item(), face[1].item(), face[2].item()

            # Three edges in the triangle
            edges = [(v0, v1), (v1, v2), (v2, v0)]

            for vi, vj in edges:
                distance = torch.norm(vertices[vi] - vertices[vj]).item()
                adj_dict[vi][vj] = distance
                adj_dict[vj][vi] = distance

        # Statistics
        edge_counts = [len(neighbors) for neighbors in adj_dict.values()]
        logger.info(f"Connectivity stats: mean={np.mean(edge_counts):.1f}, "
                   f"min={np.min(edge_counts)}, max={np.max(edge_counts)} edges per vertex")

        return adj_dict

    def compute_edge_lengths(self, vertices: torch.Tensor, connectivity_info: Dict[int, Dict[int, float]]) -> torch.Tensor:
        """
        Compute average edge length for each vertex (used for Gaussian scale)

        Args:
            vertices: [N, 3] vertex positions
            connectivity_info: connectivity dictionary

        Returns:
            edge_lengths: [N, 3] mean edge vector (dx, dy, dz) per vertex
        """
        N = vertices.shape[0]
        edge_lengths = torch.zeros(N, 3).to(self.device)

        for i in range(N):
            neighbors = list(connectivity_info[i].keys())
            if len(neighbors) == 0:
                # Isolated vertex, use small default
                edge_lengths[i] = torch.tensor([0.01, 0.01, 0.01]).to(self.device)
                continue

            vecs = []
            for j in neighbors:
                vec = vertices[j] - vertices[i]
                vecs.append(vec.abs())  # [|dx|, |dy|, |dz|]

            # Average edge vector
            mean_vec = torch.stack(vecs).mean(dim=0)
            edge_lengths[i] = mean_vec

        return edge_lengths

    def mesh_to_gaussian_params(self, mesh_trellis, scale_factor: float = 1.0) -> Tuple[Dict[str, torch.Tensor], Dict[int, Dict[int, float]]]:
        """
        Convert TRELLIS mesh to Gaussian parameters with topology

        Args:
            mesh_trellis: TRELLIS Mesh object
            scale_factor: scale adjustment for Gaussian sizes (default 1.0, Animate3D uses 1/1.1≈0.91)

        Returns:
            gaussian_params: dict with 'positions', 'scales', 'rotations', 'opacities', 'sh_features'
            connectivity_info: topology dictionary
        """
        logger.info("Converting mesh to Gaussian with topology...")

        # Extract mesh data
        if hasattr(mesh_trellis, 'vertices') and hasattr(mesh_trellis, 'faces'):
            # Direct access
            vertices = torch.tensor(mesh_trellis.vertices, dtype=torch.float32).to(self.device)
            faces = torch.tensor(mesh_trellis.faces, dtype=torch.int64).to(self.device)
        elif hasattr(mesh_trellis, 'to_trimesh'):
            # Convert to trimesh first
            import trimesh
            mesh = mesh_trellis.to_trimesh()
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device)
            faces = torch.tensor(mesh.faces, dtype=torch.int64).to(self.device)
        else:
            raise ValueError(f"Unknown mesh type: {type(mesh_trellis)}")

        N = vertices.shape[0]
        logger.info(f"Mesh has {N} vertices")

        # 1. Extract connectivity (for ARAP)
        connectivity_info = self.extract_mesh_connectivity(vertices, faces)

        # 2. Compute edge lengths (for Gaussian scales)
        edge_lengths = self.compute_edge_lengths(vertices, connectivity_info)

        # 3. Initialize Gaussian parameters (similar to mesh2gaussian.py)
        # Add epsilon before log to prevent -inf for zero edge lengths
        # Use adaptive epsilon based on edge length distribution
        edge_lengths_scaled = edge_lengths * scale_factor
        eps = torch.clamp(edge_lengths_scaled[edge_lengths_scaled > 0].min() * 0.1, max=1e-6)  # 10% of minimum non-zero value
        edge_lengths_safe = edge_lengths_scaled + eps

        # CRITICAL: Account for Gaussian's scaling_bias (0.01)
        # Gaussian.get_scaling = exp(_scaling + log(scaling_bias))
        # So we need: _scaling = log(desired_scale) - log(scaling_bias)
        scaling_bias = 0.01
        scales_log = torch.log(edge_lengths_safe) - torch.log(torch.tensor(scaling_bias))

        # CRITICAL: opacity initialization needs to account for opacity_bias
        # With opacity_bias=0.1, bias in logit space = inverse_sigmoid(0.1) ≈ -2.2
        # To get actual opacity ≈ 0.8: need _opacity such that sigmoid(_opacity - 2.2) ≈ 0.8
        # inverse_sigmoid(0.8) ≈ 1.4, so _opacity ≈ 1.4 + 2.2 = 3.6
        initial_opacity_logit = 3.6  # Will give actual opacity ≈ 0.8 after adding opacity_bias

        gaussian_params = {
            'positions': vertices,  # [N, 3]
            'scales': scales_log,  # [N, 3] (log space, adjusted for scaling_bias)
            'rotations': torch.tensor([[1, 0, 0, 0]] * N, dtype=torch.float32).to(self.device),  # [N, 4] unit quaternions
            'opacities': torch.ones(N, 1).to(self.device) * initial_opacity_logit,  # [N, 1] logit space
            # Initialize colors to small random values (not 0) for better gradient flow
            'sh_features': torch.randn(N, 3, 16).to(self.device) * 0.01,  # [N, 3, 16] small random initialization
        }

        logger.info(f"Initialized Gaussian with {N} points (no color)")
        logger.info(f"  Position range: [{vertices.min().item():.3f}, {vertices.max().item():.3f}]")
        logger.info(f"  Scale range: [{edge_lengths.min().item():.4f}, {edge_lengths.max().item():.4f}]")

        return gaussian_params, connectivity_info


# ==================== Gaussian Alignment Optimizer ====================

class GaussianAlignmentOptimizer:
    """
    Optimizes mesh-based Gaussian to align with TRELLIS Gaussian while preserving topology
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def align_to_target_with_rendering(self,
                                      gaussian_params: Dict[str, torch.Tensor],
                                      connectivity_info: Dict[int, Dict[int, float]],
                                      gaussian_trellis: Gaussian,
                                      num_views: int = 16,
                                      num_steps: int = 3000,
                                      resolution: int = 512,
                                      lr: float = 1e-3,
                                      lambda_dssim: float = 0.2,
                                      log_interval: int = 100) -> Tuple[Gaussian, Dict[int, Dict[int, float]]]:
        """
        Simple approach:
        - Use mesh vertices for positions (FIXED)
        - Initialize other params from TRELLIS (via KNN matching)
        - Optimize colors, scales, rotations, opacities with rendering loss

        Args:
            gaussian_params: initial Gaussian parameters from mesh
            connectivity_info: mesh topology (preserved automatically)
            gaussian_trellis: target TRELLIS Gaussian
            num_views: number of views to render
            num_steps: optimization steps
            resolution: rendering resolution
            lr: learning rate
            lambda_dssim: weight for D-SSIM loss
            log_interval: logging frequency

        Returns:
            optimized_gaussian: aligned Gaussian with topology
            connectivity_info: preserved topology
        """
        from trellis.renderers import GaussianRenderer
        from trellis.utils import render_utils
        from easydict import EasyDict as edict

        logger.info("=" * 60)
        logger.info("Mesh-guided Gaussian optimization (simplified)")
        logger.info(f"  Mesh vertices: {len(gaussian_params['positions'])}")
        logger.info(f"  TRELLIS Gaussians: {len(gaussian_trellis._xyz)}")
        logger.info(f"  Strategy: Fix positions, optimize colors/scales/rotations/opacities")
        logger.info(f"  Steps: {num_steps}, Views: {num_views}, Resolution: {resolution}")
        logger.info("=" * 60)

        # Step 1: Get mesh vertices in normalized space
        positions_real = gaussian_params['positions'].clone()  # [N, 3] in real space
        aabb = gaussian_trellis.aabb if hasattr(gaussian_trellis, 'aabb') else torch.tensor([-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]).to(self.device)
        positions_norm = (positions_real - aabb[:3]) / aabb[3:]  # Convert to [0, 1]

        logger.info(f"\nMesh positions: {len(positions_norm)} vertices")
        logger.info(f"  Real space: [{positions_real.min():.3f}, {positions_real.max():.3f}]")
        logger.info(f"  Normalized: [{positions_norm.min():.3f}, {positions_norm.max():.3f}]")

        # Step 2: Create Gaussian object with mesh geometry
        aabb_list = gaussian_trellis.aabb.detach().cpu().tolist() if isinstance(gaussian_trellis.aabb, torch.Tensor) else gaussian_trellis.aabb

        logger.info(f"\nCreating topology-aware Gaussian from mesh...")
        current_gaussian = Gaussian(
            aabb=aabb_list,
            sh_degree=0,
            mininum_kernel_size=0.0,
            scaling_bias=0.01,
            opacity_bias=0.1,  # CRITICAL: Use 0.1 (TRELLIS default), NOT 0.0 which causes -inf bias
            scaling_activation="exp",
            device=self.device
        )

        # Initialize parameters from mesh geometry and TRELLIS matching:
        # - positions: from mesh vertices (FIXED)
        # - scales: from mesh edge lengths (TRAINABLE)
        # - rotations: unit quaternions (TRAINABLE)
        # - opacities: high initial values (TRAINABLE)
        # - colors: from TRELLIS via KNN matching (TRAINABLE - better initialization)
        current_gaussian._xyz = positions_norm  # Fixed mesh positions

        # KNN matching to copy colors from TRELLIS Gaussian
        logger.info(f"  Matching colors from TRELLIS Gaussian via KNN...")
        from pytorch3d.ops import knn_points
        knn_result = knn_points(
            positions_norm.unsqueeze(0),
            gaussian_trellis._xyz.unsqueeze(0),
            K=1
        )
        nearest_idx = knn_result.idx[0, :, 0]  # [N]
        colors_matched = gaussian_trellis._features_dc[nearest_idx].clone()  # [N, 3, 1]
        logger.info(f"  Matched {len(nearest_idx)} colors from TRELLIS via KNN")

        current_gaussian._features_dc = colors_matched.requires_grad_(True)  # From TRELLIS via KNN
        current_gaussian._scaling = gaussian_params['scales'].clone().requires_grad_(True)  # From edge lengths
        current_gaussian._rotation = gaussian_params['rotations'].clone().requires_grad_(True)  # Unit quaternions
        current_gaussian._opacity = gaussian_params['opacities'].clone().requires_grad_(True)  # High values

        logger.info(f"  Initialized from mesh geometry and TRELLIS:")
        logger.info(f"    Positions: {len(positions_norm)} vertices (fixed)")
        logger.info(f"    Scales (log): [{current_gaussian._scaling.min():.3f}, {current_gaussian._scaling.max():.3f}] (from edge lengths)")
        logger.info(f"    Opacities (logit): [{current_gaussian._opacity.min():.3f}, {current_gaussian._opacity.max():.3f}]")

        # Verify actual opacity after bias
        with torch.no_grad():
            actual_opacity = torch.sigmoid(current_gaussian._opacity + current_gaussian.opacity_bias)
            logger.info(f"    Opacities (real, after bias): [{actual_opacity.min():.3f}, {actual_opacity.max():.3f}]")

        logger.info(f"    Colors: [{current_gaussian._features_dc.min():.3f}, {current_gaussian._features_dc.max():.3f}] (from TRELLIS KNN)")

        # Step 4: Setup optimizer
        optimizer = torch.optim.Adam([
            {'params': [current_gaussian._features_dc], 'lr': lr},
            {'params': [current_gaussian._scaling], 'lr': lr},
            {'params': [current_gaussian._rotation], 'lr': lr},
            {'params': [current_gaussian._opacity], 'lr': lr},
        ])

        # Step 5: Setup camera views
        from trellis.utils.random_utils import sphere_hammersley_sequence
        camera_views = []
        radius = 1.5
        for i in range(num_views):
            azimuth, elevation = sphere_hammersley_sequence(i, num_views)
            camera_views.append({'azimuth': azimuth, 'elevation': elevation, 'radius': radius})
        logger.info(f"\nCamera views: {num_views} views on sphere")

        # Step 6: Setup renderer
        renderer = GaussianRenderer()
        renderer.rendering_options = edict({
            'resolution': resolution,
            'near': 0.8,
            'far': 1.6,
            'bg_color': torch.tensor([0.85, 0.85, 0.85]).to(self.device),
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
        logger.info("Renderer initialized")

        # Step 7: Pre-render target TRELLIS images
        logger.info("\nPre-rendering target images from TRELLIS Gaussian...")
        target_images = []
        with torch.no_grad():
            for view in camera_views:
                extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                    yaws=float(view['azimuth']),
                    pitchs=float(view['elevation']),
                    rs=float(view['radius']),
                    fovs=60.0
                )
                result = renderer.render(gaussian_trellis, extrinsics, intrinsics)
                rendered_image = result['color'].permute(1, 2, 0)  # [H, W, 3]
                target_images.append(torch.clamp(rendered_image, 0.0, 1.0))
        logger.info(f"  Pre-rendered {len(target_images)} target images")

        logger.info("\n" + "=" * 60)
        logger.info("Starting optimization...")
        logger.info("=" * 60)

        # Create debug directory for intermediate visualizations
        import os
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_renders')
        os.makedirs(debug_dir, exist_ok=True)
        logger.info(f"Debug renders will be saved to: {debug_dir}")

        # Step 8: Optimization loop (using 4 views per step)
        num_views_per_step = 4  # Use 4 random views per optimization step
        for step in range(num_steps):
            optimizer.zero_grad()

            # Sample 4 random views
            view_indices = np.random.choice(num_views, size=num_views_per_step, replace=False)

            total_loss = 0.0
            total_l1 = 0.0
            total_dssim = 0.0

            for view_idx in view_indices:
                view = camera_views[view_idx]
                target_image = target_images[view_idx]

                # Setup camera
                extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                    yaws=float(view['azimuth']),
                    pitchs=float(view['elevation']),
                    rs=float(view['radius']),
                    fovs=60.0
                )

                # Render current Gaussian
                result = renderer.render(current_gaussian, extrinsics, intrinsics)
                rendered_image = result['color'].permute(1, 2, 0)  # [H, W, 3]
                rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

                # Loss: L1 + D-SSIM
                l1 = F.l1_loss(rendered_image, target_image)
                dssim = d_ssim_loss(rendered_image, target_image)
                loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim

                total_loss += loss
                total_l1 += l1
                total_dssim += dssim

            # Average loss over 4 views
            total_loss = total_loss / num_views_per_step
            total_l1 = total_l1 / num_views_per_step
            total_dssim = total_dssim / num_views_per_step

            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([
                current_gaussian._features_dc,
                current_gaussian._scaling,
                current_gaussian._rotation,
                current_gaussian._opacity
            ], max_norm=1.0)
            optimizer.step()

            # Normalize rotations after update
            with torch.no_grad():
                current_gaussian._rotation.data = F.normalize(current_gaussian._rotation.data, dim=1)

            # Logging
            if step % log_interval == 0:
                logger.info(f"Step {step:4d}/{num_steps}: loss={total_loss.item():.6f} | L1={total_l1.item():.6f} D-SSIM={total_dssim.item():.6f}")

            # Save debug renders at key steps
            if step in [0, 100, 500, 1000, 2500, 5000, num_steps-1]:
                logger.info(f"  Saving debug render at step {step}...")
                with torch.no_grad():
                    # Render from first view
                    debug_view = camera_views[0]
                    debug_extrinsics, debug_intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                        yaws=float(debug_view['azimuth']),
                        pitchs=float(debug_view['elevation']),
                        rs=float(debug_view['radius']),
                        fovs=60.0
                    )
                    debug_result = renderer.render(current_gaussian, debug_extrinsics, debug_intrinsics)
                    debug_image = debug_result['color'].permute(1, 2, 0)
                    debug_image = torch.clamp(debug_image, 0.0, 1.0)

                    # Save as image
                    from PIL import Image
                    debug_img_np = (debug_image.cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(debug_img_np).save(os.path.join(debug_dir, f'step_{step:05d}.png'))

                    # Also save target for comparison at step 0
                    if step == 0:
                        target_img_np = (target_images[0].cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(target_img_np).save(os.path.join(debug_dir, f'target.png'))

                    logger.info(f"    Render range: [{debug_image.min():.3f}, {debug_image.max():.3f}], mean: {debug_image.mean():.3f}")

        # Detach gradients for final Gaussian
        with torch.no_grad():
            current_gaussian._features_dc = current_gaussian._features_dc.detach()
            current_gaussian._scaling = current_gaussian._scaling.detach()
            current_gaussian._rotation = current_gaussian._rotation.detach()
            current_gaussian._opacity = current_gaussian._opacity.detach()

        logger.info("\n" + "=" * 60)
        logger.info("Optimization completed!")
        logger.info(f"  Final loss: {total_loss.item():.6f}")
        logger.info(f"  Optimized Gaussian: {len(current_gaussian._xyz)} points")

        # Check final parameter ranges
        logger.info("\nFinal parameter ranges:")
        logger.info(f"  Colors (features_dc): [{current_gaussian._features_dc.min():.6f}, {current_gaussian._features_dc.max():.6f}]")
        logger.info(f"  Scales (log): [{current_gaussian._scaling.min():.6f}, {current_gaussian._scaling.max():.6f}]")
        final_scales_real = torch.exp(current_gaussian._scaling + current_gaussian.scale_bias)
        logger.info(f"  Scales (real): [{final_scales_real.min():.6f}, {final_scales_real.max():.6f}]")
        logger.info(f"  Opacities (logit): [{current_gaussian._opacity.min():.6f}, {current_gaussian._opacity.max():.6f}]")
        final_opacities_real = torch.sigmoid(current_gaussian._opacity + current_gaussian.opacity_bias)
        logger.info(f"  Opacities (real): [{final_opacities_real.min():.6f}, {final_opacities_real.max():.6f}]")
        logger.info(f"  Rotations norm: [{torch.norm(current_gaussian._rotation, dim=1).min():.6f}, {torch.norm(current_gaussian._rotation, dim=1).max():.6f}]")

        # Test render to check visibility
        logger.info("\nTest rendering final Gaussian...")
        with torch.no_grad():
            test_view = camera_views[0]
            test_extrinsics, test_intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                yaws=float(test_view['azimuth']),
                pitchs=float(test_view['elevation']),
                rs=float(test_view['radius']),
                fovs=60.0
            )
            test_result = renderer.render(current_gaussian, test_extrinsics, test_intrinsics)
            test_image = test_result['color'].permute(1, 2, 0)
            logger.info(f"  Test render output: [{test_image.min():.6f}, {test_image.max():.6f}]")
            logger.info(f"  Test render mean: {test_image.mean():.6f}")

        logger.info("=" * 60)

        return current_gaussian, connectivity_info


# ==================== Quick Alignment (Fast Alternative) ====================

def quick_align_gaussian(gaussian_params: Dict[str, torch.Tensor],
                        gaussian_trellis: Gaussian,
                        device: str = "cuda") -> Gaussian:
    """
    Quick alignment: only copy colors from TRELLIS, keep mesh geometry
    Much faster than full optimization (~seconds vs minutes)

    Args:
        gaussian_params: mesh-based Gaussian parameters
        gaussian_trellis: target TRELLIS Gaussian
        device: cuda or cpu

    Returns:
        aligned_gaussian: Gaussian with mesh geometry + TRELLIS colors
    """
    logger.info("Quick aligning Gaussian (color copy only)...")

    positions_real = gaussian_params['positions']

    # CRITICAL: Convert positions from real space to normalized space
    aabb = gaussian_trellis.aabb if hasattr(gaussian_trellis, 'aabb') else torch.tensor([-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]).to(positions_real.device)
    positions_normalized = (positions_real - aabb[:3]) / aabb[3:]

    # KNN matching (in normalized space)
    knn_result = knn_points(
        positions_normalized.unsqueeze(0),
        gaussian_trellis._xyz.unsqueeze(0),
        K=1
    )
    nearest_idx = knn_result.idx[0, :, 0]

    # Copy colors from matched TRELLIS Gaussians
    colors_matched = gaussian_trellis._features_dc[nearest_idx]

    # Build Gaussian
    aligned_gaussian = Gaussian(
        aabb=gaussian_trellis.aabb if hasattr(gaussian_trellis, 'aabb') else None,
        sh_degree=0,
        mininum_kernel_size=0.0,
        scaling_bias=0.01,
        opacity_bias=0.0,
        scaling_activation="exp",
    )

    aligned_gaussian._xyz = positions_normalized  # Use normalized coordinates!
    aligned_gaussian._features_dc = colors_matched
    aligned_gaussian._scaling = gaussian_params['scales']
    aligned_gaussian._rotation = gaussian_params['rotations']
    aligned_gaussian._opacity = gaussian_params['opacities']

    logger.info(f"Quick alignment done: {len(positions_normalized)} points")

    return aligned_gaussian


# ==================== Video Rendering ====================

def render_gaussian_video(gaussian: Gaussian,
                         output_path: str,
                         num_frames: int = 120,
                         resolution: int = 512,
                         radius: float = 1.2,
                         elevation: float = 0.0,
                         bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                         device: str = "cuda"):
    """
    Render 360-degree rotating video of a Gaussian

    Args:
        gaussian: Gaussian object to render
        output_path: Path to save the video
        num_frames: Number of frames in the video (default 120 = 4 seconds at 30fps)
        resolution: Resolution of the video frames
        radius: Camera distance from center
        elevation: Camera elevation angle
        bg_color: Background color RGB tuple (default white)
        device: cuda or cpu
    """
    from trellis.renderers import GaussianRenderer
    from trellis.utils import render_utils
    from easydict import EasyDict as edict
    import imageio

    logger.info(f"Rendering video: {num_frames} frames at {resolution}x{resolution}...")
    logger.info(f"  Output: {output_path}")

    # Setup renderer
    renderer = GaussianRenderer()
    renderer.rendering_options = edict({
        'resolution': resolution,
        'near': 0.8,
        'far': 1.6,
        'bg_color': torch.tensor(list(bg_color)).to(device),
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

    frames = []

    with torch.no_grad():
        for frame_idx in range(num_frames):
            # Calculate azimuth for 360-degree rotation (convert to radians)
            azimuth = (frame_idx / num_frames) * 2 * np.pi

            # Setup camera
            extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                yaws=azimuth,
                pitchs=elevation,
                rs=radius,
                fovs=60.0
            )

            # Render
            result = renderer.render(gaussian, extrinsics, intrinsics)
            rendered_image = result['color'].permute(1, 2, 0)  # [H, W, 3]
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

            # Convert to numpy uint8
            frame = (rendered_image.cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)

            if (frame_idx + 1) % 20 == 0:
                logger.info(f"  Rendered {frame_idx + 1}/{num_frames} frames")

    # Save video
    import os
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path contains a directory component
        os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=30, codec='libx264', quality=8)
    logger.info(f"Video rendering complete: {output_path}")


# ==================== Main Pipeline ====================

class MeshGuidedGaussianPipeline:
    """
    Complete pipeline: TRELLIS mesh + Gaussian → topology-aware optimized Gaussian
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.converter = MeshToGaussianConverter(device=device)
        self.optimizer = GaussianAlignmentOptimizer(device=device)

    def process(self,
               mesh_trellis,
               gaussian_trellis: Gaussian,
               optimize: bool = True,
               num_steps: int = 3000,
               num_views: int = 16,
               resolution: int = 512,
               scale_factor: float = 1.0,
               render_video: bool = True,
               video_output_path: str = "output_video.mp4",
               video_num_frames: int = 120,
               video_resolution: int = 512,
               **optimizer_kwargs) -> Tuple[Gaussian, Dict[int, Dict[int, float]]]:
        """
        Complete pipeline (Animate3D style)

        Args:
            mesh_trellis: TRELLIS mesh
            gaussian_trellis: TRELLIS Gaussian (target)
            optimize: if True, run full optimization; if False, quick align
            num_steps: optimization steps (if optimize=True)
            num_views: number of rendering views
            resolution: rendering resolution
            scale_factor: scale adjustment for Gaussian sizes
            render_video: if True, render a video after generating Gaussian (default True)
            video_output_path: path to save the rendered video (default "output_video.mp4")
            video_num_frames: number of frames in the video (default 120)
            video_resolution: resolution of the video frames (default 512)
            **optimizer_kwargs: additional arguments for optimizer

        Returns:
            final_gaussian: optimized Gaussian with topology
            connectivity_info: mesh topology
        """
        logger.info("\n" + "=" * 80)
        logger.info("MESH-GUIDED GAUSSIAN PIPELINE (Animate3D style)")
        logger.info("=" * 80)

        # Step 1: Convert mesh to Gaussian with topology
        gaussian_params, connectivity_info = self.converter.mesh_to_gaussian_params(
            mesh_trellis, scale_factor=scale_factor
        )

        # Step 2: Align to TRELLIS Gaussian
        if optimize:
            logger.info("\nRunning RENDERING-BASED optimization...")
            final_gaussian, connectivity_info = self.optimizer.align_to_target_with_rendering(
                gaussian_params,
                connectivity_info,
                gaussian_trellis,
                num_steps=num_steps,
                num_views=num_views,
                resolution=resolution,
                **optimizer_kwargs
            )
        else:
            logger.info("\nRunning QUICK alignment (color copy only)...")
            final_gaussian = quick_align_gaussian(
                gaussian_params,
                gaussian_trellis,
                device=self.device
            )

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED")
        logger.info(f"  Final Gaussian: {len(final_gaussian._xyz)} points")
        logger.info(f"  Topology edges: {sum(len(v) for v in connectivity_info.values()) // 2}")
        logger.info("=" * 80 + "\n")

        # Step 3: Render video if requested
        if render_video:
            logger.info("\n" + "=" * 80)
            logger.info("RENDERING VIDEO")
            logger.info("=" * 80)
            render_gaussian_video(
                gaussian=final_gaussian,
                output_path=video_output_path,
                num_frames=video_num_frames,
                resolution=video_resolution,
                device=self.device
            )
            logger.info("=" * 80 + "\n")

        return final_gaussian, connectivity_info


# ==================== Convenience Functions ====================

def convert_mesh_to_topology_gaussian(mesh_trellis,
                                     gaussian_trellis: Gaussian,
                                     device: str = "cuda",
                                     optimize: bool = True,
                                     num_steps: int = 3000,
                                     num_views: int = 16,
                                     resolution: int = 512,
                                     scale_factor: float = 1.0,
                                     render_video: bool = True,
                                     video_output_path: str = "output_video.mp4",
                                     video_num_frames: int = 120,
                                     video_resolution: int = 512) -> Tuple[Gaussian, Dict]:
    """
    Convenience function: one-line mesh → topology-aware Gaussian (Animate3D style)

    Args:
        mesh_trellis: TRELLIS mesh
        gaussian_trellis: TRELLIS Gaussian
        device: cuda or cpu
        optimize: full optimization (True) or quick align (False)
        num_steps: optimization steps (default 3000)
        num_views: number of rendering views (default 16)
        resolution: rendering resolution (default 512)
        scale_factor: Gaussian size adjustment
        render_video: if True, render a video after generating Gaussian (default True)
        video_output_path: path to save the rendered video (default "output_video.mp4")
        video_num_frames: number of frames in the video (default 120)
        video_resolution: resolution of the video frames (default 512)

    Returns:
        gaussian: optimized Gaussian with topology
        connectivity: topology information

    Example:
        >>> gaussian, topology = convert_mesh_to_topology_gaussian(
        ...     mesh_trellis=out['mesh'][0],
        ...     gaussian_trellis=out['gaussian'][0],
        ...     optimize=True,
        ...     num_steps=3000,
        ...     num_views=16,
        ...     render_video=True,
        ...     video_output_path="my_gaussian.mp4"
        ... )
    """
    pipeline = MeshGuidedGaussianPipeline(device=device)
    return pipeline.process(
        mesh_trellis=mesh_trellis,
        gaussian_trellis=gaussian_trellis,
        optimize=optimize,
        num_steps=num_steps,
        num_views=num_views,
        resolution=resolution,
        scale_factor=scale_factor,
        render_video=render_video,
        video_output_path=video_output_path,
        video_num_frames=video_num_frames,
        video_resolution=video_resolution
    )
