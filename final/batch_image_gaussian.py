#!/usr/bin/env python3
"""
Batch Generate Topology Gaussians from Human/Animal Data Images
Reads images from human_data/{name}/image.png or animal_data/{name}/image.png
Saves results to mesh_gaussian/ with mesh, gaussian (PLY), and connectivity
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
from PIL import Image
from tqdm import tqdm
import re

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Import TRELLIS pipeline
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

# Import mesh guidance module
from mesh_guidance import convert_mesh_to_topology_gaussian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_topology_gaussian_from_image(
    image_name: str,
    image_path: Path,
    output_base_dir: Path,
    pipeline,
    device: str = "cuda"
):
    """
    Generate topology gaussian from a human image

    Args:
        image_name: Name of the image file (without extension), used as folder name
        image_path: Path to input image (PNG)
        output_base_dir: Base directory to save results (mesh_gaussian/human_gaussian)
        pipeline: Pre-loaded TRELLIS image-to-3D pipeline
        device: CUDA device
    """
    # Create output directory using image name
    output_dir = output_base_dir / image_name
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {image_name}")
    logger.info(f"Image: {image_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*80}")

    # Define output file paths
    ply_path = output_dir / f"{image_name}_topology_gaussian.ply"
    mesh_path = output_dir / f"{image_name}_mesh.glb"
    connectivity_path = output_dir / f"{image_name}_connectivity.pkl"
    video_path = output_dir / f"{image_name}_topology_gaussian.mp4"

    # Check what files exist
    if ply_path.exists() and mesh_path.exists() and connectivity_path.exists():
        logger.info(f"✓ All files already exist (PLY, mesh, connectivity), SKIPPING...")
        return "skipped"

    # If ply exists but mesh/connectivity missing, supplement them
    if ply_path.exists():
        missing = []
        if not mesh_path.exists():
            missing.append("mesh")
        if not connectivity_path.exists():
            missing.append("connectivity")

        if missing:
            logger.info(f"⚠ PLY exists but missing: {', '.join(missing)}, SUPPLEMENTING...")

            try:
                # Load image
                if not image_path.exists():
                    logger.error(f"  Image not found: {image_path}")
                    return False

                image = Image.open(image_path).convert('RGB')

                # Generate 3D to get mesh
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

                # Save mesh if missing
                if not mesh_path.exists():
                    import trimesh
                    mesh_to_save = trimesh.Trimesh(
                        vertices=mesh_trellis.vertices.cpu().numpy(),
                        faces=mesh_trellis.faces.cpu().numpy()
                    )
                    mesh_to_save.export(str(mesh_path))
                    logger.info(f"  ✓ Saved mesh to {mesh_path}")

                # Extract and save connectivity if missing
                if not connectivity_path.exists():
                    from mesh_guidance import MeshToGaussianConverter
                    logger.info(f"  Extracting connectivity from mesh topology...")
                    converter = MeshToGaussianConverter(device=device)
                    connectivity = converter.extract_mesh_connectivity(
                        vertices=mesh_trellis.vertices.to(device),
                        faces=mesh_trellis.faces.to(device)
                    )
                    num_edges = sum(len(v) for v in connectivity.values()) // 2
                    logger.info(f"  Connectivity extracted: {len(connectivity)} vertices, {num_edges} edges")

                    import pickle
                    with open(connectivity_path, 'wb') as f:
                        pickle.dump(connectivity, f)
                    logger.info(f"  ✓ Saved connectivity to {connectivity_path}")

                logger.info(f"  ✓ Supplemented missing files for {image_name}")
                return "supplemented"

            except Exception as e:
                logger.error(f"  ✗ Failed to supplement {image_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False

    # Full generation needed
    try:
        # Step 1: Load image
        logger.info(f"  Loading image...")
        if not image_path.exists():
            logger.error(f"  Image not found: {image_path}")
            return False

        image = Image.open(image_path).convert('RGB')
        logger.info(f"  Image loaded: {image.size}")

        # Step 2: Generate 3D mesh and gaussian using TRELLIS image-to-3D
        logger.info(f"  Generating 3D from image using TRELLIS image-to-3D...")
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

        # Extract mesh and gaussian
        mesh_trellis = outputs['mesh'][0]
        gaussian_trellis = outputs['gaussian'][0]

        # Get mesh info
        num_vertices = len(mesh_trellis.vertices)
        logger.info(f"  Generated mesh with {num_vertices} vertices")
        logger.info(f"  Generated gaussian with {len(gaussian_trellis._xyz)} points")

        # Step 3: Use mesh guidance to create topology-aware gaussian
        logger.info(f"  Converting mesh to topology-aware Gaussian (64 views, 10k steps)...")
        gaussian_final, connectivity = convert_mesh_to_topology_gaussian(
            mesh_trellis=mesh_trellis,
            gaussian_trellis=gaussian_trellis,
            device=device,
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

        # Step 4: Save mesh
        import trimesh
        mesh_to_save = trimesh.Trimesh(
            vertices=mesh_trellis.vertices.cpu().numpy(),
            faces=mesh_trellis.faces.cpu().numpy()
        )
        mesh_to_save.export(str(mesh_path))
        logger.info(f"  ✓ Saved mesh to {mesh_path}")

        # Step 5: Save connectivity
        import pickle
        with open(connectivity_path, 'wb') as f:
            pickle.dump(connectivity, f)
        logger.info(f"  ✓ Saved connectivity to {connectivity_path}")

        # Step 6: Save Gaussian (PLY format)
        gaussian_final.save_ply(str(ply_path))
        logger.info(f"  ✓ Saved Gaussian to {ply_path}")

        # Step 7: Save metadata
        info_path = output_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump({
                'image_name': image_name,
                'num_points': len(gaussian_final._xyz),
                'num_topology_edges': sum(len(v) for v in connectivity.values()) // 2,
                'num_vertices': num_vertices,
                'num_views': 64,
                'num_steps': 10000,
                'scale_factor': 1.0/1.1,
                'use_mesh_guidance': True,
                'image_path': str(image_path),
            }, f, indent=2)
        logger.info(f"  ✓ Saved metadata to {info_path}")

        logger.info(f"  ✓ Successfully processed {image_name}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to process {image_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Batch generate topology gaussians from human/animal data')
    parser.add_argument('--input_dir', type=str,
                       default=None,
                       help='Path to input directory with {name}/image.png structure (default: human_data or animal_data)')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory for generated gaussians (default: mesh_gaussian)')
    parser.add_argument('--data_type', type=str, default='human',
                       choices=['human', 'animal'],
                       help='Data type: human or animal')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start from this index in folder list')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End at this index (exclusive), None for all')

    args = parser.parse_args()

    # 设置默认路径（相对于脚本目录）
    if args.input_dir is None:
        args.input_dir = str(SCRIPT_DIR / f"{args.data_type}_data")
    if args.output_dir is None:
        args.output_dir = str(SCRIPT_DIR / "mesh_gaussian")

    # Scan input directory for PNG files
    input_dir = Path(args.input_dir)
    logger.info(f"Scanning input directory: {input_dir}...")

    # 查找所有PNG文件
    image_files = sorted(list(input_dir.glob("*.png")))

    logger.info(f"Found {len(image_files)} PNG files in {input_dir}")

    if len(image_files) == 0:
        logger.error(f"No PNG files found in {input_dir}")
        return

    # Apply index range
    if args.end_idx is not None:
        images_to_process = image_files[args.start_idx:args.end_idx]
    else:
        images_to_process = image_files[args.start_idx:]

    logger.info(f"Processing {len(images_to_process)} images (index {args.start_idx} to {args.end_idx or 'end'})")

    # Create output directory
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_base_dir}")

    # Pre-scan to count what needs to be done
    logger.info("=" * 80)
    logger.info("PRE-SCAN: Checking existing files...")
    logger.info("=" * 80)

    need_full_generation = []
    need_supplement = []
    already_complete = []

    for image_path in images_to_process:
        image_name = image_path.stem  # 文件名（不含扩展名）
        output_dir = output_base_dir / image_name

        ply_path = output_dir / f"{image_name}_topology_gaussian.ply"
        mesh_path = output_dir / f"{image_name}_mesh.glb"
        connectivity_path = output_dir / f"{image_name}_connectivity.pkl"

        if ply_path.exists() and mesh_path.exists() and connectivity_path.exists():
            already_complete.append(image_name)
        elif ply_path.exists():
            need_supplement.append(image_name)
        else:
            need_full_generation.append(image_name)

    logger.info(f"📊 Status Summary:")
    logger.info(f"  ✓ Already complete: {len(already_complete)}")
    logger.info(f"  ⚠ Need supplement (mesh/connectivity): {len(need_supplement)}")
    logger.info(f"  🔨 Need full generation: {len(need_full_generation)}")
    logger.info(f"  📝 Total to process: {len(need_supplement) + len(need_full_generation)}")

    if need_supplement:
        logger.info(f"\n📋 Will supplement these instances:")
        for name in need_supplement[:10]:  # Show first 10
            logger.info(f"    - {name}")
        if len(need_supplement) > 10:
            logger.info(f"    ... and {len(need_supplement) - 10} more")

    if need_full_generation:
        logger.info(f"\n🔨 Will fully generate these instances:")
        for name in need_full_generation[:10]:  # Show first 10
            logger.info(f"    - {name}")
        if len(need_full_generation) > 10:
            logger.info(f"    ... and {len(need_full_generation) - 10} more")

    logger.info("=" * 80)

    # Load TRELLIS image-to-3D pipeline once
    logger.info("Loading TRELLIS image-to-3D pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.to(args.device)
    logger.info("Pipeline loaded successfully!")

    # Process each image
    success_count = 0
    fail_count = 0
    supplement_count = 0
    skip_count = 0

    for image_path in tqdm(images_to_process, desc="Generating topology gaussians"):
        image_name = image_path.stem  # 文件名（不含扩展名）

        result = generate_topology_gaussian_from_image(
            image_name=image_name,
            image_path=image_path,
            output_base_dir=output_base_dir,
            pipeline=pipeline,
            device=args.device
        )

        if result == "skipped":
            skip_count += 1
            success_count += 1
        elif result == "supplemented":
            supplement_count += 1
            success_count += 1
        elif result is True:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"\n{'='*80}")
    logger.info(f"Batch processing complete!")
    logger.info(f"  Total processed: {len(images_to_process)}")
    logger.info(f"  ✓ Successful: {success_count}")
    logger.info(f"    - Fully generated: {success_count - supplement_count - skip_count}")
    logger.info(f"    - Supplemented (mesh/connectivity): {supplement_count}")
    logger.info(f"    - Skipped (already complete): {skip_count}")
    logger.info(f"  ✗ Failed: {fail_count}")
    logger.info(f"Results saved to: {output_base_dir}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()