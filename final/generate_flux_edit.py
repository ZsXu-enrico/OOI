#!/usr/bin/env python3
"""
Generate FLUX-Edited Reference Images for Humanoid-Object Interactions
- Loads humanoid Gaussians from mesh_gaussian/human_gaussian/
- Loads object images from object_data/ (PNG files)
- Converts object PNGs to 3D Gaussians using TRELLIS image-to-3D
- Renders humanoid and object Gaussians at 225 degrees with radius 1.2
- Uses FLUX Kontext to generate edited reference images
- Each humanoid gets 3 random actions (objects) from prompt.txt
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
import random

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Import TRELLIS components
from trellis.representations import Gaussian
from trellis.renderers import GaussianRenderer
from trellis.utils import render_utils
from easydict import EasyDict as edict

# Import FLUX Kontext
from diffusers import FluxKontextPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FluxEditReferenceGenerator:
    def __init__(self,
                 humanoid_gaussian_dir: str = None,
                 prompt_file: str = None,
                 object_prompt_file: str = None,
                 output_dir: str = None,
                 flux_model_path: str = None,
                 object_data_dir: str = None,
                 num_actions_per_humanoid: int = 3,
                 device: str = "cuda"):

        # 设置默认路径（相对于脚本目录）
        if humanoid_gaussian_dir is None:
            humanoid_gaussian_dir = str(SCRIPT_DIR / "mesh_gaussian")
        if object_data_dir is None:
            object_data_dir = str(SCRIPT_DIR / "object_data")
        if output_dir is None:
            output_dir = str(SCRIPT_DIR / "flux_edit")
        if prompt_file is None:
            prompt_file = str(SCRIPT_DIR / "action_prompts.txt")
        if object_prompt_file is None:
            object_prompt_file = str(SCRIPT_DIR / "object_prompts.txt")
        if flux_model_path is None:
            flux_model_path = str(SCRIPT_DIR.parent / "pretrained_models" / "FLUX.1-Kontext-dev")

        self.humanoid_gaussian_dir = Path(humanoid_gaussian_dir)
        self.prompt_file = Path(prompt_file)
        self.object_prompt_file = Path(object_prompt_file)
        self.output_dir = Path(output_dir)
        self.flux_model_path = flux_model_path
        self.object_data_dir = Path(object_data_dir)
        self.num_actions_per_humanoid = num_actions_per_humanoid
        self.device = device

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _get_humanoid_caption(self, folder_name: str) -> str:
        """Convert folder name to humanoid caption

        Examples:
            '18th-century soldier' -> '18th-century soldier'
            'anthropomorphic_piglet_character' -> 'anthropomorphic piglet character'
            '207_character' -> 'character'
        """
        # Replace underscores with spaces
        caption = folder_name.replace('_', ' ')

        # For numbered folders like "207_character", just use the suffix
        if caption.split()[0].isdigit():
            parts = caption.split()
            caption = ' '.join(parts[1:]) if len(parts) > 1 else caption

        return caption

    def load_gaussian(self, ply_path: Path) -> Gaussian:
        """Load Gaussian from PLY file"""
        # Create Gaussian instance first, then load ply file
        gaussian = Gaussian(
            aabb=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            sh_degree=0,
            device=self.device
        )
        # Load ply file into the instance
        gaussian.load_ply(str(ply_path))
        return gaussian

    def load_object_from_data(self, object_text: str) -> Image.Image:
        """Load object from object_data/{name}.png, generate 3D gaussian, and render it

        Args:
            object_text: Object text description (e.g., "a frisbee")

        Returns:
            Rendered 512x512 image of the object, or None if not found
        """
        from trellis.pipelines import TrellisImageTo3DPipeline

        # Extract keywords from object text (e.g., "a frisbee" -> "frisbee")
        words = object_text.lower().split()
        words = [w for w in words if w not in ['a', 'an', 'the']]
        keywords = ' '.join(words) if words else object_text.lower()

        # Try to find matching PNG file in object_data/
        png_path = None
        if self.object_data_dir.exists():
            for png_file in self.object_data_dir.glob("*.png"):
                file_name = png_file.stem.lower()
                # Check if keyword matches the file name
                if file_name in keywords or any(kw in file_name for kw in keywords.split()):
                    png_path = png_file
                    break

        if not png_path:
            return None

        try:
            # Load the PNG from object_data
            logger.info(f"    Loading object from object_data: {png_path.name}...")
            object_image = Image.open(png_path).convert('RGB')

            # Load image-to-3D pipeline if not already loaded
            if not hasattr(self, 'image_to_3d_pipeline'):
                logger.info(f"    Loading TRELLIS image-to-3D pipeline...")
                self.image_to_3d_pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
                self.image_to_3d_pipeline.to(self.device)

            # Generate 3D gaussian from image
            logger.info(f"    Generating 3D gaussian from object_data image...")
            outputs = self.image_to_3d_pipeline.run(
                image=object_image,
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

            gaussian = outputs['gaussian'][0]
            logger.info(f"    Generated gaussian with {len(gaussian._xyz)} points")

            # Render the gaussian at 225 degrees
            logger.info(f"    Rendering gaussian at 225 degrees...")
            rendered_image = self.render_gaussian_view(gaussian, azimuth=np.radians(225))

            return rendered_image

        except Exception as e:
            logger.error(f"    Error processing object from object_data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def render_gaussian_view(self,
                             gaussian: Gaussian,
                             azimuth: float = np.radians(135),  # Side view
                             elevation: float = 0.0,
                             radius: float = 1.2,
                             resolution: int = 512) -> Image.Image:
        """Render a single view of a Gaussian using TRELLIS renderer"""

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
            'bg_color': (1.0, 1.0, 1.0),  # White background
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

        # Render
        with torch.no_grad():
            result = renderer.render(gaussian, extrinsics, intrinsics)
            rendered_image = result['color'].permute(1, 2, 0)  # [H, W, 3]
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

        # Convert to PIL
        rendered_np = (rendered_image.cpu().numpy() * 255).astype(np.uint8)
        rendered_pil = Image.fromarray(rendered_np)

        return rendered_pil

    def extract_object_from_action(self, action_prompt: str) -> str:
        """Extract object name from action prompt
        Examples:
            'studying a map' -> 'map'
            'pumping gas' -> 'gas'
            'playing basketball' -> 'basketball'
        """
        words = action_prompt.split()

        # Try to find article (a, an, the) and take word after it
        articles = ['a', 'an', 'the']
        for i, word in enumerate(words):
            if word.lower() in articles and i + 1 < len(words):
                return words[i + 1]

        # Otherwise, take last word
        return words[-1] if words else action_prompt

    def generate_flux_edit(self,
                          humanoid_img: Image.Image,
                          object_img: Image.Image,
                          humanoid_caption: str,
                          action_prompt: str,
                          flux_pipe) -> Image.Image:
        """Generate edited reference image using FLUX Kontext"""

        # Concatenate images horizontally (FLUX multi-image input format)
        total_width = humanoid_img.width + object_img.width
        max_height = max(humanoid_img.height, object_img.height)

        concatenated_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        # Paste images with vertical centering
        y_offset_humanoid = (max_height - humanoid_img.height) // 2
        y_offset_object = (max_height - object_img.height) // 2
        concatenated_image.paste(humanoid_img, (0, y_offset_humanoid))
        concatenated_image.paste(object_img, (humanoid_img.width, y_offset_object))

        # Extract object from action prompt (e.g., "studying a map" -> "map")
        extracted_object = self.extract_object_from_action(action_prompt)

        # Create FLUX prompt using the template from sds_mv_mesh.py:
        # "Create a unified, cohesive image where the [humanoid] is [action].
        #  Maintain the identity and characteristics of each subject while adjusting their
        #  proportions, scale, and positioning. Do not change the color of the background."
        # Note: action_prompt already contains the full action (e.g., "studying a map")
        prompt = (
            f"Create a unified, cohesive image where the {humanoid_caption} is "
            f"{action_prompt}. "
            f"Maintain the identity and characteristics of each subject while adjusting their "
            f"proportions, scale, and positioning. Do not change the color of the background."
        )

        logger.info(f"  Action: {action_prompt}")
        logger.info(f"  Extracted object: {extracted_object}")
        logger.info(f"  FLUX prompt: '{prompt[:200]}...'")

        # Generate edited image using FLUX Kontext
        with torch.no_grad():
            result = flux_pipe(
                prompt=prompt,
                image=concatenated_image,
                height=max_height,
                width=total_width,
                guidance_scale=2.5,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]

        # Crop left half as the final composed result
        result_width, result_height = result.size
        mid_x = result_width // 2
        left_half = result.crop((0, 0, mid_x, result_height))

        return left_half, result

    def load_prompts(self):
        """Load action prompts from prompt.txt"""
        with open(self.prompt_file, 'r') as f:
            prompts = []
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
        return prompts

    def load_object_prompts(self):
        """Load object texts from object_prompt.txt"""
        with open(self.object_prompt_file, 'r') as f:
            object_prompts = []
            for line in f:
                line = line.strip()
                if line:
                    object_prompts.append(line)
        return object_prompts

    def get_humanoid_folders(self):
        """Get all humanoid folders from mesh_gaussian/human_gaussian/"""
        humanoid_folders = sorted([f for f in self.humanoid_gaussian_dir.iterdir() if f.is_dir()])
        return humanoid_folders

    def generate_all_references(self, humanoid_id=None, object_ids=None):
        """Generate all FLUX-edited reference images

        Args:
            humanoid_id: Optional specific humanoid folder name to process
            object_ids: Optional list of object IDs to use (e.g., [1, 5, 10])
        """
        logger.info("="*80)
        logger.info("FLUX-EDITED REFERENCE IMAGE GENERATION")
        logger.info("="*80)

        if humanoid_id:
            logger.info(f"Processing specific humanoid: {humanoid_id}")
        if object_ids:
            logger.info(f"Using specific object IDs: {object_ids}")

        # Load prompts
        action_prompts = self.load_prompts()
        object_prompts = self.load_object_prompts()
        logger.info(f"Loaded {len(action_prompts)} action prompts")
        logger.info(f"Loaded {len(object_prompts)} object texts")

        # Ensure they have the same length
        if len(action_prompts) != len(object_prompts):
            logger.error(f"Mismatch: {len(action_prompts)} actions vs {len(object_prompts)} objects")
            return

        # Get all humanoids
        humanoid_folders = self.get_humanoid_folders()

        # Filter to specific humanoid if requested
        if humanoid_id:
            humanoid_folders = [f for f in humanoid_folders if f.name == humanoid_id]
            if not humanoid_folders:
                logger.error(f"Humanoid '{humanoid_id}' not found in {self.humanoid_gaussian_dir}")
                return

        logger.info(f"Found {len(humanoid_folders)} humanoids")
        logger.info(f"Total objects: {len(object_prompts)}")

        # Load FLUX Kontext pipeline once
        logger.info("\nLoading FLUX Kontext pipeline...")
        flux_pipe = FluxKontextPipeline.from_pretrained(
            self.flux_model_path,
            torch_dtype=torch.bfloat16
        )
        flux_pipe.to(self.device)
        flux_pipe.enable_vae_slicing()
        flux_pipe.enable_vae_tiling()
        logger.info("FLUX pipeline loaded!")

        # Process each humanoid
        total_generated = 0
        total_skipped = 0

        for humanoid_folder in tqdm(humanoid_folders, desc="Processing humanoids"):
            humanoid_name = humanoid_folder.name
            # Look for topology_gaussian.ply file
            humanoid_ply = humanoid_folder / f"{humanoid_name}_topology_gaussian.ply"

            if not humanoid_ply.exists():
                logger.warning(f"PLY not found for {humanoid_name}, skipping")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing humanoid: {humanoid_name}")
            logger.info(f"{'='*60}")

            # Get humanoid caption from folder name
            humanoid_caption = self._get_humanoid_caption(humanoid_name)
            logger.info(f"  Caption: {humanoid_caption}")

            # Check if this humanoid already has enough completed objects
            existing_outputs = list(self.output_dir.glob(f"{humanoid_name}_*"))
            completed_count = sum(1 for folder in existing_outputs
                                 if (folder / "reference_edited.png").exists())

            if completed_count >= self.num_actions_per_humanoid:
                logger.info(f"  Humanoid already completed ({completed_count}/{self.num_actions_per_humanoid}), skipping")
                total_skipped += self.num_actions_per_humanoid
                continue

            logger.info(f"  Already completed: {completed_count}/{self.num_actions_per_humanoid}")

            # Load humanoid Gaussian
            logger.info(f"  Loading humanoid Gaussian from {humanoid_ply}")
            humanoid_gaussian = self.load_gaussian(humanoid_ply)

            # Render humanoid view (225 degrees = 180+45, back-side view)
            logger.info(f"  Rendering humanoid view at 225 degrees...")
            humanoid_img = self.render_gaussian_view(humanoid_gaussian, azimuth=np.radians(225))

            # Select object IDs to process
            if object_ids:
                # Use specified object IDs
                selected_object_ids = object_ids
                logger.info(f"  Using specified objects: {selected_object_ids}")
            else:
                # Select N random object IDs from all available objects (1-172)
                total_objects = len(object_prompts)
                all_object_ids = list(range(1, total_objects + 1))
                # Only select the number we still need
                num_to_select = self.num_actions_per_humanoid - completed_count
                selected_object_ids = random.sample(all_object_ids, min(num_to_select, len(all_object_ids)))
                logger.info(f"  Selected {len(selected_object_ids)} new objects: {selected_object_ids}")

            for object_id in selected_object_ids:
                action_prompt = action_prompts[object_id - 1]  # 1-indexed to 0-indexed
                object_text = object_prompts[object_id - 1]  # 1-indexed to 0-indexed

                # Create output folder name using object_id
                output_folder_name = f"{humanoid_name}_{object_id:03d}"
                output_folder = self.output_dir / output_folder_name
                output_folder.mkdir(exist_ok=True, parents=True)

                # Check if already processed
                reference_path = output_folder / "reference_edited.png"
                if reference_path.exists():
                    logger.info(f"  Already processed: {output_folder_name}, skipping")
                    total_skipped += 1
                    continue

                logger.info(f"  Processing object ID {object_id}: {object_text}")
                logger.info(f"  Action: {action_prompt}")
                logger.info(f"  Output: {output_folder_name}")

                # Load object from object_data PNG
                logger.info(f"    Loading object from object_data...")
                object_img = self.load_object_from_data(object_text)

                if object_img is None:
                    logger.error(f"    Object not found in object_data, skipping...")
                    continue

                object_source = "object_data"

                # Save individual renders
                humanoid_img.save(output_folder / "humanoid_render.png")
                object_img.save(output_folder / "object_render.png")

                # Generate FLUX edit
                logger.info(f"    Generating FLUX edit...")
                edited_img, full_output = self.generate_flux_edit(
                    humanoid_img=humanoid_img,
                    object_img=object_img,
                    humanoid_caption=humanoid_caption,
                    action_prompt=action_prompt,
                    flux_pipe=flux_pipe
                )

                # Save results
                edited_img.save(reference_path)
                full_output.save(output_folder / "flux_output_full.png")

                # Save metadata
                metadata = {
                    'humanoid_name': humanoid_name,
                    'object_id': object_id,
                    'object_text': object_text,
                    'action_prompt': action_prompt,
                    'object_source': object_source,
                    'humanoid_ply': str(humanoid_ply)
                }

                with open(output_folder / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"    ✓ Saved to {output_folder} (source: {object_source})")
                total_generated += 1

        # Cleanup
        del flux_pipe
        if hasattr(self, 'image_to_3d_pipeline'):
            del self.image_to_3d_pipeline
        torch.cuda.empty_cache()

        logger.info("\n" + "="*80)
        logger.info("GENERATION COMPLETE")
        logger.info(f"  Total generated: {total_generated}")
        logger.info(f"  Total skipped: {total_skipped}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info("="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate FLUX-edited reference images')
    parser.add_argument('--humanoid_gaussian_dir', type=str,
                       default='/data4/zishuo/mesh_gaussian/animal_data',
                       help='Path to humanoid gaussian directory')
    parser.add_argument('--prompt_file', type=str,
                       default='/data4/zishuo/TRELLIS/animal_object/object/1.txt',
                       help='Path to prompt.txt (action prompts)')
    parser.add_argument('--object_prompt_file', type=str,
                       default='/data4/zishuo/TRELLIS/animal_object.txt',
                       help='Path to object_prompt.txt (object texts)')
    parser.add_argument('--output_dir', type=str,
                       default='/data4/zishuo/TRELLIS/flux_edit_animal',
                       help='Output directory for flux-edited references')
    parser.add_argument('--flux_model_path', type=str,
                       default='/data4/zishuo/FLUX.1-Kontext-dev',
                       help='Path to FLUX Kontext model')
    parser.add_argument('--object_data_dir', type=str,
                       default='/data4/zishuo/TRELLIS/animal_object/object/name',
                       help='Path to object_data directory (containing PNG images for objects)')
    parser.add_argument('--num_actions', type=int, default=3,
                       help='Number of objects per humanoid (default: 3)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--humanoid_id', type=str, default=None,
                       help='Specific humanoid folder name to process (e.g., "18th-century soldier"). If not specified, process all humanoids.')
    parser.add_argument('--object_ids', type=str, default=None,
                       help='Comma-separated object IDs to use (e.g., "1,5,10"). If not specified, randomly select objects.')

    args = parser.parse_args()

    # Parse object_ids if provided
    object_ids = None
    if args.object_ids:
        object_ids = [int(x.strip()) for x in args.object_ids.split(',')]

    generator = FluxEditReferenceGenerator(
        humanoid_gaussian_dir=args.humanoid_gaussian_dir,
        prompt_file=args.prompt_file,
        object_prompt_file=args.object_prompt_file,
        output_dir=args.output_dir,
        flux_model_path=args.flux_model_path,
        object_data_dir=args.object_data_dir,
        num_actions_per_humanoid=args.num_actions,
        device=args.device
    )

    generator.generate_all_references(
        humanoid_id=args.humanoid_id,
        object_ids=object_ids
    )


if __name__ == '__main__':
    main()
