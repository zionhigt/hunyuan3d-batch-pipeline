"""Generate a single textured GLB from one image.

Usage:
    python src/single.py --image input/test.png [--output output/test.glb] [options...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import add_config_cli_arguments, resolve_config
from pipeline import Hunyuan3DPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a single GLB from an image with Hunyuan3D-2.1")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, default=None, help="Output mesh path (default: output/<image_stem>.<format>)")
    add_config_cli_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(config.output_dir) / f"{image_path.stem}.{config.output_format}"

    pipeline = Hunyuan3DPipeline(
        model_path=config.model_path,
        enable_texture=config.enable_texture,
        low_vram=config.low_vram,
        paint_max_num_view=config.paint_max_num_view,
        paint_resolution=config.paint_resolution,
    )

    mesh = pipeline.generate(
        image_path=str(image_path),
        face_limit=config.face_limit,
        octree_resolution=config.octree_resolution,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
    )
    pipeline.export(mesh, str(output_path))

    logger.info("Done: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
