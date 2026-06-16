"""Batch-generate textured GLBs for every image in `input/`.

Usage:
    python src/batch.py [options...]

Already-generated assets (matching output file already exists) are skipped,
so the batch can be safely re-run/resumed. A failure on one image is logged
and does not stop the batch.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

from config import add_config_cli_arguments, resolve_config
from pipeline import Hunyuan3DPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-generate GLBs from a folder of images with Hunyuan3D-2.1")
    add_config_cli_arguments(parser)
    return parser


def find_input_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())


def peak_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except ImportError:
        pass
    return 0.0


def reset_vram_stats() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        return 1

    images = find_input_images(input_dir)
    if not images:
        logger.warning("No images found in %s (extensions: %s)", input_dir, sorted(IMAGE_EXTENSIONS))
        return 0

    logger.info("Found %d image(s) in %s", len(images), input_dir)

    pipeline = Hunyuan3DPipeline(
        model_path=config.model_path,
        enable_texture=config.enable_texture,
        low_vram=config.low_vram,
        paint_max_num_view=config.paint_max_num_view,
        paint_resolution=config.paint_resolution,
    )

    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for image_path in tqdm(images, desc="Batch generation", unit="asset"):
        output_path = output_dir / f"{image_path.stem}.{config.output_format}"

        if output_path.exists():
            logger.info("Skipping '%s' (already generated: '%s')", image_path.name, output_path.name)
            skipped.append(image_path.name)
            continue

        reset_vram_stats()
        start = time.monotonic()
        try:
            mesh = pipeline.generate(
                image_path=str(image_path),
                face_limit=config.face_limit,
                octree_resolution=config.octree_resolution,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
            )
            pipeline.export(mesh, str(output_path))
            elapsed = time.monotonic() - start
            vram = peak_vram_gb()
            logger.info(
                "OK '%s' -> '%s' (%.1fs, peak VRAM %.2f GB)",
                image_path.name,
                output_path.name,
                elapsed,
                vram,
            )
            succeeded.append(image_path.name)
        except Exception as exc:  # noqa: BLE001 - keep batch alive on any error
            elapsed = time.monotonic() - start
            logger.error("FAILED '%s' after %.1fs: %s", image_path.name, elapsed, exc, exc_info=True)
            failed.append((image_path.name, str(exc)))

    total = len(images)
    logger.info(
        "Batch finished: %d succeeded, %d skipped, %d failed (total %d)",
        len(succeeded),
        len(skipped),
        len(failed),
        total,
    )
    if failed:
        logger.info("Failed assets:")
        for name, error in failed:
            logger.info("  - %s: %s", name, error)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
