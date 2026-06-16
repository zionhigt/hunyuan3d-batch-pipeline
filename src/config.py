"""Configuration loading and validation for the Hunyuan3D batch pipeline.

Loads defaults from a YAML file and allows overriding individual fields
(typically from CLI arguments).
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class PipelineConfig:
    """Runtime configuration for shape + texture generation."""

    model_path: str = "tencent/Hunyuan3D-2.1"
    enable_texture: bool = True
    octree_resolution: int = 256
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    face_limit: Optional[int] = 40000
    output_format: str = "glb"
    low_vram: bool = False
    paint_max_num_view: int = 6
    paint_resolution: int = 512
    input_dir: str = "input"
    output_dir: str = "output"

    def validate(self) -> None:
        if self.octree_resolution <= 0:
            raise ValueError("octree_resolution must be a positive integer")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        if self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be positive")
        if self.face_limit is not None and self.face_limit <= 0:
            raise ValueError("face_limit must be a positive integer or null")
        if self.output_format.lower() not in {"glb", "obj", "ply", "stl"}:
            raise ValueError(f"unsupported output_format: {self.output_format}")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    valid_fields = {f.name for f in fields(PipelineConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config key(s) in {path}: {sorted(unknown)}")

    config = PipelineConfig(**data)
    config.validate()
    return config


def apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Return a copy of `config` with any non-None CLI args overriding fields."""
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in {f.name for f in fields(PipelineConfig)} and value is not None
    }
    updated = dataclasses.replace(config, **overrides)
    updated.validate()
    return updated


def add_config_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the common config-override CLI flags on `parser`."""
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument("--model-path", dest="model_path", type=str, default=None, help="HF model id or local path")
    parser.add_argument(
        "--enable-texture",
        dest="enable_texture",
        action="store_true",
        default=None,
        help="Enable PBR texture generation",
    )
    parser.add_argument(
        "--no-texture",
        dest="enable_texture",
        action="store_false",
        default=None,
        help="Disable texture generation (geometry only)",
    )
    parser.add_argument("--octree-resolution", dest="octree_resolution", type=int, default=None)
    parser.add_argument("--num-inference-steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance-scale", dest="guidance_scale", type=float, default=None)
    parser.add_argument(
        "--face-limit",
        dest="face_limit",
        type=int,
        default=None,
        help="Target triangle count for decimation (0 disables decimation)",
    )
    parser.add_argument("--output-format", dest="output_format", type=str, default=None, choices=["glb", "obj", "ply", "stl"])
    parser.add_argument("--low-vram", dest="low_vram", action="store_true", default=None)
    parser.add_argument("--paint-max-num-view", dest="paint_max_num_view", type=int, default=None)
    parser.add_argument("--paint-resolution", dest="paint_resolution", type=int, default=None)
    parser.add_argument("--input-dir", dest="input_dir", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)


def resolve_config(args: argparse.Namespace) -> PipelineConfig:
    """Load config.yaml (path from --config) and apply CLI overrides."""
    config = load_config(args.config)

    # --face-limit 0 means "disable decimation" -> map to None.
    if getattr(args, "face_limit", None) == 0:
        args.face_limit = None
        config = dataclasses.replace(config, face_limit=None)

    return apply_cli_overrides(config, args)
