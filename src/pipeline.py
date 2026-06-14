"""Wrapper around the Tencent Hunyuan3D-2.1 pipelines.

This module does NOT reimplement the model. It orchestrates the shape
(Hunyuan3DDiTFlowMatchingPipeline) and texture (Hunyuan3DPaintPipeline)
pipelines from the official Tencent repository, which must be importable
(either as an editable install or via PYTHONPATH). See INSTALL.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import trimesh

logger = logging.getLogger(__name__)

_IMPORT_ERROR: Optional[BaseException] = None

try:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import (
        DegenerateFaceRemover,
        FaceReducer,
        FloaterRemover,
    )
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    HY3D_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - depends on external repo
    HY3D_AVAILABLE = False
    _IMPORT_ERROR = exc

_NOT_AVAILABLE_MESSAGE = (
    "Le repo Hunyuan3D-2.1 de Tencent n'est pas importable (modules "
    "'hy3dshape' / 'hy3dpaint' introuvables).\n"
    "Clonez https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 et installez-le "
    "(pip install -e ../Hunyuan3D-2.1, ou PYTHONPATH, ou sous-module git).\n"
    "Voir INSTALL.md, section 5, pour la procedure complete.\n"
    f"Erreur d'import d'origine: {_IMPORT_ERROR!r}"
)


class Hunyuan3DPipeline:
    """High-level wrapper: image -> textured 3D mesh."""

    def __init__(
        self,
        model_path: str = "tencent/Hunyuan3D-2.1",
        enable_texture: bool = True,
        device: str = "cuda",
        low_vram: bool = False,
    ) -> None:
        if not HY3D_AVAILABLE:
            raise RuntimeError(_NOT_AVAILABLE_MESSAGE)

        self.model_path = model_path
        self.enable_texture = enable_texture
        self.device = device
        self.low_vram = low_vram

        logger.info("Loading shape pipeline from '%s' on %s...", model_path, device)
        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            device=device,
        )
        if low_vram and hasattr(self.shape_pipeline, "enable_model_cpu_offload"):
            self.shape_pipeline.enable_model_cpu_offload()

        self.paint_pipeline: Optional["Hunyuan3DPaintPipeline"] = None
        if enable_texture:
            logger.info("Loading texture (paint) pipeline...")
            paint_config = Hunyuan3DPaintConfig()
            self.paint_pipeline = Hunyuan3DPaintPipeline(paint_config)

        self.face_reducer = FaceReducer()
        self.floater_remover = FloaterRemover()
        self.degenerate_remover = DegenerateFaceRemover()

    def generate(
        self,
        image_path: str,
        face_limit: Optional[int] = None,
        octree_resolution: int = 256,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> trimesh.Trimesh:
        """Generate a textured (or untextured) mesh from a single image."""
        image_path = str(image_path)
        logger.info("Generating shape for '%s'...", image_path)

        outputs = self.shape_pipeline(
            image=image_path,
            octree_resolution=octree_resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        mesh: trimesh.Trimesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        mesh = self.floater_remover(mesh)
        mesh = self.degenerate_remover(mesh)

        if face_limit:
            logger.info("Decimating mesh to %d faces...", face_limit)
            mesh = self.face_reducer(mesh, max_facenum=face_limit)

        if self.enable_texture and self.paint_pipeline is not None:
            logger.info("Generating PBR texture for '%s'...", image_path)
            mesh = self.paint_pipeline(mesh, image_path=image_path)

        return mesh

    def export(self, mesh: trimesh.Trimesh, output_path: str) -> Path:
        """Export `mesh` to `output_path` (format inferred from extension)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))
        logger.info("Exported mesh to '%s'", output_path)
        return output_path
