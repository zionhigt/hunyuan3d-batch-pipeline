# hunyuan3d-batch-pipeline

Pipeline d'orchestration autour de **Hunyuan3D-2.1** (Tencent) pour générer en
batch des assets 3D `.glb` texturés PBR à partir d'un dossier d'images. Ce
projet ne réimplémente pas le modèle : il charge les pipelines officiels
(`Hunyuan3DDiTFlowMatchingPipeline` pour la géométrie, `Hunyuan3DPaintPipeline`
pour la texture) et automatise génération, reprise, logs et export GLB.

Conçu pour s'insérer **en amont** d'un pipeline Blender headless existant
(retopo / rigging / animation → export Godot 4) : la sortie `.glb` est propre
et directement scriptable.

## Installation

Voir [`INSTALL.md`](INSTALL.md) pour la procédure complète (Windows 11 natif,
testé sur Shadow PC Power - RTX A4500 20 Go VRAM).

## Quickstart

```bash
conda activate hy3d
python src/check_env.py              # vérifie CUDA / VRAM / torch
# déposer des images dans input/ (png, jpg, jpeg, webp)
python src/single.py --image input/test.png   # test sur une image
python src/batch.py                            # génère tout input/ -> output/
```

## Flux

```
input/*.{png,jpg,jpeg,webp}
        │
        ▼
  Hunyuan3D-2.1 (DiT shape + Paint texture PBR)
        │
        ▼
output/*.glb
        │
        ▼
  [pipeline Blender headless externe : retopo/rigging/anim → Godot 4]
```

## Configuration (`config.yaml`)

Tous les paramètres peuvent être surchargés en ligne de commande (voir
`python src/batch.py --help` / `python src/single.py --help`).

| Clé | Défaut | Description |
| --- | --- | --- |
| `model_path` | `tencent/Hunyuan3D-2.1` | ID HuggingFace ou chemin local des poids |
| `enable_texture` | `true` | Génère la texture PBR (`false` = géométrie seule, plus rapide, moins de VRAM) |
| `octree_resolution` | `256` | Résolution de l'octree (qualité géométrie) : 128/256/384... |
| `num_inference_steps` | `30` | Nombre de pas de diffusion (plus = plus précis, plus lent) |
| `guidance_scale` | `7.5` | Guidance scale (CFG) |
| `face_limit` | `40000` | Décimation finale (triangles). `null` = mesh brut high-poly |
| `output_format` | `glb` | Format de sortie (`glb`, `obj`, `ply`, `stl`) |
| `low_vram` | `false` | Active l'offload modèle entre étapes (GPU < 16 Go) |
| `input_dir` | `input` | Dossier d'images source |
| `output_dir` | `output` | Dossier de sortie des `.glb` |

## Structure

```
hunyuan3d-batch-pipeline/
├─ src/
│  ├─ pipeline.py   # wrapper Hunyuan3D-2.1 (shape + paint -> mesh -> export)
│  ├─ batch.py      # traitement de input/ vers output/, reprise, logs
│  ├─ single.py     # génération unitaire (debug/tests)
│  ├─ config.py     # config.yaml + surcharge CLI
│  └─ check_env.py  # diagnostic CUDA/VRAM
├─ config.yaml
├─ input/   # images source (ignoré par git)
├─ output/  # GLB générés (ignoré par git)
└─ scripts/run_batch.bat
```

## Matériel

Testé sur **Shadow PC Power** (RTX A4500, 20 Go VRAM, 28 Go RAM, Windows 11
natif). Fonctionne à partir de ~16 Go VRAM ; passer `low_vram: true` (ou
`--low-vram`) en dessous.
