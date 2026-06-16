# Guide d'installation - Windows 11 natif (Shadow PC Power)

Cible : **Shadow PC Power**, GPU **NVIDIA RTX A4500 (20 Go VRAM)**, **28 Go
RAM**, **Windows 11 natif** (pas de WSL2). Machine vierge : ce guide part de
zéro.

## 1. Prérequis système

### Miniconda
Installeur Windows : https://www.anaconda.com/download/success (section
*Miniconda*). Installe Python 3.10+ et `conda`.

### Git pour Windows
https://git-scm.com/download/win

### Visual Studio Build Tools (MSVC + SDK)
Nécessaires pour compiler les kernels CUDA custom de la partie texture
(`custom_rasterizer` / `hy3dpaint`).
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Lors de l'installation, cocher impérativement **« Développement Desktop en
C++ »** (inclut MSVC + Windows 10/11 SDK).

### Pilote NVIDIA
Normalement déjà présent sur Shadow (machine gaming). Vérifier dans un
terminal :

```bash
nvidia-smi
```

Si la commande échoue ou ne détecte pas le GPU, installer/mettre à jour le
pilote depuis : https://www.nvidia.com/Download/index.aspx

### CUDA Toolkit
PyTorch embarque ses propres binaires CUDA via le wheel `pip` : **une
installation CUDA système complète n'est en général PAS nécessaire pour
l'inférence**. Le CUDA Toolkit complet (avec `nvcc`) n'est requis que si la
compilation des kernels custom l'exige explicitement :
https://developer.nvidia.com/cuda-downloads

## 2. Création de l'environnement

```bash
conda create -n hy3d python=3.10 -y
conda activate hy3d
```

## 3. Installation de PyTorch avec CUDA

Utiliser le sélecteur officiel pour obtenir la commande exacte adaptée à votre
pilote (généralement `cu121` ou `cu124`) :
https://pytorch.org/get-started/locally/

Exemple (à adapter selon le sélecteur) :

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Vérification immédiate :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Doit afficher `True` et `NVIDIA RTX A4500`.

## 4. Récupération du modèle Hunyuan3D-2.1

- **Dépôt GitHub officiel** (code + instructions) :
  https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- **Organisation GitHub** (écosystème complet) :
  https://github.com/Tencent-Hunyuan
- **Poids HuggingFace** (modèle principal) :
  https://huggingface.co/tencent/Hunyuan3D-2.1
  - Fallback 2.0 si besoin : https://huggingface.co/tencent/Hunyuan3D-2
- **Segmentation par parties** (utile pour véhicules : roues / carrosserie /
  vitres) : https://github.com/Tencent-Hunyuan/Hunyuan3D-Part

### Clés de recherche (si les URLs ci-dessus changent)
- HuggingFace : `tencent Hunyuan3D-2.1`
- GitHub : `Tencent-Hunyuan Hunyuan3D-2.1`
- Segmentation : `Tencent-Hunyuan Hunyuan3D-Part`

### Clone et installation des dépendances du repo Tencent

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1
pip install -r requirements.txt
# Compilation des composants texture (custom rasterizer) :
# suivre les commandes du README du repo Tencent (hy3dpaint/custom_rasterizer
# selon la version) - elles évoluent, ne pas les recopier en dur ici.
```

> Le README du repo Tencent contient les commandes exactes de build des
> sous-modules de texture. **Toujours suivre la version du README au moment
> du clone**, car ces étapes changent fréquemment entre releases.

### Téléchargement des poids

Les poids se téléchargent **automatiquement au premier lancement** via
`huggingface_hub`, dans :
- Linux/macOS : `~/.cache/huggingface`
- Windows : `%USERPROFILE%\.cache\huggingface`

Compter **10+ Go**. Si un modèle nécessite l'acceptation de conditions
d'utilisation sur HuggingFace, se connecter au préalable :

```bash
huggingface-cli login
```

Sur Shadow, le disque système peut être limité : rediriger le cache vers un
disque avec plus d'espace via la variable d'environnement `HF_HOME` (Shadow
permet d'étendre le stockage jusqu'à 5 To) :

```bash
setx HF_HOME "D:\hf_cache"
```

## 5. Installation du pipeline (ce projet)

```bash
git clone https://github.com/zionhigt/hunyuan3d-batch-pipeline.git
cd hunyuan3d-batch-pipeline
pip install -r requirements.txt
```

Ce projet importe `hy3dshape` et `hy3dpaint` depuis le repo Tencent cloné à
l'étape 4. Deux options pour le lier :

> **Note** : le repo Tencent ne contient pas de `setup.py` ni de
> `pyproject.toml`, donc `pip install -e` échoue avec « does not appear to
> be a Python project ». Utiliser `conda develop` ou `PYTHONPATH` à la place.

### Option A (recommandée) : `conda develop`

`conda develop` est l'équivalent de `pip install -e` mais sans nécessiter de
`setup.py`. Il ajoute le chemin dans l'env conda de façon permanente.

```bash
conda activate hy3d
conda develop C:\Users\Shadow\Hunyuan3D-2.1
```

Vérification :
```bash
python -c "import hy3dshape; print('OK')"
```

### Option B : `PYTHONPATH` permanent

```bash
setx PYTHONPATH "C:\Users\Shadow\Hunyuan3D-2.1"
```

Fermer et rouvrir le terminal conda pour que `setx` prenne effet. Pour la
session courante seulement : `set PYTHONPATH=C:\Users\Shadow\Hunyuan3D-2.1`.

### Option C : sous-module git (si le repo Tencent est dans ce projet)

```bash
git submodule add https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git Hunyuan3D-2.1
git submodule update --init --recursive
conda develop Hunyuan3D-2.1
```

## 6. Vérification

```bash
python src/check_env.py
```

Doit afficher : CUDA disponible, GPU `NVIDIA RTX A4500`, VRAM totale ~20 Go,
version torch/CUDA.

## 7. Premier test

```bash
# Déposer une image dans input/, puis :
python src/single.py --image input/test.png

# ou batch complet :
python src/batch.py
```

## 8. Dépannage (Troubleshooting)

### `torch.cuda.is_available() == False`
- Vérifier le pilote avec `nvidia-smi`.
- Réinstaller PyTorch avec le bon wheel CUDA (étape 3) - une mauvaise version
  CUDA (ou un wheel CPU-only) est la cause la plus fréquente.

### Erreur de compilation des kernels texture
- Vérifier que **Visual Studio Build Tools** est installé avec le composant
  **« Développement Desktop en C++ »** (MSVC + Windows SDK).
- Relancer la compilation depuis un terminal **"x64 Native Tools Command
  Prompt for VS"** si nécessaire.

### `OutOfMemoryError` (CUDA OOM)
- Réduire `octree_resolution` (ex: 256 -> 128).
- Activer `low_vram: true` (ou `--low-vram`).
- Désactiver la texture : `enable_texture: false` (ou `--no-texture`).

### Le cache HuggingFace sature le disque
- Rediriger `HF_HOME` vers un disque avec plus d'espace (voir étape 4).

### Lenteur extrême
- Vérifier avec `nvidia-smi` pendant la génération que le GPU est bien
  utilisé (utilisation > 0%, VRAM occupée). Une utilisation à 0% indique un
  fallback CPU (souvent dû à un mauvais wheel PyTorch, voir étape 3).
