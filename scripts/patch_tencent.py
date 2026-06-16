"""Patches for known bugs in the Tencent Hunyuan3D-2.1 repository.

Run once after cloning the Tencent repo:
    python scripts/patch_tencent.py C:/Users/Shadow/Hunyuan3D-2.1
"""

import sys
from pathlib import Path


def patch_file(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            print(f"  [SKIP] {label} — already patched")
            return True
        print(f"  [WARN] {label} — pattern not found, skipping")
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  [OK]   {label}")
    return True


def patch_mesh_render(repo: Path) -> None:
    target = repo / "hy3dpaint" / "DifferentiableRenderer" / "MeshRender.py"
    if not target.exists():
        print(f"  [WARN] MeshRender.py not found at {target}")
        return
    # Line ~1182: depth[visible_mask > 0] may be empty when no pixel is visible
    # from a given camera angle, causing max()/min() to fail on empty tensors.
    old = "depth_max, depth_min = depth[visible_mask > 0].max(), depth[visible_mask > 0].min()"
    new = (
        "_vd = depth[visible_mask > 0]; "
        "depth_max, depth_min = (_vd.max(), _vd.min()) if _vd.numel() > 0 "
        "else (depth.max(), depth.max())"
    )
    patch_file(target, old, new, "MeshRender.py: empty visible_mask guard")


def patch_basicsr(conda_env: Path) -> None:
    target = conda_env / "Lib" / "site-packages" / "basicsr" / "data" / "degradations.py"
    if not target.exists():
        print(f"  [WARN] degradations.py not found at {target}")
        return
    old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    new = "from torchvision.transforms.functional import rgb_to_grayscale"
    patch_file(target, old, new, "basicsr/degradations.py: functional_tensor -> functional")


def patch_lightning_fabric(conda_env: Path) -> None:
    target = conda_env / "Lib" / "site-packages" / "lightning_fabric" / "__init__.py"
    if not target.exists():
        print(f"  [WARN] lightning_fabric/__init__.py not found at {target}")
        return
    old = '__import__("pkg_resources").declare_namespace(__name__)'
    new = "# patched: pkg_resources.declare_namespace not required"
    patch_file(target, old, new, "lightning_fabric: remove pkg_resources.declare_namespace")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/patch_tencent.py <path-to-Hunyuan3D-2.1>")
        print("Example: python scripts/patch_tencent.py C:/Users/Shadow/Hunyuan3D-2.1")
        sys.exit(1)

    repo = Path(sys.argv[1])
    if not repo.exists():
        print(f"ERROR: repo path does not exist: {repo}")
        sys.exit(1)

    # Infer conda env from Python executable location.
    conda_env = Path(sys.executable).parent.parent
    print(f"Repo    : {repo}")
    print(f"Conda env: {conda_env}\n")

    print("Patching Tencent repo...")
    patch_mesh_render(repo)

    print("\nPatching conda env site-packages...")
    patch_basicsr(conda_env)
    patch_lightning_fabric(conda_env)

    print("\nDone.")


if __name__ == "__main__":
    main()
