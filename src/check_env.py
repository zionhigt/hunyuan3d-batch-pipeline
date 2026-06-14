"""Diagnostic script: checks PyTorch/CUDA availability and GPU memory.

Usage:
    python src/check_env.py
"""

from __future__ import annotations

import sys


def main() -> int:
    print("=== Hunyuan3D batch pipeline - environment check ===\n")

    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed.")
        print("       See INSTALL.md section 3 (https://pytorch.org/get-started/locally/)")
        return 1

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA build      : {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available  : {cuda_available}")

    if not cuda_available:
        print("\n[FAIL] torch.cuda.is_available() == False")
        print("       -> Check NVIDIA driver (nvidia-smi) and reinstall the")
        print("          correct CUDA wheel (see INSTALL.md, Troubleshooting).")
        return 1

    device_count = torch.cuda.device_count()
    print(f"GPU count       : {device_count}")

    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(idx)
        free_mem_gb = free_mem_bytes / (1024 ** 3)
        print(f"\nGPU {idx}: {name}")
        print(f"  Total VRAM    : {total_mem_gb:.1f} GB")
        print(f"  Free VRAM now : {free_mem_gb:.1f} GB")

    print("\n[OK] Environment looks ready for Hunyuan3D-2.1 inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
