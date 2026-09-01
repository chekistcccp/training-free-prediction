from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from src.data import resolve_msd_root
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    if sys.version_info < (3, 10):
        raise SystemExit("Python >=3.10 is required.")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    if "microsoft" not in platform.release().lower() and "WSL" not in os.environ:
        print("WARNING: WSL was requested, but this environment does not look like WSL. Linux is still supported.")

    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch is not installed. run.sh should install requirements first.") from e

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Verify NVIDIA Windows driver + WSL CUDA support.")
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / (1024**3)
    print(f"GPU: {props.name}")
    print(f"VRAM: {total_gib:.1f} GiB")
    print(f"CUDA runtime: {torch.version.cuda}")
    if total_gib < 40:
        raise SystemExit(
            f"Only {total_gib:.1f} GiB VRAM detected. This repository is configured for non-quantized Qwen3.5-9B on ~48 GB VRAM."
        )

    try:
        msd = resolve_msd_root(cfg["paths"]["msd_root"])
        print(f"MSD: OK ({msd})")
    except Exception as e:
        raise SystemExit(str(e))

    stageii = Path(cfg["paths"]["stageii_root"])
    if not stageii.exists():
        raise SystemExit(
            f"StageII dataset directory not found: {stageii}. Place the downloaded TCIA collection there."
        )
    print(f"StageII: OK ({stageii})")

    if str(cfg["model"]["id"]) != "Qwen/Qwen3.5-9B":
        raise SystemExit("Model id must remain exactly Qwen/Qwen3.5-9B for this experiment.")
    dtype = str(cfg["model"].get("dtype", "")).lower()
    if dtype not in {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}:
        raise SystemExit("Only non-quantized floating dtypes are permitted.")


if __name__ == "__main__":
    main()
