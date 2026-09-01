from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils import load_yaml


def valid_model_dir(path: Path) -> bool:
    cfg = path / "config.json"
    if not cfg.exists():
        return False
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return False
    model_type = str(data.get("model_type", ""))
    weights = list(path.glob("*.safetensors")) + list(path.glob("**/*.safetensors"))
    return model_type == "qwen3_5" and len(weights) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download exact Qwen3.5-9B weights from ModelScope")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    model_id = str(cfg["model"]["id"])
    local_dir = Path(cfg["model"]["local_dir"])
    if model_id != "Qwen/Qwen3.5-9B":
        raise SystemExit(f"Refusing model id {model_id}; this project requires Qwen/Qwen3.5-9B exactly.")

    if valid_model_dir(local_dir):
        print(f"Model already present: {local_dir}")
        return

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download

    print(f"Downloading {model_id} from ModelScope to {local_dir} ...")
    snapshot_download(model_id, local_dir=str(local_dir))
    if not valid_model_dir(local_dir):
        raise SystemExit("Model download completed but validation failed (qwen3_5 config/safetensors missing).")
    print("Model download validated. No quantized weights or quantization loader is used by this repository.")


if __name__ == "__main__":
    main()
