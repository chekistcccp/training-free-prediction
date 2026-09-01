#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/experiment.yaml}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: No active conda environment detected."
  echo "Create and activate the environment manually before running experiments."
  echo "Example:"
  echo "  conda create -n causalcrc python=3.11 -y"
  echo "  conda activate causalcrc"
  echo "  pip install -r requirements.txt"
  echo "  ./run.sh"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is unavailable in the active conda environment: ${CONDA_PREFIX}"
  exit 1
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/.modelscope_cache}"

echo "Using conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Conda prefix: ${CONDA_PREFIX}"
echo "Python: $(command -v python)"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  echo "INSTALL_DEPS=1 -> installing/updating requirements in the active conda environment"
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install -r requirements.txt
fi

python scripts/check_env.py --config "$CONFIG"
python -m pytest -q tests
python scripts/download_model.py --config "$CONFIG"
python -m src.main --config "$CONFIG" all

echo "All requested experiments completed. See outputs/summary/ for aggregated results."
