#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CONFIG="${CONFIG:-configs/experiment.yaml}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
if [[ ! -f "$VENV_DIR/.requirements-installed" || "${FORCE_INSTALL:-0}" == "1" ]]; then
  python -m pip install -r requirements.txt
  touch "$VENV_DIR/.requirements-installed"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/.modelscope_cache}"

python scripts/check_env.py --config "$CONFIG"
python -m pytest -q tests
python scripts/download_model.py --config "$CONFIG"
python -m src.main --config "$CONFIG" all

echo "All requested experiments completed. See outputs/summary/ for aggregated results."
