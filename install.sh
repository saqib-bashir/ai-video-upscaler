#!/usr/bin/env bash
# Linux / macOS setup. Needs python3 (3.10+) and ffmpeg on PATH.
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "Install ffmpeg first (apt install ffmpeg / brew install ffmpeg)"; exit 1; }
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
if command -v nvidia-smi >/dev/null; then
  pip install torch --index-url https://download.pytorch.org/whl/cu128
fi
pip install -e .
aiupscale models --download general general-wdn x2plus
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
