#!/usr/bin/env bash
# One-time setup of the whole Unspoken backend on a Brev CUDA instance (idempotent).
# Runs on the instance from ~/unspoken (shipped by scripts/brev/sync_app.sh). ~15-30 min the first time.
#   /workspace/venv-app   gateway + agents (torch cu128, faster-whisper, kokoro, Chaplin deps, EEG)
#   /workspace/venv-vllm  vLLM for the Nemotron coordinator (its own torch pins)
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=${HF_HOME:-/workspace/hf}
UV=${UV:-$HOME/.local/bin/uv}
APP=/workspace/venv-app
LLM=/workspace/venv-vllm
TORCH=${TORCH:-2.11.0}

echo "== system packages"
need=()
for p in ffmpeg espeak-ng libsndfile1 unzip; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
if ((${#need[@]})); then sudo -n apt-get update -qq && sudo -n apt-get install -y -qq "${need[@]}"; fi

echo "== app venv (Python 3.12, torch $TORCH + cu128)"
[[ -x $APP/bin/python ]] || "$UV" venv -q --python 3.12 "$APP"
"$UV" pip install -q --python "$APP/bin/python" "torch==$TORCH" "torchaudio==$TORCH" torchvision --index-url https://download.pytorch.org/whl/cu128
"$UV" pip freeze --python "$APP/bin/python" | grep -E "^torch(audio|vision)?==" > /workspace/app-torch-constraint.txt
"$UV" pip install -q --python "$APP/bin/python" -r requirements-cuda.txt -c /workspace/app-torch-constraint.txt
mkdir -p .run
"$UV" pip freeze --python "$APP/bin/python" > .run/requirements-cuda.lock

echo "== vLLM venv"
[[ -x $LLM/bin/python ]] || "$UV" venv -q --python 3.12 "$LLM"
"$LLM/bin/python" -c "import vllm" 2>/dev/null || "$UV" pip install -q --python "$LLM/bin/python" "vllm>=0.11"
"$UV" pip freeze --python "$LLM/bin/python" > .run/requirements-vllm.lock

echo "== checkpoints and data"
WITH_GRID=0 PY="$APP/bin/python" bash scripts/fetch_assets.sh

echo "== EEG decoders"
PYTHONPATH=$PWD "$APP/bin/python" -c "from services.eeg import p300; [p300.train_and_evaluate(s) for s in p300.available_subjects()]"

echo "== GPU check"
"$APP/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "Done. Start with: bash scripts/brev/start_app.sh"
