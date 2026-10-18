#!/usr/bin/env bash
# One-time setup on Apple Silicon: Python env, pinned upstream repos, checkpoints, datasets, web deps.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Python 3.12 env"
uv venv --python 3.12 .venv 2>/dev/null || true
uv pip install --python .venv -r requirements.txt

PY=.venv/bin/python bash scripts/fetch_assets.sh

echo "== Web app + on-device browser models"
(cd web && npm install)
mkdir -p web/public/mediapipe web/public/models
cp -r web/node_modules/@mediapipe/tasks-vision/wasm web/public/mediapipe/
[[ -s web/public/mediapipe/face_landmarker.task ]] || curl -sSL -o web/public/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# gaze: MobileGaze MobileOne-S0 eye-image model (MIT, yakhyo/gaze-estimation)
[[ -s web/public/models/mobileone_s0_gaze.onnx ]] || curl -sSL -o web/public/models/mobileone_s0_gaze.onnx   https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx

echo "== Coordinator model (Ollama, Metal)"
if command -v ollama >/dev/null; then ollama pull nemotron-3.5-lightning:30b-a3b; else echo "  install Ollama, or set UNSPOKEN_LLM_BACKEND=openai (see scripts/serve_coordinator_mlx.sh)"; fi

echo "== Pre-train EEG decoders on calibration runs"
PYTHONPATH=$PWD .venv/bin/python -c "from services.eeg import p300; [p300.train_and_evaluate(s) for s in p300.available_subjects()]"
echo "Done. Start everything with ./scripts/run_all.sh"
