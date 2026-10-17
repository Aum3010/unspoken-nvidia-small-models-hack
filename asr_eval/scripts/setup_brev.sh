#!/usr/bin/env bash
# One-time setup on a Brev GPU instance (H100 / RTX PRO 6000). Audio models only.
# Run inside tmux: tmux new -s asr 'bash scripts/setup_brev.sh 2>&1 | tee setup.log'
set -euo pipefail
cd "$(dirname "$0")/.."

WORK=${WORK:-/workspace}
if ! mkdir -p "$WORK" 2>/dev/null || [[ ! -w "$WORK" ]]; then
  { sudo -n mkdir -p "$WORK" && sudo -n chown "$USER" "$WORK"; } 2>/dev/null || WORK="$HOME/workspace"
fi
mkdir -p "$WORK/hf" "$WORK/asr_eval_data"
export HF_HOME="$WORK/hf"
echo "WORK=$WORK  HF_HOME=$HF_HOME"

if command -v sudo >/dev/null && command -v apt-get >/dev/null; then
  { sudo -n apt-get update -qq && sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev tmux espeak-ng ffmpeg; } >/dev/null     || echo "  (apt packages skipped)"
fi

python3 -m venv "$WORK/venv-asr"
# shellcheck disable=SC1091
source "$WORK/venv-asr/bin/activate"
pip install -q -U pip wheel
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python - <<'PY' > "$WORK/torch-constraint.txt"
import torch, torchaudio
print(f"torch=={torch.__version__}\ntorchaudio=={torchaudio.__version__}")
PY
pip install -q -r requirements.in -c "$WORK/torch-constraint.txt"
pip freeze > requirements.lock

ln -sfn "$WORK/asr_eval_data" data
cat > .env.brev <<EOF
export WORK=$WORK
export HF_HOME=$HF_HOME
source $WORK/venv-asr/bin/activate
EOF
python scripts/env_report.py --config configs/eval.yaml
echo "Done. Next:  source .env.brev && python run_eval.py --config configs/eval.yaml"
