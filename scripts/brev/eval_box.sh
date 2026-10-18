#!/usr/bin/env bash
# Evaluation box (2x H100): vLLM for the 30B coordinator (GPU 0) and a 4B comparison (GPU 1), plus a NeMo venv for the
# voice agent. Run on the instance from ~/unspoken after sync. Usage: eval_box.sh setup | serve | status | stop
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${W:-$HOME/work}
export HF_HOME=$W/hf
mkdir -p "$W" .run/logs
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
BIG=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
SMALL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

case "${1:-}" in
setup)
  [[ -x $UV ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
  UV=$HOME/.local/bin/uv
  # vLLM and NeMo pin different torch builds: one venv each
  [[ -x $W/venv-vllm/bin/python ]] || "$UV" venv -q --python 3.12 "$W/venv-vllm"
  "$UV" pip install -q --python "$W/venv-vllm/bin/python" vllm huggingface_hub hf_transfer &
  [[ -x $W/venv-asr/bin/python ]] || "$UV" venv -q --python 3.12 "$W/venv-asr"
  "$UV" pip install -q --python "$W/venv-asr/bin/python" "nemo_toolkit[asr]" av numpy &
  # the eval scripts (Gym runner, neuro_comm, fusion) in the app's own deps, CPU torch is enough for them
  [[ -x $W/venv-app/bin/python ]] || "$UV" venv -q --python 3.12 "$W/venv-app"
  "$UV" pip install -q --python "$W/venv-app/bin/python" httpx fastapi pydantic numpy pandas scipy scikit-learn pyriemann==0.7 \
    mne==1.12.1 "numpy<2" matplotlib pytest &
  wait
  HF_HUB_ENABLE_HF_TRANSFER=1 "$W/venv-vllm/bin/python" -c "
from huggingface_hub import snapshot_download as s
for r in ('$BIG', '$SMALL'): print(s(r))"
  echo "setup done"
  ;;
serve)
  CUDA_VISIBLE_DEVICES=0 nohup "$W/venv-vllm/bin/vllm" serve $BIG --port 8001 --served-model-name big \
    --max-model-len 16384 --gpu-memory-utilization 0.92 --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
    > .run/logs/vllm_big.log 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 nohup "$W/venv-vllm/bin/vllm" serve $SMALL --port 8002 --served-model-name small \
    --max-model-len 16384 --gpu-memory-utilization 0.45 --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
    > .run/logs/vllm_small.log 2>&1 < /dev/null &
  echo "serving: big :8001 (GPU 0), small :8002 (GPU 1); logs .run/logs/vllm_*.log"
  ;;
status)
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  for p in 8001 8002; do printf ":%s " $p; curl -s -m 3 localhost:$p/v1/models | head -c 120; echo; done
  ;;
stop)
  pkill -f "vllm serve" || true
  ;;
*) echo "usage: $0 setup | serve | status | stop"; exit 1 ;;
esac
