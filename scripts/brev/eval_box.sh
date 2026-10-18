#!/usr/bin/env bash
# Evaluation box (2x H100): vLLM for the 30B coordinator on GPU 0 (neuro_comm, E4) and a NeMo venv for the voice
# agent on GPU 1 (lips+voice fusion, E1). Run on the instance from ~/unspoken after sync. Usage: eval_box.sh setup | serve | status | stop
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${W:-$HOME/work}
export HF_HOME=$W/hf
# Scaleway H100 driver is CUDA 12.8; current torch/vLLM wheels are CUDA 13. The forward-compat libcuda (extracted, no
# sudo) lets the 12.8 driver run them: compat=$W/compat from cuda-compat-13-0_*.deb via dpkg -x
export LD_LIBRARY_PATH=$W/compat/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$W" .run/logs
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
BIG=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

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
print(s('$BIG'))"
  echo "setup done"
  ;;
serve)
  # no nvcc on the image: FlashInfer would JIT-compile, so Triton attention and the PyTorch sampler (docs/BREV.md)
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 nohup "$W/venv-vllm/bin/vllm" serve $BIG --port 8001 --served-model-name big \
    --attention-backend TRITON_ATTN \
    --max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.92 --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
    > .run/logs/vllm_big.log 2>&1 < /dev/null &
  echo "serving: big :8001 (GPU 0); GPU 1 is free for the voice agent (E1); log .run/logs/vllm_big.log"
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
