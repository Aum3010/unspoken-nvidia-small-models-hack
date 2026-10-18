#!/usr/bin/env bash
# (Re)start the whole Unspoken backend on the Brev instance, detached from SSH.
#   vLLM (Nemotron 3.5 Lightning NVFP4) 127.0.0.1:8001 -> coordinator 8105
#   gateway 8000 · vsr 8101 · audio 8102 · eeg 8103 · tts 8104
# Everything binds 127.0.0.1: the laptop reaches the gateway only through `brev port-forward`
# (scripts/brev/connect.sh). `start_app.sh stop` stops everything; `start_app.sh status` checks health.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=${HF_HOME:-/workspace/hf}
APP=/workspace/venv-app/bin/python
# the voice agent (Nemotron Speech + asr_eval's decision code) runs in the NeMo venv built by asr_eval/scripts/setup_brev.sh
ASR=/workspace/venv-asr/bin/python
LLM=/workspace/venv-vllm/bin/vllm
LLM_MODEL=${UNSPOKEN_LLM_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
# the GPU is shared with the agents (and the asr_eval bench): keep vLLM well under the whole card
LLM_MEM=${LLM_MEM:-0.55}
# FlashInfer's attention JIT-compiles with nvcc on first use and the image has no CUDA toolkit;
# Triton attention needs no compiler. Set LLM_ATTN=FLASHINFER once a toolkit matching the driver is installed.
LLM_ATTN=${LLM_ATTN:-TRITON_ATTN}
LOGS=.run/logs
mkdir -p "$LOGS" .run/pids

stop() {
  for f in .run/pids/*.pid; do
    [[ -e $f ]] || continue
    kill "$(cat "$f")" 2>/dev/null && echo "  stopped $(basename "$f" .pid)"
    rm -f "$f"
  done
}

status() {
  printf '  8000  '; curl -s -m 3 http://127.0.0.1:8000/api/health | cut -c1-80; echo
  for p in 8101 8102 8103 8104 8105; do
    printf '  %s  ' "$p"; curl -s -m 3 "http://127.0.0.1:$p/health" | cut -c1-160; echo
  done
  printf '  8001  vllm '; curl -s -m 3 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/v1/models
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

case "${1:-start}" in
  stop) stop; exit 0 ;;
  status) status; exit 0 ;;
esac

stop
launch() {  # name, then the command
  local name=$1; shift
  nohup "$@" >"$LOGS/$name.log" 2>&1 </dev/null &
  echo $! >".run/pids/$name.pid"
  echo "  • $name (pid $!, log $LOGS/$name.log)"
}

echo "Starting Unspoken backend on $(hostname)"
# FlashInfer's sampler JIT-compiles too; the PyTorch sampler needs no compiler (greedy decoding anyway)
launch vllm env VLLM_USE_FLASHINFER_SAMPLER=0 "$LLM" serve "$LLM_MODEL" --host 127.0.0.1 --port 8001 \
  --max-model-len 16384 --gpu-memory-utilization "$LLM_MEM" --attention-backend "$LLM_ATTN" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
  --enable-prefix-caching --max-num-seqs 8

export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
export UNSPOKEN_HOST_LABEL="NVIDIA Brev"
export UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:8001 UNSPOKEN_LLM_MODEL="$LLM_MODEL"
for svc in gateway:8000 coordinator:8105 eeg:8103 tts:8104 audio:8102 vsr:8101; do
  py=$APP; [[ ${svc%%:*} == audio ]] && py=$ASR
  launch "${svc%%:*}" "$py" -m uvicorn "services.${svc%%:*}.app:app" --host 127.0.0.1 --port "${svc##*:}" --log-level warning
done
echo "Models load in the background (vLLM takes a few minutes the first time). Check: bash scripts/brev/start_app.sh status"
