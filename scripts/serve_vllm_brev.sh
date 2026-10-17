#!/usr/bin/env bash
# Serve the coordinator with vLLM on an NVIDIA Brev GPU instance (Red Hat vLLM path).
# Then point the local coordinator at it:
#   UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://<brev-host>:8001 \
#   UNSPOKEN_LLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 ./scripts/run_all.sh
# Only structured evidence summaries (candidate text + quality flags) leave the Mac —
# never camera frames, audio, or EEG. The UI must disclose cloud processing when this is on.
set -euo pipefail
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
pip install -U "vllm>=0.11"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "${PORT:-8001}" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-prefix-caching \
  --max-num-seqs 32
