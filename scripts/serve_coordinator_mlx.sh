#!/usr/bin/env bash
# Alternative local coordinator backend: Nemotron 3.5 Lightning on MLX (Metal), OpenAI-compatible API.
#   UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:8090 \
#   UNSPOKEN_LLM_MODEL=mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit ./scripts/run_all.sh
cd "$(dirname "$0")/.."
exec .venv/bin/python -m mlx_lm server --model mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit --port 8090
