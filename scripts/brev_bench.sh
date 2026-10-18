#!/usr/bin/env bash
# Benchmarks on an NVIDIA Brev GPU instance (Linux + CUDA). Results land in evaluations/results/brev/.
#   ./scripts/brev_bench.sh setup        Python env + Auto-AVSR checkpoints (no web app, no Ollama)
#   ./scripts/brev_bench.sh coordinator  Nemotron on vLLM -> NeMo Gym tasks (raw model vs deployed guard)
#   ./scripts/brev_bench.sh vsr          lip reading: original vs fine-tuned on all GRID training speakers
# Env: MODEL (vLLM model id), GRID_TRAIN (speakers), PER_SPEAKER (clips per training speaker), JOBS
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=.venv/bin/python
OUT=evaluations/results/brev
mkdir -p "$OUT" .run/logs
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
TEST="s1 s2"
# GRID has no video for s21; s1/s2 stay held out for testing
GRID_TRAIN=${GRID_TRAIN:-$(for i in $(seq 3 34); do [[ $i == 21 ]] || printf 's%d ' $i; done)}

case "${1:-}" in
setup)
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  WITH_WEB=0 WITH_OLLAMA=0 GRID_SPEAKERS="$TEST" ./scripts/setup_models.sh
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"
  ;;

coordinator)
  # vLLM pins its own torch, so it gets its own venv
  [[ -x .venv-vllm/bin/vllm ]] || { uv venv --python 3.12 .venv-vllm && uv pip install --python .venv-vllm "vllm>=0.11"; }
  PATH="$PWD/.venv-vllm/bin:$PATH" MODEL="$MODEL" ./scripts/serve_vllm_brev.sh >.run/logs/vllm.log 2>&1 &
  VLLM=$!
  trap 'kill $VLLM 2>/dev/null || true' EXIT
  echo "waiting for vLLM ($MODEL) — first start downloads the weights"
  until curl -sf localhost:8001/v1/models >/dev/null; do
    kill -0 $VLLM 2>/dev/null || { echo "vLLM exited; see .run/logs/vllm.log"; tail -20 .run/logs/vllm.log; exit 1; }
    sleep 10
  done
  UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:8001 UNSPOKEN_LLM_MODEL="$MODEL" \
    $PY evaluations/nemo_gym/run_local.py --policies "raw:$MODEL" guarded deterministic
  cp evaluations/results/coordinator.json "$OUT/coordinator_vllm.json"
  ;;

vsr)
  GRID_SPEAKERS="$TEST $GRID_TRAIN" WITH_WEB=0 WITH_OLLAMA=0 ./scripts/setup_models.sh >/dev/null
  export UNSPOKEN_VSR_DEVICE=cuda
  # face detection is CPU-bound, so extract several speakers in parallel
  for s in $TEST; do echo "$s 100"; done >.run/extract.txt
  for s in $GRID_TRAIN; do echo "$s ${PER_SPEAKER:-500}"; done >>.run/extract.txt
  xargs -P "${JOBS:-8}" -L 1 sh -c "$PY scripts/finetune_vsr_grid.py extract --speakers \$0 --per-speaker \$1 2>&1 | grep -E 'extracted|Traceback'" <.run/extract.txt
  $PY scripts/finetune_vsr_grid.py train --train $GRID_TRAIN --epochs 4 --batch-size 32
  $PY scripts/finetune_vsr_grid.py eval --test $TEST
  $PY scripts/finetune_vsr_grid.py phrases --test $TEST
  $PY scripts/finetune_vsr_grid.py phrases --test $TEST --sizes 10 50 --max-diff 1
  for f in vsr_finetune vsr_phrases vsr_phrases_hard; do cp "evaluations/results/$f.json" "$OUT/$f.json"; done
  ;;

*)
  sed -n 2,6p "$0"; exit 1
  ;;
esac
echo "results in $OUT/"
