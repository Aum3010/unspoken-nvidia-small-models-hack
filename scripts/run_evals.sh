#!/usr/bin/env bash
# Run every evaluation against the running services (start them with scripts/run_all.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=.venv/bin/python
echo "== weights ledger";            $PY scripts/verify_weights.py
echo "== safety unit tests";         $PY -m pytest -q
echo "== end-to-end exactness";      $PY evaluations/test_cases/e2e_exactness.py
echo "== weak speech";               $PY evaluations/modality_baselines/eval_audio.py
echo "== lip reading (GRID)";        $PY evaluations/modality_baselines/eval_vsr_grid.py "${VSR_PER_SPEAKER:-20}"
if [[ -f data/cache/vsr_ft/decoder_ctc_ft.pt ]]; then
  echo "== lip reading: original vs fine-tuned vs phrase list"
  $PY scripts/finetune_vsr_grid.py eval --test s1 s2
  $PY scripts/finetune_vsr_grid.py phrases --test s1 s2
  $PY scripts/finetune_vsr_grid.py phrases --test s1 s2 --sizes 10 50 --max-diff 1
fi
echo "== coordinator (NeMo Gym env)"; $PY evaluations/nemo_gym/run_local.py
echo "== summary";                   $PY evaluations/build_summary.py
