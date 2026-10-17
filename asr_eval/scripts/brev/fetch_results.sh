#!/usr/bin/env bash
# Local side: copy results (summary, plots, failures, thresholds, env, lock file) back from the instance.
# Per-clip raw CSVs stay on the instance unless --raw is given (they hold volunteers' transcripts).
#   bash asr_eval/scripts/brev/fetch_results.sh <instance> [--raw]
set -euo pipefail
INST=${1:?usage: fetch_results.sh <instance> [--raw]}
cd "$(git rev-parse --show-toplevel)/asr_eval"
REMOTE_HOME=$(brev exec "$INST" 'echo $HOME' 2>/dev/null | grep '^/' | head -1)
mkdir -p results
for f in summary.md failures.md env.md thresholds.json config_used.yaml plots; do
  brev copy "$INST:$REMOTE_HOME/asr_eval/results/$f" "results/" >/dev/null && echo "  results/$f"
done
brev copy "$INST:$REMOTE_HOME/asr_eval/requirements.lock" "requirements.lock" >/dev/null && echo "  requirements.lock"
if [[ "${2:-}" == "--raw" ]]; then
  brev copy "$INST:$REMOTE_HOME/asr_eval/results/raw" "results/" >/dev/null && echo "  results/raw (gitignored)"
fi
