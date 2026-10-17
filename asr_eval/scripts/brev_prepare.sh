#!/usr/bin/env bash
# Run on the Brev box after copying the harness: environment, datasets, TTS sanity clips.
# nohup bash scripts/brev_prepare.sh > prepare.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_brev.sh
# shellcheck disable=SC1091
source .env.brev
python scripts/fetch_data.py
python scripts/make_tts.py
python scripts/env_report.py --config configs/eval.yaml
echo "PREPARE DONE"
