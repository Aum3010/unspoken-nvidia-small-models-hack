#!/usr/bin/env bash
# Laptop: copy evaluations/results/serving/ (JSON, markdown, plots, GPU trace) from the Brev box into this checkout.
# Uses the ssh host alias that `brev` writes (as connect.sh does). Usage: fetch_results.sh <instance>
set -euo pipefail
INST=${1:?usage: fetch_results.sh <instance>}
cd "$(dirname "$0")/../../.."
ssh -o LogLevel=ERROR "$INST" 'tar -C ~/unspoken -czf - evaluations/results/serving' | tar -xzf -
ls -la evaluations/results/serving
