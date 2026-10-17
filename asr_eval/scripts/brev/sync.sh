#!/usr/bin/env bash
# Local side (WSL / macOS / Linux, with the brev CLI logged in): ship the COMMITTED harness to an
# instance and unpack it over ~/asr_eval (its data symlink, logs and results are left alone).
#   bash asr_eval/scripts/brev/sync.sh <instance>
set -euo pipefail
INST=${1:?usage: sync.sh <instance>}
cd "$(git rev-parse --show-toplevel)"
REMOTE_HOME=$(brev exec "$INST" 'echo $HOME' 2>/dev/null | grep '^/' | head -1)
git archive --format=tar.gz -o /tmp/asr_eval.tgz HEAD asr_eval
brev copy /tmp/asr_eval.tgz "$INST:$REMOTE_HOME/asr_eval.tgz" >/dev/null
for f in asr_eval/scripts/brev/*.sh; do brev copy "$f" "$INST:$REMOTE_HOME/$(basename "$f")" >/dev/null; done
brev exec "$INST" 'cd ~ && tar xzf asr_eval.tgz && echo "synced $(date +%H:%M:%S)"' | grep synced
