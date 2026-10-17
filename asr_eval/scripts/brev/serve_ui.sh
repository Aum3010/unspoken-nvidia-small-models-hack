#!/usr/bin/env bash
# Instance side: (re)start the live test bench on 127.0.0.1:8800 in the background.
#   bash ~/serve_ui.sh [model] [method]       e.g. bash ~/serve_ui.sh parakeet-ctc-0.6b ctc_sum
# Laptop: brev port-forward <instance> --port 8800:8800, then open http://localhost:8800
cd ~/asr_eval
# shellcheck disable=SC1091
source .env.brev
pip install -q fastapi uvicorn python-multipart av 2>/dev/null
pkill -f "serve.py" 2>/dev/null; sleep 1
nohup python serve.py --config configs/eval.yaml --model "${1:-parakeet-ctc-0.6b}" --method "${2:-ctc_sum}" --port 8800 > serve.log 2>&1 < /dev/null &
for _ in $(seq 1 60); do grep -q "ready:" serve.log && break; sleep 2; done
grep -E "ready:|Error|Traceback" serve.log | tail -3
