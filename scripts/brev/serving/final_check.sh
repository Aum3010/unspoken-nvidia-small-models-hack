#!/usr/bin/env bash
# After the serving experiments: is the app back at its defaults? Writes evaluations/results/serving/final_check.json.
#   /api/health all ready · safety tests · flags at default (no UNSPOKEN_SERVING_PANEL, MTP off, vLLM's default
#   args, MPS on) · smoke through the live gateway (accepted voice auto-spoken, ambiguous waits + No, Yes speaks)
#   bash scripts/brev/serving/final_check.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
source scripts/brev/serving/common.sh
ready=0; all_ready && ready=1
safety=$($PY -m pytest -q evaluations/safety_tests 2>&1 | tail -n 1)
status=$(bash scripts/brev/start_app.sh status 2>&1)
args=$(pgrep -af '[v]llm serve' | head -1 | sed 's/.* serve //')
flags=$(for n in gateway coordinator audio vsr tts eeg vllm; do
  p=$(cat .run/pids/$n.pid 2>/dev/null) && tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -E '^(UNSPOKEN_|LLM_MTP|CUDA_MPS_)' | grep -vE '^UNSPOKEN_(LLM_BACKEND|LLM_BASE_URL|LLM_MODEL|HOST_LABEL)=' | sed "s/^/$n: /"
done)
$PY scripts/brev/serving/smoke_box.py --gateway http://127.0.0.1:8000 --all-health --checks health accepted ambiguous_no yes --out /tmp/serving_final_smoke.json
smoke=$?
$PY - "$ready" "$safety" "$status" "$args" "$flags" "$smoke" <<'PY'
import json, sys, time
ready, safety, status, args, flags, smoke = sys.argv[1:]
res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "all_ready": ready == "1", "safety_tests": safety,
       "safety_ok": "passed" in safety and "failed" not in safety, "mps_on": "MPS on" in status, "mtp_on": "MTP on" in status,
       "vllm_args": args, "vllm_default_args": "--max-num-seqs 8" in args and "--gpu-memory-utilization 0.55" in args and "speculative" not in args,
       "non_default_flags": [f for f in flags.splitlines() if f], "smoke": json.load(open("/tmp/serving_final_smoke.json"))}
res["smoke_ok"] = smoke == "0"
res["ok"] = all([res["all_ready"], res["safety_ok"], res["mps_on"], not res["mtp_on"], res["vllm_default_args"],
                 not res["non_default_flags"], res["smoke_ok"]])
open("evaluations/results/serving/final_check.json", "w").write(json.dumps(res, indent=2))
print("FINAL", "OK" if res["ok"] else "NOT OK", json.dumps({k: v for k, v in res.items() if k != "smoke"}))
PY
