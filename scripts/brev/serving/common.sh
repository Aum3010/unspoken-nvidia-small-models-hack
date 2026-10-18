# Shared by the intrusive serving experiments (isolation_experiment.sh, max_capacity.sh). Source it from the repo root.
#   wait_idle NAME     block until nobody used the app for IDLE_MIN minutes (re-check every 5 min, up to 2 h)
#   self_window_start / self_window_end   mark when this experiment ran (idle_check.py leaves it out)
#   restore NAME       start_app.sh stop && start_app.sh, wait for /api/health all ready (10 min), safety tests
OUT=evaluations/results/serving
PY=/workspace/venv-app/bin/python
IDLE_MIN=${IDLE_MIN:-10}
IDLE_MAX_WAIT_MIN=${IDLE_MAX_WAIT_MIN:-120}
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
export UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:8001 UNSPOKEN_LLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
mkdir -p "$OUT" .run/logs

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

wait_idle() {
  local name=$1 waited=0 idle
  while :; do
    idle=$($PY scripts/brev/serving/idle_check.py)
    log "$name: last use of the app ${idle}s ago (need $((IDLE_MIN * 60))s)"
    ((idle >= IDLE_MIN * 60)) && return 0
    ((waited >= IDLE_MAX_WAIT_MIN)) && { IDLE_SKIP_REASON="app in use: last use ${idle}s ago after waiting ${waited} min"; return 1; }
    sleep 300; waited=$((waited + 5))
  done
}

SELF_T0=
self_window_start() { SELF_T0=$(date +%s); echo "$SELF_T0" >> .run/serving_self_windows.txt; }
self_window_end() {  # replace this run's open "start" line by "start end"
  [[ -n $SELF_T0 ]] && sed -i "s/^$SELF_T0\$/$SELF_T0 $(date +%s)/" .run/serving_self_windows.txt
}

all_ready() {  # every service in /api/health is ready
  curl -s -m 5 http://127.0.0.1:8000/api/health | $PY -c '
import json, sys
h = json.load(sys.stdin)
bad = [k for k, v in h.items() if isinstance(v, dict) and v.get("state") != "ready"]
print("not ready:", bad) if bad else None
sys.exit(1 if bad else 0)' 2>/dev/null
}

# Writes $OUT/restore_<name>.json; returns 0 only if the app is back at defaults, all ready, safety tests pass.
restore() {
  local name=$1 t0 ready=0 safety=1 safety_line
  t0=$(date +%s)
  log "$name: RESTORE: start_app.sh stop && start_app.sh (defaults)"
  cp .run/logs/gateway.log ".run/logs/gateway.pre_${name}_$t0.log" 2>/dev/null
  bash scripts/brev/start_app.sh stop
  # anything of vLLM not in a pid file (a relaunch that died half way) would hold port 8001
  pkill -f '[v]llm serve' 2>/dev/null; pkill -9 -f 'VLLM::EngineCore' 2>/dev/null; sleep 3
  rm -f .run/serving_vllm_orig.json
  env -u UNSPOKEN_LLM_BACKEND -u UNSPOKEN_LLM_BASE_URL -u UNSPOKEN_LLM_MODEL -u PYTHONPATH bash scripts/brev/start_app.sh
  for _ in $(seq 1 120); do
    sleep 5
    all_ready && { ready=1; break; }
  done
  safety_line=$($PY -m pytest -q evaluations/safety_tests 2>&1 | tail -n 1); [[ $safety_line == *passed* && $safety_line != *failed* ]] && safety=0
  local status; status=$(bash scripts/brev/start_app.sh status 2>&1)
  local mtp=0; [[ $status == *"MTP on"* ]] && mtp=1
  local mps=0; [[ $status == *"MPS on"* ]] && mps=1
  local args; args=$(pgrep -af '[v]llm serve' | head -1 | sed 's/.* serve //')
  $PY - "$name" "$ready" "$safety" "$safety_line" "$mtp" "$mps" "$args" "$(( $(date +%s) - t0 ))" <<'PY'
import json, sys
name, ready, safety, line, mtp, mps, args, secs = sys.argv[1:]
default = "--max-num-seqs 8" in args and "--gpu-memory-utilization 0.55" in args and "speculative" not in args
res = {"experiment": name, "all_ready": ready == "1", "safety_tests": line, "safety_ok": safety == "0", "mtp_on": mtp == "1",
       "mps_on": mps == "1", "vllm_args": args, "vllm_default_args": default, "seconds": int(secs)}
res["ok"] = res["all_ready"] and res["safety_ok"] and not res["mtp_on"] and res["mps_on"] and default
open(f"evaluations/results/serving/restore_{name}.json", "w").write(json.dumps(res, indent=2))
print("RESTORE", "OK" if res["ok"] else "FAILED", json.dumps(res))
sys.exit(0 if res["ok"] else 1)
PY
}
