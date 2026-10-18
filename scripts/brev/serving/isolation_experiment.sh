#!/usr/bin/env bash
# Experiment C (intrusive, ~15 min): vLLM outside MPS (its own CUDA context) + MTP, the small models under MPS as today.
#   1. compute mode (EXCLUSIVE_PROCESS: record it, go to restore)
#   2. restart only vLLM, from its own running command line (relaunch_vllm.py), with CUDA_MPS_PIPE_DIRECTORY and
#      CUDA_MPS_LOG_DIRECTORY at an empty temp dir (so it is not an MPS client) and
#      --speculative-config '{"method":"mtp","num_speculative_tokens":1}'; verify it is not an MPS client
#   3. bench_latency.py --n 10 -> serving/latency_isolated_mtp.json
#   4. NeMo Gym held-out, guarded policy -> serving/heldout_isolated_mtp.json (baseline 0.985)
#   5. fault test once: kill -9 the EngineCore; within 10 s: voice/lips/TTS ready, accepted clip auto-spoken,
#      ambiguous clip gets the rules' question
#   6. restore (trap on EXIT/INT/TERM; ERR is logged): start_app.sh stop && start_app.sh, all ready, safety tests
# Runs only when nobody used the app for 10 min (waits up to 2 h, else skips). Defaults are never changed.
#   nohup bash scripts/brev/serving/isolation_experiment.sh > .run/logs/serving_C.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.."
source scripts/brev/serving/common.sh
R=$OUT/isolation_parts
mkdir -p "$R"
rm -f "$R"/*.json

if ! wait_idle C; then
  log "C: SKIPPED: $IDLE_SKIP_REASON"
  echo "{\"skipped\": true, \"reason\": \"$IDLE_SKIP_REASON\"}" > $OUT/isolation.json
  exit 0
fi

restored=0
finish() {
  ((restored)) && return
  restored=1
  trap - ERR
  restore C; local rc=$?
  self_window_end
  $PY scripts/brev/serving/isolation_report.py
  log "C: done (restore rc=$rc)"
  exit $rc
}
trap finish EXIT
trap 'exit 130' INT TERM
trap 'log "C: command failed (line $LINENO, rc=$?)"' ERR
self_window_start

mode=$(nvidia-smi -q | grep -i "compute mode" | head -1 | sed 's/.*: *//')
echo "{\"compute_mode\": \"$mode\"}" > "$R/compute_mode.json"
log "C: compute mode: $mode"
if [[ $mode == *Exclusive_Process* || $mode == *EXCLUSIVE_PROCESS* ]]; then
  log "C: EXCLUSIVE_PROCESS: a second context next to the MPS server is not possible; skipping to restore"
  exit 0
fi

empty=$(mktemp -d /tmp/serving_nomps.XXXX)
log "C: relaunch vLLM outside MPS ($empty) with MTP"
$PY scripts/brev/serving/relaunch_vllm.py --env "CUDA_MPS_PIPE_DIRECTORY=$empty/pipe" --env "CUDA_MPS_LOG_DIRECTORY=$empty/log" \
  --add --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --log .run/logs/vllm_serving_C.log --out "$R/relaunch.json" || { log "C: vLLM did not start"; tail -n 30 .run/logs/vllm_serving_C.log; exit 1; }

# not an MPS client: absent from every MPS server's client list, and nvidia-smi type C (not M+C)
engine=$(pgrep -f 'VLLM::EngineCore' | head -1)
clients=$(for s in $(echo get_server_list | nvidia-cuda-mps-control); do echo "get_client_list $s" | nvidia-cuda-mps-control; done | tr '\n' ' ')
smi=$(nvidia-smi | grep -E "^\|.*[0-9]+ +(M\+C|C|M\+C\+G|C\+G) " | sed 's/  */ /g' | tr '\n' ';')
in_mps=0; for c in $clients; do [[ $c == "$engine" ]] && in_mps=1; done
$PY - "$engine" "$clients" "$smi" "$in_mps" <<'PY' > "$R/mps_check.json"
import json, sys
engine, clients, smi, in_mps = sys.argv[1:]
row = next((r for r in smi.split(";") if f" {engine} " in r), "")
print(json.dumps({"engine_pid": engine, "mps_client_pids": clients.split(), "nvidia_smi_engine_row": row.strip(),
                  "engine_is_mps_client": in_mps == "1" or "M+C" in row, "nvidia_smi_rows": [r.strip() for r in smi.split(";") if r]}))
PY
cat "$R/mps_check.json"; echo
if grep -q '"engine_is_mps_client": true' "$R/mps_check.json"; then log "C: vLLM is still an MPS client; stopping"; exit 1; fi
for _ in $(seq 1 60); do all_ready && break; sleep 2; done

log "C: bench_latency.py --n 10"
$PY scripts/brev/bench_latency.py --n 10 --out $OUT/latency_isolated_mtp.json 2>&1 | grep -v "httpx INFO" | tail -n 25

log "C: NeMo Gym held-out, guarded"
$PY evaluations/nemo_gym/run_local.py --policies guarded --seed 2026 --distinct-from 7 --out serving/heldout_isolated_mtp.json 2>&1 \
  | grep -v "httpx INFO" | tail -n 3
curl -s -m 5 http://127.0.0.1:8001/metrics | grep -E "^vllm:spec_decode_num_(draft|accepted)_tokens_total" > "$R/spec_decode.txt"

log "C: fault test: kill -9 the EngineCore"
engine=$(pgrep -f 'VLLM::EngineCore' | head -1)
small=$(for n in audio vsr tts; do cat .run/pids/$n.pid; done | tr '\n' ' ')
t_kill=$(date +%s.%N)
kill -9 "$engine"
$PY - "$t_kill" "$engine" "$small" <<'PY' > "$R/fault.json"
import asyncio, json, subprocess, sys, time, httpx
t_kill, engine, small = float(sys.argv[1]), sys.argv[2], sys.argv[3].split()
out = {"killed_engine_pid": engine, "checks": {}}
def rec(name, ok, detail=""):
    out["checks"][name] = {"ok": bool(ok), "s_after_kill": round(time.time() - t_kill, 2), "detail": detail}
async def main():
    async with httpx.AsyncClient(timeout=5) as c:
        for name, port in (("voice", 8102), ("lips", 8101), ("tts", 8104)):
            try:
                h = (await c.get(f"http://127.0.0.1:{port}/health")).json()
                rec(f"{name}_health_ready", h.get("state") == "ready", h.get("state"))
            except Exception as exc:
                rec(f"{name}_health_ready", False, type(exc).__name__)
    r = subprocess.run([sys.executable, "scripts/brev/serving/smoke_box.py", "--checks", "accepted", "rules_question",
                        "--out", "/tmp/serving_C_fault_smoke.json"], capture_output=True, text=True, timeout=60)
    sm = json.load(open("/tmp/serving_C_fault_smoke.json"))["checks"]
    for k, v in sm.items():
        rec({"accepted": "accepted_voice_auto_spoken", "rules_question": "ambiguous_gets_rules_question"}[k], v["ok"], v["detail"])
    out["within_10_s"] = all(v["s_after_kill"] <= 10 for v in out["checks"].values())
    out["small_model_pids_alive"] = {p: subprocess.run(["kill", "-0", p]).returncode == 0 for p in small}
    out["gpu_processes"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                                          capture_output=True, text=True).stdout.strip().splitlines()
    out["pass"] = all(v["ok"] for v in out["checks"].values()) and out["within_10_s"] and all(out["small_model_pids_alive"].values())
asyncio.run(main())
print(json.dumps(out, indent=1))
PY
cat "$R/fault.json"
sleep 5
tail -n 15 .run/logs/vllm_serving_C.log > "$R/vllm_after_kill.txt"
for f in audio vsr tts; do tail -n 5 .run/logs/$f.log; done | grep -iE "error|illegal|cuda" > "$R/agents_after_kill.txt" || true
log "C: experiment steps finished; restoring"
