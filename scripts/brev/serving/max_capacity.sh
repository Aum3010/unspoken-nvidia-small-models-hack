#!/usr/bin/env bash
# Experiment D (intrusive, ~15-20 min): how many concurrent coordinator requests vLLM sustains on this GPU.
#   mode 1 SHARED     every agent loaded; only vLLM restarted with --max-num-seqs 64 (retry 32, 16), util 0.55
#   mode 2 DEDICATED  voice, lip, TTS agents and gateway stopped; vLLM --max-num-seqs 128 --gpu-memory-utilization 0.85
#                     (retry 64 / 0.80)
# Ramp per mode: 1 2 4 (below the ramp, for the p95 < 500 ms point) then 8 16 32 64 128, skipping levels above
# 2 x max-num-seqs; 3 warm-up + max(20, 2 x level) measured, the same requests as experiment A (bench_concurrency.py).
# A watchdog samples every 2 s (nvidia-smi -> serving/gpu_trace.csv; agents' /health in shared mode; the vLLM log).
# Safety stops (end everything, restore): CUDA error / illegal memory / OOM / Traceback in the vLLM log, GPU memory
#   > 92 GiB, temperature > 87 C, (shared) an agent's /health not ready twice in a row.
# Performance limits (end this mode's ramp, go on to the next mode): > 2% errors/timeouts at a level, p95 > 5 s.
# Restore on EXIT/INT/TERM (ERR is logged): start_app.sh stop && start_app.sh, all ready, safety tests.
#   nohup bash scripts/brev/serving/max_capacity.sh > .run/logs/serving_D.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.."
source scripts/brev/serving/common.sh
R=$OUT/max_capacity_parts
STOP=/tmp/serving_D_stop MODEF=/tmp/serving_D_mode LOGSTART=/tmp/serving_D_logstart VLOG=.run/logs/vllm_serving_D.log
TRACE=$OUT/gpu_trace.csv
MEM_MAX_MIB=$((92 * 1024)) TEMP_MAX=87
mkdir -p "$R"; rm -f "$R"/*.json "$STOP" "$MODEF"

if ! pgrep -f '[i]solation_experiment.sh' >/dev/null && wait_idle D; then :; else
  reason=${IDLE_SKIP_REASON:-experiment C still running}
  log "D: SKIPPED: $reason"
  echo "{\"skipped\": true, \"reason\": \"$reason\"}" > $OUT/max_capacity.json
  exit 0
fi

WD=
restored=0
finish() {
  ((restored)) && return
  restored=1
  trap - ERR
  echo restore > "$MODEF"
  [[ -n $WD ]] && kill "$WD" 2>/dev/null
  restore D; local rc=$?
  self_window_end
  $PY scripts/brev/serving/max_capacity_report.py
  log "D: done (restore rc=$rc)"
  exit $rc
}
trap finish EXIT
trap 'exit 130' INT TERM
trap 'log "D: command failed (line $LINENO, rc=$?)"' ERR
self_window_start

watchdog() {
  local fails=0 q mem util temp pow clk mode bad start
  echo "time,mode,mem_used_mib,util_pct,temp_c,power_w,sm_clock_mhz,agents_not_ready,stop" > "$TRACE"
  while :; do
    mode=$(cat "$MODEF" 2>/dev/null)
    q=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits | tr -d ' ')
    IFS=, read -r mem util temp pow clk <<< "$q"
    bad=""
    if [[ $mode == shared ]]; then
      for p in 8101 8102 8104; do
        curl -s -m 3 "http://127.0.0.1:$p/health" | grep -q '"state":"ready"' || bad+="$p "
      done
      [[ -n $bad ]] && fails=$((fails + 1)) || fails=0
    fi
    if [[ ! -e $STOP && ( $mode == shared || $mode == dedicated ) ]]; then
      start=$(cat "$LOGSTART" 2>/dev/null || echo 1)
      hit=$(tail -n +"$start" "$VLOG" 2>/dev/null | grep -m1 -E 'CUDA error|illegal memory|\bOOM\b|out of memory|OutOfMemory|Traceback')
      if [[ -n $hit ]]; then echo "safety: vLLM log: ${hit:0:200}" > "$STOP"
      elif ((${mem%.*} > MEM_MAX_MIB)); then echo "safety: GPU memory ${mem} MiB > ${MEM_MAX_MIB} MiB" > "$STOP"
      elif ((${temp%.*} > TEMP_MAX)); then echo "safety: GPU temperature ${temp} C > ${TEMP_MAX} C" > "$STOP"
      elif ((fails >= 2)); then echo "safety: agent /health not ready: $bad" > "$STOP"
      fi
    fi
    echo "$(date +%s.%N | cut -c1-14),$mode,$mem,$util,$temp,$pow,$clk,${bad% },$([[ -e $STOP ]] && echo 1 || echo 0)" >> "$TRACE"
    sleep 2
  done
}
echo relaunch > "$MODEF"
watchdog & WD=$!

# ramp MODE MAX_NUM_SEQS: levels 1 2 4 8 16 32 64 128 up to 2 x max-num-seqs; returns 2 on a safety stop
ramp() {
  local mode=$1 mns=$2 levels=()
  for l in 1 2 4 8 16 32 64 128; do ((l <= 2 * mns)) && levels+=("$l"); done
  echo $(( $(wc -l < "$VLOG") + 1 )) > "$LOGSTART"   # only what vLLM logs from here on
  echo "$mode" > "$MODEF"
  log "D: $mode ramp, max-num-seqs $mns: levels ${levels[*]}"
  $PY scripts/brev/serving/bench_concurrency.py --levels "${levels[@]}" --measured auto --reference --stop-rules \
    --stop-file "$STOP" --no-plot --label "$mode, max-num-seqs $mns" --out "$R/ramp_$mode" 2>&1 | grep -v "httpx INFO" | grep -E "level|STOP|reference|p50"
  echo relaunch > "$MODEF"
  [[ -e $STOP ]] && { log "D: $(cat "$STOP")"; cp "$STOP" "$R/stop_$mode.txt"; return 2; }
  return 0
}

relaunch() {  # relaunch MODE MNS [UTIL]: restart only vLLM with these settings
  local mode=$1 mns=$2 util=${3:-}
  local args=(--set --max-num-seqs "$mns")
  [[ -n $util ]] && args+=(--set --gpu-memory-utilization "$util")
  log "D: $mode: relaunch vLLM ${args[*]}"
  $PY scripts/brev/serving/relaunch_vllm.py "${args[@]}" --log "$VLOG" --out "$R/relaunch_$mode.json"
}

# ---- mode 1: shared ----
mns=
for m in 64 32 16; do relaunch shared "$m" && { mns=$m; break; }; tail -n 20 "$VLOG"; done
[[ -z $mns ]] && { log "D: shared: vLLM would not start with 64, 32 or 16"; exit 1; }
for _ in $(seq 1 60); do all_ready && break; sleep 2; done
ramp shared "$mns"; rc=$?
((rc == 2)) && { log "D: safety stop in shared mode; skipping dedicated mode"; exit 0; }

# ---- mode 2: dedicated ----
log "D: dedicated: stopping voice, lip, TTS agents and the gateway"
for n in audio vsr tts gateway; do [[ -e .run/pids/$n.pid ]] && kill "$(cat .run/pids/$n.pid)" 2>/dev/null; done
for _ in $(seq 1 30); do
  alive=0; for n in audio vsr tts gateway; do kill -0 "$(cat .run/pids/$n.pid 2>/dev/null)" 2>/dev/null && alive=1; done
  ((alive)) || break; sleep 1
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$R/gpu_apps_dedicated.txt"
mns=
relaunch dedicated 128 0.85 && mns=128 || { tail -n 20 "$VLOG"; relaunch dedicated 64 0.80 && mns=64; }
[[ -z $mns ]] && { log "D: dedicated: vLLM would not start with 128/0.85 or 64/0.80"; exit 1; }
ramp dedicated "$mns"
log "D: ramps finished; restoring"
