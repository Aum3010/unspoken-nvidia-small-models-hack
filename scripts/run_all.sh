#!/usr/bin/env bash
# Start every Unspoken service locally (Apple Silicon / Metal). Ctrl-C stops all.
#   --lips    only the gateway and the lip agent (voice, EEG, TTS and the coordinator model stay off;
#             the gateway's deterministic policy plans). Open the web app at /?lips for the lips-only room.
#   --gaze    only the gateway (gaze runs in the browser). Open the web app at /?gaze for the gaze-only room.
#   --no-web  do not start the web dev server
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p .run/logs
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

pids=()
start() {
  local name=$1 port=$2
  "$PY" -m uvicorn "services.$name.app:app" --host 127.0.0.1 --port "$port" --log-level warning \
    >".run/logs/$name.log" 2>&1 &
  pids+=($!)
  echo "  • $name  → http://127.0.0.1:$port  (log: .run/logs/$name.log)"
}

LIPS=0 GAZE=0 WEB=1
for a in "$@"; do
  case $a in --lips) LIPS=1 ;; --gaze) GAZE=1 ;; --no-web) WEB=0 ;; esac
done

if ((GAZE)); then
  echo "Starting gaze only (the gateway; eye tracking runs in the browser)"
  start gateway 8000
elif ((LIPS)); then
  echo "Starting lip reading only"
  start gateway 8000
  start vsr 8101
else
  if [[ "${UNSPOKEN_LLM_BACKEND:-ollama}" == "ollama" ]] && ! curl -s localhost:11434/api/tags >/dev/null; then
    echo "  ! Ollama is not running — coordinator will use its deterministic fallback (start: ollama serve)"
  fi
  echo "Starting Unspoken services"
  start gateway 8000
  start coordinator 8105
  start eeg 8103
  start tts 8104
  start audio 8102
  start vsr 8101
fi

if ((WEB)); then
  (cd web && npm run dev >../.run/logs/web.log 2>&1) &
  pids+=($!)
  echo "  • web      → http://localhost:8800$( ((GAZE)) && echo '/?gaze' || { ((LIPS)) && echo '/?lips'; })"
fi

trap 'echo; echo "Stopping…"; kill "${pids[@]}" 2>/dev/null || true' INT TERM EXIT
wait
