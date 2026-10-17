#!/usr/bin/env bash
# Instance side: block while a process matching PATTERN runs (max MAX_MIN), then tail LOG.
# Lets a local `brev exec` call act as a single "notify me when done" step.
#   bash ~/wait_for.sh ~/asr_eval/run.log run_eval.py 90
LOG=$1; PAT=$2; MAX=${3:-60}
for _ in $(seq 1 $((MAX * 3))); do
  pgrep -f "$PAT" >/dev/null || break
  sleep 20
done
tail -n 40 "$LOG"
pgrep -af "$PAT" >/dev/null && echo "STILL RUNNING after ${MAX} min" || echo "process finished"
