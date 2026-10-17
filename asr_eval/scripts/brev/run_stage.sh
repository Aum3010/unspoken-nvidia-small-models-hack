#!/usr/bin/env bash
# Instance side: run the evaluation in the background (survives the SSH session ending).
#   bash ~/run_stage.sh quick   # build manifest, then every phase on 10 clips per group
#   bash ~/run_stage.sh full    # every phase on all clips (manifest must exist)
#   bash ~/run_stage.sh phases decide,report
cd ~/asr_eval
# shellcheck disable=SC1091
source .env.brev
case "${1:-full}" in
  quick)  nohup bash -c "python run_eval.py --config configs/eval.yaml --phases manifest && python run_eval.py --config configs/eval.yaml --phases vad,models,decide,report --limit 10" > run.log 2>&1 < /dev/null & ;;
  full)   nohup python run_eval.py --config configs/eval.yaml --phases vad,models,decide,report > run.log 2>&1 < /dev/null & ;;
  phases) nohup python run_eval.py --config configs/eval.yaml --phases "$2" > run.log 2>&1 < /dev/null & ;;
esac
sleep 2
echo "launched ${1:-full}; follow with: bash ~/status.sh"
