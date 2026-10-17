#!/usr/bin/env bash
# Instance side: one-screen status of preparation / evaluation jobs, data sizes and the GPU.
tail -n 8 ~/asr_eval/prepare.log 2>/dev/null
echo "--- run.log"; tail -n 12 ~/asr_eval/run.log 2>/dev/null
echo "--- data"; du -sh /workspace/asr_eval_data/raw/* /workspace/asr_eval_data/tts /workspace/asr_eval_data/degraded 2>/dev/null
echo "--- jobs"; pgrep -af "fetch_data|make_tts|run_eval|build_manifest" | grep -v pgrep | head -5
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
