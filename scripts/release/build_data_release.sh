#!/usr/bin/env bash
# Assemble the open data release (docs/DATA_RELEASE.md) into one folder for the Hugging Face dataset repo.
#   bash scripts/release/build_data_release.sh [out_dir] [brev_instance]
# Voice clips are fetched from the Brev instance (/workspace/asr_eval_data/own); everything else is local.
# Upload afterwards:  hf upload AumPandya01/unspoken-data <out_dir> . --repo-type dataset
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/.run/unspoken-data}"
BOX="${2:-marginal-sapphire-whale}"
cd "$ROOT"

rm -rf "$OUT"
mkdir -p "$OUT"/{coordinator_gym,neuro_comm_gym,neuro_comm_student,coordinator_student,lip_enrollment,voice_recordings}

cp scripts/release/DATASET_CARD.md "$OUT/README.md"
cp evaluations/nemo_gym/unspoken_communication/data/*.jsonl "$OUT/coordinator_gym/"
cp evaluations/nemo_gym/neuro_comm/data/*.jsonl "$OUT/neuro_comm_gym/"
cp data/student/train.jsonl "$OUT/neuro_comm_student/train_hindsight.jsonl"
cp data/student/train_value.jsonl "$OUT/neuro_comm_student/train_value.jsonl"
cp data/coord_student/s?_train.jsonl data/coord_student/s?_test.jsonl "$OUT/coordinator_student/"
cp -R data/enroll/. "$OUT/lip_enrollment/"
cp asr_eval/configs/eval.yaml "$OUT/voice_recordings/phrases_and_conditions.yaml"

# consent.csv is left out: its text predates the team's decision to publish (docs/DATA_RELEASE.md, consent)
scp -q "$BOX:/workspace/asr_eval_data/own/*.wav" "$BOX:/workspace/asr_eval_data/own/speakers.csv" "$OUT/voice_recordings/"

echo "wav: $(ls "$OUT"/voice_recordings/*.wav | wc -l) (expect 307)"
echo "npy: $(find "$OUT/lip_enrollment" -name '*.npy' | wc -l) (expect 90)"
du -sh "$OUT"
