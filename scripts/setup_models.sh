#!/usr/bin/env bash
# One-time setup on Apple Silicon: Python env, pinned upstream repos, checkpoints, datasets, web deps.
set -euo pipefail
cd "$(dirname "$0")/.."
CHAPLIN_COMMIT=7aee1f8fca776ce4f63690063310b53573b7d804

echo "== Python 3.12 env"
uv venv --python 3.12 .venv 2>/dev/null || true
uv pip install --python .venv -r requirements.txt

echo "== Chaplin (Auto-AVSR wrapper) @ $CHAPLIN_COMMIT"
mkdir -p third_party
if [[ ! -d third_party/chaplin ]]; then
  git clone -q https://github.com/amanvirparhar/chaplin.git third_party/chaplin
fi
git -C third_party/chaplin fetch -q --depth 1 origin "$CHAPLIN_COMMIT" 2>/dev/null || true
git -C third_party/chaplin checkout -q "$CHAPLIN_COMMIT" 2>/dev/null || true
# MPS support: upstream CTC scorer assumes CUDA or CPU
.venv/bin/python - <<'PY'
p = "third_party/chaplin/espnet/nets/ctc_prefix_score.py"
s = open(p).read()
old = '''        self.device = (
            torch.device("cuda:%d" % x.get_device())
            if x.is_cuda
            else torch.device("cpu")
        )'''
if old in s:
    open(p, "w").write(s.replace(old, "        self.device = x.device  # unspoken patch: support MPS (Apple Metal)"))
    print("patched CTC scorer for MPS")
PY
B=third_party/chaplin/benchmarks/LRS3
mkdir -p $B/models/LRS3_V_WER19.1 $B/language_models/lm_en_subword
for f in model.json model.pth; do
  [[ -s $B/models/LRS3_V_WER19.1/$f ]] || curl -sSL -o $B/models/LRS3_V_WER19.1/$f https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/$f
  [[ -s $B/language_models/lm_en_subword/$f ]] || curl -sSL -o $B/language_models/lm_en_subword/$f https://huggingface.co/Amanvir/lm_en_subword/resolve/main/$f
done

echo "== BNCI2014-008 ALS P300 (CC BY-NC-ND 4.0, local use only)"
mkdir -p data/bnci2014_008
for i in 1 2 3 4 5 6 7 8; do
  [[ -s data/bnci2014_008/A0$i.mat ]] || curl -sSL -o data/bnci2014_008/A0$i.mat http://bnci-horizon-2020.eu/database/data-sets/008-2014/A0$i.mat &
done; wait

if [[ "${WITH_GRID:-1}" == "1" ]]; then
  echo "== GRID corpus speakers 1-2 + alignments (CC BY 4.0, evaluation only)"
  mkdir -p data/grid
  for f in s1 s2 alignments; do
    [[ -d data/grid/$f ]] || { curl -sSL -o data/grid/$f.zip "https://zenodo.org/records/3625687/files/$f.zip?download=1" && unzip -q -o data/grid/$f.zip -d data/grid && rm data/grid/$f.zip; }
  done
  rm -rf data/grid/__MACOSX
fi

echo "== Web app + on-device browser models"
(cd web && npm install)
mkdir -p web/public/mediapipe
cp -r web/node_modules/@mediapipe/tasks-vision/wasm web/public/mediapipe/
[[ -s web/public/mediapipe/face_landmarker.task ]] || curl -sSL -o web/public/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# gaze: MobileGaze MobileOne-S0 eye-image model (MIT, yakhyo/gaze-estimation)
mkdir -p web/public/models
[[ -s web/public/models/mobileone_s0_gaze.onnx ]] || curl -sSL -o web/public/models/mobileone_s0_gaze.onnx \
  https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx

echo "== Coordinator model (Ollama, Metal)"
if command -v ollama >/dev/null; then ollama pull nemotron-3.5-lightning:30b-a3b; else echo "  install Ollama, or set UNSPOKEN_LLM_BACKEND=openai (see scripts/serve_coordinator_mlx.sh)"; fi

echo "== Pre-train EEG decoders on calibration runs"
PYTHONPATH=$PWD .venv/bin/python -c "from services.eeg import p300; [p300.train_and_evaluate(s) for s in p300.available_subjects()]"
echo "Done. Start everything with ./scripts/run_all.sh"
