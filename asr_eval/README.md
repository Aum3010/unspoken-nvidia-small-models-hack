# asr_eval — audio-only ASR evaluation for Unspoken

Which open ASR model can turn a weak or impaired utterance into **one of 20 request phrases**, and refuse when it can't? Each clip is one gaze-armed utterance window. The agent outputs exactly one of:

| Output | Meaning |
|---|---|
| `candidate` | a phrase id; the patient then confirms it with eye gaze |
| `unknown` | speech, but out of list, ambiguous or low confidence |
| `no_speech` | nothing to act on |

The worst failure is proposing a phrase on silence, noise or unrelated speech. The next worst is proposing the wrong phrase. Abstaining is acceptable. No LLM is involved.

## Run it

```bash
python run_eval.py --config configs/eval.yaml                 # everything
python run_eval.py --config configs/eval.yaml --phases decide,report   # re-tune/re-report without re-running models
python run_eval.py --config configs/eval.yaml --limit 10      # a few clips per group, end to end
```

On a Brev GPU box:

```bash
nohup bash scripts/brev_prepare.sh > prepare.log 2>&1 &   # venv under /workspace, requirements.lock, datasets, Kokoro clips
source .env.brev && python run_eval.py --config configs/eval.yaml
```

On a laptop (CPU) the same code runs with `configs/local_smoke.yaml`, which writes to `data_local/` and `results_local/`.

## Your own recordings (the numbers that matter)

```bash
pip install sounddevice soundfile numpy pyyaml
python scripts/record.py --list-devices
python scripts/record.py --speaker p01 --condition normal     # then soft, whisper, slow, muffled, noisy
```

- Clips are saved as `data/own/{speaker}_{condition}_{phraseid}_{take}.wav`, 16 kHz mono, raw mic signal. Negatives use `neg_{name}`.
- Use pseudonyms for speaker names.
- Copy `data/own/` to the GPU box and rerun from `--phases manifest`.
- Splits are by speaker. With a single speaker there are no human calib positives, so thresholds fall back to TTS positives and the summary says so. Record at least two speakers.

## What's measured

| Phase | Output |
|---|---|
| 0 | `results/env.md`, `requirements.lock` |
| 1 | `data/manifest.csv`: own, TORGO surrogate words (dysarthric/control), TORGO sentences, LibriSpeech, MUSAN / Speech Commands / ESC-50 non-speech, generated silence, Kokoro TTS (sanity only), degradation grid |
| 2 | Silero frame probabilities, speech seconds at VAD thresholds 0.2–0.6 |
| 3 | Models, each in its own process: transcript, forced phrase scores where supported, checkpoint revision, params, load time, peak GPU memory |
| 4 | Text ranking (CMU phonetic similarity) and forced-CTC ranking; abs/margin/negation gates; thresholds tuned on calib only (false-accept rate ≤ 1% per negative type) |
| 5–6 | `results/summary.md`, `results/plots/`, `results/failures.md`, `results/raw/{model}_{method}.csv`, `results/thresholds.json` |

Rules the harness enforces:
- TTS and degraded-synthetic rows are reported separately from real recordings.
- Every rate carries its counts.
- Zero-event rates show a Wilson interval and the rule-of-three bound.
- The best configuration is selected on calib and reported on test.
