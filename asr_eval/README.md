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

On a Brev GPU box (full guide: [../docs/BREV.md](../docs/BREV.md)):

```bash
bash asr_eval/scripts/brev/sync.sh <instance>                       # from the repo root, on your laptop
brev exec <instance> 'cd ~/asr_eval && nohup bash scripts/brev_prepare.sh > prepare.log 2>&1 < /dev/null &'
brev exec <instance> 'bash ~/run_stage.sh quick'                    # then: bash ~/run_stage.sh full
bash asr_eval/scripts/brev/fetch_results.sh <instance>
```

On a laptop (CPU) the same code runs with `configs/local_smoke.yaml`, which writes to `data_local/` and `results_local/`. Test there before spending GPU time.

## Code map

| Path | Role |
|---|---|
| `run_eval.py` | Orchestrator: `manifest → vad → models → decide → report`. Each model runs in its own subprocess. `--limit`, `--phases`, `--models` |
| `configs/eval.yaml` | Every setting: phrases, hard negatives, conditions, grids, models, decision rules. `local_smoke.yaml` overlays it for the laptop |
| `harness/config.py` | Loads the YAML (`extends:` overlays), resolves paths |
| `harness/audio.py` | Load/resample, the degradation effects, generated silence/mic noise |
| `harness/text.py` | Normalisation (contractions expanded), negation detection |
| `harness/vad.py` | Silero frame probabilities → speech seconds per threshold |
| `harness/models.py` | Model interface: `hf_ctc` (Parakeet-CTC, forced scoring), `nemo_transducer` (Parakeet-TDT, Nemotron streaming incl. 160/560 ms lookahead), `faster_whisper` |
| `harness/pipeline.py` | One clip through one model (transcript, phrase scores, timings), shared by `run_eval.py` and `serve.py` |
| `serve.py`, `static/index.html` | Live push-to-talk test bench and labelled-recording capture |
| `harness/decide.py` | Rankers (`text`, `forced_sum`, `forced_norm`, `forced_llr`), `decide()`, calib-only threshold search |
| `harness/metrics.py` | Positive/negative metrics, Wilson CI, rule of three, WER/CER, latency |
| `harness/report.py` | `summary.md`, plots, `failures.md`, literature context |
| `scripts/build_manifest.py` | Ingests every source, speaker split (TORGO stratified), degraded copies in a process pool |
| `scripts/fetch_data.py` | Dataset downloads with mirror fallback and licence evidence |
| `scripts/record.py`, `make_tts.py`, `degrade.py` | Own recordings, Kokoro sanity clips, degradation grid |
| `scripts/setup_brev.sh`, `brev_prepare.sh`, `env_report.py` | GPU-box environment (CUDA 12.8 torch, `requirements.lock`), preparation, `results/env.md` |
| `scripts/brev/` | Laptop ↔ instance helpers: sync, run, status, wait, fetch results, CLI install |

## Design decisions that affect the numbers

- **Closed set, not open transcription.** Every model's transcript is matched to the phrases phonetically (CMU dictionary, Levenshtein on phonemes; char ratio for out-of-dictionary words). CTC models also score each phrase directly: `log P(phrase | audio)` via `ctc_loss`, one encoder pass for everything.
- **Impossible is not perfect.** `ctc_loss(zero_infinity=True)` maps impossible alignments (a long phrase on a short clip) to loss 0, which reads as a perfect score. The harness keeps them impossible.
- **Silence favours short phrases.** In local tests, forced scoring on silence ranked "yes" first among phrases. That's why the VAD gate comes first, and why `forced_llr` compares each phrase with the model's own free transcript.
- **Negation is a hard gate.** "I'm not in pain" must never become `pain`. A transcript whose negation (not/no/never, after expanding contractions) disagrees with the best phrase becomes `unknown`.
- **Tuning sees calib only.** Objective: correct accepts ÷ positives, subject to a false-accept rate ≤ 1% for *each* negative type (stricter than pooled). Ties go to the stricter thresholds. If a task has no human calib speakers, positives fall back to TTS, and the summary says so.
- **The winner is chosen on calib** and reported on test, so selection never sees test labels.
- **Real, degraded and TTS results are always in separate tables.**

## Status

| Milestone | State |
|---|---|
| 1. Manifest + Silero + Parakeet-CTC end to end | Passes on CPU (local smoke); running on Brev |
| 2. Parakeet-TDT, Nemotron Speech Streaming (default, 160 ms, 560 ms lookahead), faster-whisper small / large-v2 / large-v3-turbo | Backends verified on Brev; full run in progress |
| Live test bench (`serve.py`) | Running on Brev, reachable through `brev port-forward` |
| Optional: sherpa-onnx keyword spotter | Not started |
| Optional: SAP dysarthric fine-tunes | Skipped (SAP data use agreement) |

## Live test bench

`serve.py` is a push-to-talk page that runs the same pipeline and frozen thresholds as the evaluation. It shows the decision, the gate that made it, the top 3 phrases, the transcript and timings. It can also save labelled recordings straight into `data/own/`. On Brev: `bash ~/serve_ui.sh <model> <method>`, then `brev port-forward <instance> --port 8800:8800` and open http://localhost:8800 ([../docs/BREV.md](../docs/BREV.md)). On a laptop: `python serve.py --config configs/local_smoke.yaml`.

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
