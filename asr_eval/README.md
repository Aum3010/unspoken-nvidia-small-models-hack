# asr_eval — audio-only ASR evaluation for Unspoken

Which open ASR model can turn a weak or impaired utterance into **one of 20 request phrases**, and refuse when it can't? Each clip is one gaze-armed utterance window. The agent outputs exactly one of:

| Output | Meaning |
|---|---|
| `candidate` | a phrase id; the patient then confirms it with eye gaze |
| `unknown` | speech, but out of list, ambiguous or low confidence |
| `no_speech` | nothing to act on |

The worst failure is proposing a phrase on silence, noise or unrelated speech. The next worst is proposing the wrong phrase. Abstaining is acceptable. No LLM is involved.

## Result (2026-10-17)

**Voice agent model: Nemotron Speech Streaming en 0.6B (`nvidia/nemotron-speech-streaming-en-0.6b`), default lookahead, with forced RNNT phrase scoring (`forced_llr`).**

Measured leave-one-speaker-out on 3 recorded speakers: 235 real phrase clips (normal, whisper, noisy, muffled) and 71 real negatives. Each person was tested by thresholds tuned on the other two. Strict operating point (`results/loso.md`):

| | Nemotron en, `forced_llr` |
|---|---|
| Right phrase accepted | 168/235 (71.5%) |
| Wrong phrase accepted | 1/235 (0.4%) |
| Resolved within one gaze question (right candidate, or right phrase in the coordinator's top 3) | 212/235 (90.2%) |
| False accepts: near-miss sentences / silence-noise-cough / unrelated speech | 0/45 · 1/19 · 0/7 |
| TORGO command words accepted (control / dysarthric) | 75.4% / 61.1% |
| Latency, VAD + ASR + scoring on the GPU | 43 ms p50, 94 ms p95 |

Why this model:
- **Versus transcript matching on the same model:** phrase scoring added 4 points of accepted phrases, 7 points of one-question resolution, and 21 points on dysarthric TORGO words.
- **Versus Parakeet-CTC with phrase scoring:** Parakeet-CTC accepts a little more, but with 3–5× more wrong and false accepts.
- **Versus the rest:** Whisper turbo/small and Parakeet-TDT score lower on one-question resolution.
- **The multilingual Nemotron 3.5 was less accurate on these English speakers** (43.8% accepted, 78.3% one-question) and slower (p95 263 ms), so it was not adopted.

**Remaining losses are noisy, muffled and whispered speech.** With three speakers and small negative sets, gaps of 2–3 points are within noise.

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
| `harness/pipeline.py` | One clip through one model (transcript, phrase scores, timings), shared by `run_eval.py` and `harness/live.py` |
| `harness/live.py` | `Bench`: one live clip through the evaluated pipeline (Silero gate, model, rankers, `decide()`, frozen thresholds). Used by `serve.py` **and by the app's voice agent** (`services/audio/`) |
| `serve.py`, `static/index.html` | Live push-to-talk test bench and labelled-recording capture |
| `harness/decide.py` | Rankers (`text`, `forced_sum`, `forced_norm`, `forced_llr`), `decide()` and `explain()`, calib-only threshold search, language-aware surface forms |
| `harness/metrics.py` | Positive/negative metrics, Wilson CI, rule of three, WER/CER, latency |
| `harness/report.py` | `summary.md`, plots, `failures.md`, literature context |
| `scripts/build_manifest.py` | Ingests every source, speaker split (TORGO stratified), degraded copies in a process pool |
| `scripts/fetch_data.py` | Dataset downloads with mirror fallback and licence evidence |
| `scripts/record.py`, `make_tts.py`, `degrade.py` | Own recordings (CLI), Kokoro sanity clips, degradation grid |
| `scripts/loso.py` | Leave-one-speaker-out over own recordings: real vs degraded tables, both operating points, the one-gaze-question view |
| `scripts/multilingual_sanity.py` | Kokoro speech in es/fr/hi/zh through the multilingual model (feasibility only) |
| `scripts/setup_brev.sh`, `brev_prepare.sh`, `env_report.py` | GPU-box environment (CUDA 12.8 torch, `requirements.lock`), preparation, `results/env.md` |
| `scripts/brev/` | Laptop ↔ instance helpers: sync, run, status, wait, fetch results, CLI install |

## Design decisions that affect the numbers

- **Closed set, not open transcription.** Every model's transcript is matched to the phrases phonetically (CMU dictionary, Levenshtein on phonemes; char ratio for out-of-dictionary words). Models that support it also score each phrase directly, `log P(phrase | audio)`, from the same encoder pass:
  - Parakeet-CTC: `ctc_loss`;
  - Nemotron: its own RNNT loss, using NeMo's pure-PyTorch implementation because the Numba kernels need libNVVM.
  - Parakeet-TDT stays on transcript matching: its pure-PyTorch TDT loss takes seconds per clip.
- **Impossible is not perfect.** `ctc_loss(zero_infinity=True)` maps impossible alignments (a long phrase on a short clip) to loss 0, which reads as a perfect score. The harness keeps them impossible.
- **Silence favours short phrases.** In local tests, forced scoring on silence ranked "yes" first among phrases. That's why the VAD gate comes first, and why `forced_llr` compares each phrase with the model's own free transcript.
- **Negation is a hard gate.** "I'm not in pain" must never become `pain`. A transcript whose negation (not/no/never, after expanding contractions) disagrees with the best phrase becomes `unknown`.
- **No words heard, no phrase.** If the model's own transcript is empty, forced scoring may not create a phrase; the result is `unknown` (`plan.md`: never infer words absent from all usable input). This removed a "yes" accepted on a breathing clip.
- **Tuning sees calib only.** Objective: correct accepts ÷ positives, subject to a false-accept rate ≤ 1% (strict, headline) or ≤ 5% (balanced, live UI default) for *each* negative type.
  - Ties go to the most permissive thresholds inside that limit. Tie-breaking toward strict made the agent abstain on real voices when tuning data was clean.
  - If a task has no human calib speakers, positives fall back to TTS, and the summary says so.
- **Every recorded speaker is tested once.** `scripts/loso.py` tunes on the others and tests the held-out person.
- **The product view is reported too.** In the full app an `unknown` makes the coordinator offer the top 3 for a gaze choice, so "resolved within one gaze question" is the metric that matches the system (README: smallest useful question).
- **The winner is chosen on calib** and reported on test, so selection never sees test labels.
- **Real, degraded and TTS results are always in separate tables.**

## Status

| Milestone | State |
|---|---|
| 1. Manifest + Silero + Parakeet-CTC end to end | Done |
| 2. Parakeet-TDT, Nemotron Speech Streaming (default, 160 ms, 560 ms), faster-whisper small / large-v2 / large-v3-turbo | Done. large-v2 then dropped (over budget, slowest) |
| 3 recorded speakers, leave-one-speaker-out | Done: 307 clips (p01 154, p02 102, p03 51) |
| Forced phrase scoring for Nemotron (RNNT) | Done; adopted |
| Multilingual Nemotron 3.5 (en-US prompt) vs English-only | Done; not adopted (less accurate, slower). Language selector and draft translations kept |
| Live test bench (`serve.py`) | Running on Brev with Nemotron en / `forced_llr` / balanced |
| In the app | Done (2026-10-18): `services/audio/` runs Nemotron en / `forced_llr` / balanced through `harness/live.py` |
| Next | More noisy/muffled/whisper recordings and negatives from p02/p03 and new speakers; native-speaker review and recordings before any non-English use |
| Optional: sherpa-onnx keyword spotter | Not started |
| Optional: SAP dysarthric fine-tunes | Skipped (SAP data use agreement) |

## Live test bench

`serve.py` is a push-to-talk page that runs the same pipeline and frozen thresholds as the evaluation.

- **Record tab (default):**
  - enter a pseudonym and a condition;
  - prompts appear in random order (the 20 phrases, optionally the negatives);
  - each clip is saved under its prompt as soon as it's spoken;
  - **R** redoes, **S** skips, and a session resumes from what's on disk;
  - non-speech prompts read as actions ("breathe, don't read it aloud").
- **Test tab:** the decision, the gate that made it, the top 3 phrases, the transcript, the VAD/Whisper signals and timings.
- **Header selectors:**
  - model and method;
  - sensitivity: strict (≤1% false accepts) or balanced (≤5%, default);
  - *Patient speaks*: English default, or Español / Français / Deutsch / 中文 / हिन्दी for prompt-conditioned multilingual models only. The answer is always the English phrase; the phrase id is the shared key, with no runtime translation. Translations are drafts and thresholds are English-tuned, and the UI says so.
- **How to run it:**
  - On Brev: `bash ~/serve_ui.sh nemotron-streaming-0.6b forced_llr balanced`, then `brev port-forward <instance> --port 8800:8800`, and open http://localhost:8800 ([../docs/BREV.md](../docs/BREV.md)).
  - On a laptop: `python serve.py --config configs/local_smoke.yaml`.

## Your own recordings (the numbers that matter)

```bash
pip install sounddevice soundfile numpy pyyaml
python scripts/record.py --list-devices
python scripts/record.py --speaker p01 --condition normal     # then soft, whisper, slow, muffled, noisy
```

- **The live bench's Record tab is the easiest way to record.** The CLI recorder above does the same from a terminal.
- **File names:** `data/own/{speaker}_{condition}_{phraseid}_{take}.wav`, 16 kHz mono, raw mic signal. Negatives use `neg_{name}`. Use pseudonyms.
- **Where they live:**
  - on the GPU box in `/workspace/asr_eval_data/own/` (`~/asr_eval/data/own`);
  - a laptop backup in `asr_eval/data/own/`, gitignored and never committed.
- **Copy them home:** `brev copy <instance>:/workspace/asr_eval_data/own/ asr_eval/data/own/` (WSL path on Windows).
- **After new recordings:** rerun from `--phases manifest`, then `scripts/loso.py`.
- **Splits are by speaker.** `manifest.calib_speakers` pins p02 as calib for the standard split, and LOSO rotates everyone.
- **Clips under 0.2 s** (button released early) are skipped and logged.

## What's measured

| Phase | Output |
|---|---|
| 0 | `results/env.md`, `requirements.lock` |
| 1 | `data/manifest.csv`: own, TORGO surrogate words (dysarthric/control), TORGO sentences, LibriSpeech, MUSAN / Speech Commands / ESC-50 non-speech, generated silence, Kokoro TTS (sanity only), degradation grid |
| 2 | Silero frame probabilities, speech seconds at VAD thresholds 0.2–0.6 |
| 3 | Models, each in its own process: transcript, forced phrase scores where supported, checkpoint revision, params, load time, peak GPU memory |
| 4 | Text ranking (CMU phonetic similarity) and forced phrase scoring (CTC/RNNT); abs/margin/negation/empty-transcript gates; thresholds tuned on calib only, strict (≤ 1%) and balanced (≤ 5%) false-accept points per negative type |
| LOSO | `results/loso.md`: every recorded speaker tested by thresholds tuned on the others; real recordings separate from degraded copies |
| 5–6 | `results/summary.md`, `results/plots/`, `results/failures.md`, `results/raw/{model}_{method}.csv`, `results/thresholds.json` |

Rules the harness enforces:
- TTS and degraded-synthetic rows are reported separately from real recordings.
- Every rate carries its counts.
- Zero-event rates show a Wilson interval and the rule-of-three bound.
- The best configuration is selected on calib and reported on test.
