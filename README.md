# Unspoken

> **A voice for those who cannot speak.**
> Small specialist models across lips, weak voice, eyes and recorded EEG, coordinated by NVIDIA Nemotron 3.5 Lightning to *ask*, never to assume. Nothing is spoken until the person confirms it.

Unspoken is an assistive-communication **research prototype** that runs entirely on an Apple-Silicon Mac (Metal via MPS + MLX), with an optional vLLM coordinator on NVIDIA Brev. It is not a medical device, not a nurse-call system, and not a mind reader.

```
lips ─┐                                    ┌─ you select ─┐
voice ┼─► specialists propose ─► coordinator asks ─►      ├─► you confirm ─► policy gate ─► voice
eyes ─┤   raw n-best + quality    (Nemotron, typed tools) └─ or correct ─┘    (exact text only)
EEG* ─┘   *recorded replay only
```

**The core multi-agent behaviour is active uncertainty resolution.** When lip reading is unsure, or lips and voice disagree, the coordinator picks the smallest useful question and another modality (gaze dwell, blink scan, board) lets the person answer. The person, never a model, decides what is said.

## Quick start (macOS, Apple Silicon)

```bash
./scripts/setup_models.sh     # venv, pinned Chaplin + MPS patch, checkpoints, ALS EEG + GRID data, web deps, ollama pull
./scripts/run_all.sh          # gateway, 5 specialist services, web → http://localhost:5317
./scripts/smoke_test.sh       # health + board → confirm → speak round trip
./scripts/run_evals.sh        # every benchmark → evaluations/results/summary.json (shown on the Evidence page)
```

Requirements: Python 3.12 via `uv`, Node 20+, Ollama with `nemotron-3.5-lightning:30b-a3b`, and about 25 GB of free unified memory. Chrome is recommended (MediaRecorder webm, WebGL).

## What's in the app

| View | What it does |
|---|---|
| **Communicate** | Live camera with a MediaPipe mouth overlay and the "model view" crop; silent-mouth and weak-voice capture; candidate cards with source badges, raw scores and low-confidence words; coordinator note with its exact tool call; separate **Speak it** confirm dock; phrase board and two-step spelling board; live agent graph and timeline; conversation history with audit trail; a button that tries to speak an unconfirmed guess so you can watch the policy gate block it |
| **Eyes** | Hands-free 9-point WebGazer calibration with a measured error in pixels (gaze stays off if it is poor), plus deliberate-blink tuning |
| **Brain replay** | 8 ALS participants; the 6×6 matrix replays the recording's real flash sequence while row/column P300 evidence accumulates; scrolling EEG trace; ERP and accuracy-vs-repetitions charts; ground truth revealed only after inference; decoded letters can be routed into the communication panel, labelled `RECORDED EEG` |
| **Evidence** | Parameter ledger counted from the weight files, per-modality held-out results, NeMo Gym coordinator scores, safety tests, saved failures, sources and licences |

Every target works by click, keyboard, **hover dwell** (head-mouse), **gaze dwell**, or **blink scanning**. The confirmation method is recorded in the audit log. Light and dark themes, high contrast, and large text are available.

## Hackathon tracks

- **Track 2: Multi-Agent Systems.** These are heterogeneous agents: a visual-speech transformer, ASR plus an enhancer, a browser gaze/blink estimator, a Riemannian P300 decoder, an LLM coordinator, and a TTS renderer. They exchange typed `Evidence` and `ToolCall` messages (`shared/schemas.py`). Disagreement is preserved, and another modality resolves it.
- **NVIDIA Nemotron 3.5 Lightning** is the coordinator and tool caller: one model, shared by all logical roles. It runs through Ollama/MLX on Metal locally, or vLLM (NVFP4) on Brev.
- **NeMo Gym bounty:** `evaluations/nemo_gym/unspoken_communication/` is a resources server with decoupled `valid / action / safety` rewards across 10 task categories. These cover ambiguity, cross-modal conflict, capture failure, repeated failure, "none of these", uncalibrated gaze, prompt injection and pressure to speak. `run_local.py` scores policies without a Gym install.
- **Red Hat vLLM bounty:** `scripts/serve_vllm_brev.sh` serves Nemotron with `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3` and prefix caching. The coordinator speaks OpenAI-compatible tools (`UNSPOKEN_LLM_BACKEND=openai`). Only structured summaries leave the device.
- **Budget ≤ 33B:** 32.04B distinct deployed parameters, counted from disk (`scripts/verify_weights.py`).

## Results (held-out; see `docs/EVALUATION.md`)

| Area | Result |
|---|---|
| EEG replay, 8 ALS participants, held-out runs | flash AUC **0.850**, letter accuracy **94%** at 10 repetitions (chance 2.8%) |
| Lip reading, unseen GRID speakers | top-1 WER **79.6%** (best-of-5 71%). Readings are therefore candidates only. Still face flagged **8/8** |
| Weak speech | raw ≤8% WER down to −5 dB SNR; **enhancement hurt in heavy noise** (8→18%, 49→65%), so both transcripts are always shown. Noise-only declined 10/10 |
| Coordinator (NeMo Gym, 60 tasks) | Nemotron raw 0.933 / **0% unsafe**; deployed (Nemotron + validator + fallback) **1.000**, 80% of decisions by Nemotron |
| Safety | **25/25** gate tests and attacks pass; spoken = confirmed string **10/10**; unconfirmed text reaching TTS **0** |
| Model budget | **32.04B** of 33B, counted from weight files |

Run `./scripts/run_evals.sh` to reproduce. Details and known gaps are in `docs/EVALUATION.md`.
Lip reading (pipeline, personal enrollment, training, accuracy and latency) is documented in `docs/LIP_READING.md`.

## Repository

```
web/                    React + Vite UI (selection engine, gaze, face, capture, views)
services/gateway/       state machine, fusion, coordinator calls, policy-gated speech, WebSocket
services/vsr/           Auto-AVSR via Chaplin → raw n-best (MPS)
services/audio/         Whisper (MLX) raw ∥ RNNoise-enhanced, word diff, VAD gating
services/eeg/           BNCI2014-008 loader, xDAWN+Riemann P300 decoder, replay payloads
services/coordinator/   Nemotron tool calling, outside validation, deterministic fallback
services/tts/           Kokoro-82M (MLX), refuses without a gateway token
shared/                 schemas, confirmation gate, provenance + ports
evaluations/            safety tests, modality baselines, e2e exactness, NeMo Gym env, results
scripts/                setup, run, evals, weights ledger, vLLM/MLX serving, smoke test
docs/                   architecture, model cards & licences, datasets & consent, evaluation
```

## Product invariant

**No unverified interpretation is ever spoken as the patient's intention.** This is enforced by code, not by prompting. See `shared/confirmation.py` and `evaluations/safety_tests/`.
