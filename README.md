# Unspoken

> **A voice for those who cannot speak.**
> Small specialist models across lips, weak voice, eyes and recorded EEG, coordinated by NVIDIA Nemotron 3.5 Lightning to *ask*, never to assume. Nothing is spoken until the person confirms it.

Unspoken is an assistive-communication **research prototype**. The web UI runs in the browser; the agents run either on an Apple-Silicon Mac (Metal via MPS + MLX) or on an NVIDIA GPU on Brev (CUDA, with the coordinator on vLLM), reached through an SSH tunnel. It is not a medical device, not a nurse-call system, and not a mind reader.

```
lips ─┐                                    ┌─ you select ─┐
voice ┼─► specialists propose ─► coordinator asks ─►      ├─► you confirm ─► policy gate ─► voice
eyes ─┤   n-best / phrases + quality (Nemotron, typed tools) └─ or correct ─┘ (exact text only)
EEG* ─┘   *recorded replay only
```

**The core multi-agent behaviour is active uncertainty resolution.** When lip reading is unsure, or lips and voice disagree, the coordinator picks the smallest useful question and another modality (gaze dwell, blink, nod, the phrase board) lets the person answer. The person, never a model, decides what is said.

## Quick start (macOS, Apple Silicon)

```bash
./scripts/setup_models.sh     # venv, pinned Chaplin + MPS patch, checkpoints, ALS EEG + GRID data, web deps, ollama pull
./scripts/run_all.sh          # gateway, 5 specialist services, web → http://localhost:8800
./scripts/smoke_test.sh       # health + board → confirm → speak round trip
./scripts/run_evals.sh        # every benchmark → evaluations/results/summary.json (also served at /api/evals)
```

Requirements: Python 3.12 via `uv`, Node 20+, Ollama with `nemotron-3.5-lightning:30b-a3b`, and about 25 GB of free unified memory. Chrome is recommended (MediaRecorder webm, WebGPU/WebGL). The voice agent (Nemotron Speech) needs NeMo: on a Mac, install `asr_eval/requirements.in` into the backend venv.

## Quick start (backend on an NVIDIA Brev GPU, UI on your laptop)

```bash
bash scripts/brev/sync_app.sh <instance>                                   # ship the committed backend to ~/unspoken
brev exec <instance> 'cd ~/unspoken && nohup bash scripts/brev/setup_app.sh > .run/setup.log 2>&1 < /dev/null &'   # once
brev exec <instance> 'bash ~/unspoken/scripts/brev/start_app.sh'           # vLLM + 6 services on 127.0.0.1 (status | stop)
bash scripts/brev/connect.sh <instance>                                    # tunnel the gateway to localhost:8000
cd web && npm install && npm run dev                                       # UI → http://localhost:8800
```

Step by step, including the CUDA port and the problems already solved: [docs/BREV.md](docs/BREV.md) §4b. Latency with every model running, and how to serve it faster: [docs/SERVING.md](docs/SERVING.md).

## What's in the app

One liquid-glass page (`web/src/glass/`) at **http://localhost:8800**, over the live camera:

| Part | What it does |
|---|---|
| **Eye calibration** | Its own plain black screen: 11 hands-free dots placed on the real targets (Yes/No, rail tiles, phrase cards), then the error measured on 3 points it was not fitted on. A good calibration is saved and reused automatically on the same window size; each gaze selection refines it. Gaze is used only when it is good enough; otherwise the pointer stands in for the eyes |
| **Every input at once** | No mode switching. Voice: always-on VAD cuts 16 kHz clips for the voice agent. Lips: mouth-movement detection cuts webm clips for the lip agent. Gaze: MediaPipe iris + MobileGaze eye-image model, calibrated per person, with zone targeting for the rail, the Yes/No band and the cards. Blink (deliberate, MediaPipe) selects what you look at; head nod = yes, shake = no. Capture pauses while a message waits for confirmation or speech is playing |
| **Writing in light** | A reading appears as faint, trembling pencil, and is drawn in by the pen once spoken. In confirm mode that is after Yes, a blink or a nod; in **demo auto-speak mode** (`UNSPOKEN_AUTO_SPEAK=1`, the current default) the coordinator's top reading is spoken without a confirmation, audited as `demo_auto_spoken` by the system |
| **Phrase cards** | Six phrase cards bloom from Phrases; choosing one (dwell, blink or click) speaks it at once, through the policy gate |
| **Rail and answers** | Phrases · Clear · Help on the rail, and two big **Yes / No** targets at the top, far apart for webcam gaze |
| **Agent activity** (top right) | The coordinator nucleus fed by six inputs (voice, lips, gaze, blink, brain replay, head); each pulses when its agent reports, with the latest gateway timeline below |
| **History** | Every spoken message with the inputs and confirmation method behind it; click to hear it again |

Every target works by click, keyboard, **gaze or pointer dwell** (confirm targets need 1.5× longer), or **deliberate blink** (Space simulates it). The confirmation method is recorded in the audit log.

## Hackathon tracks

- **Track 2: Multi-Agent Systems.** These are heterogeneous agents: a visual-speech transformer, a streaming ASR that scores fixed phrases, a browser gaze/blink/head estimator, a Riemannian P300 decoder, an LLM coordinator, and a TTS renderer. They exchange typed `Evidence` and `ToolCall` messages (`shared/schemas.py`). Disagreement is preserved, and another modality resolves it.
- **Bounty write-ups** (Nemotron Lightning, NeMo Gym, Red Hat vLLM), every number with its result file: [docs/BOUNTIES.md](docs/BOUNTIES.md).
- **NVIDIA Nemotron 3.5 Lightning** is the coordinator and tool caller: one model, shared by all logical roles. It runs through Ollama/MLX on Metal locally, or vLLM (NVFP4) on Brev.
- **NeMo Gym bounty:** `evaluations/nemo_gym/unspoken_communication/` is a resources server with decoupled `valid / action / safety` rewards across 12 task categories. These cover ambiguity, cross-modal conflict, capture failure, repeated failure, "none of these", uncalibrated gaze, prompt injection, pressure to speak, misread words and negations that must survive a correction. `run_local.py` scores policies without a Gym install.
- **Red Hat vLLM bounty:** `scripts/serve_vllm_brev.sh` serves Nemotron with `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3` and prefix caching. The coordinator speaks OpenAI-compatible tools (`UNSPOKEN_LLM_BACKEND=openai`). With only the coordinator on Brev, just structured summaries leave the device; with the whole backend there (`scripts/brev/start_app.sh`), mouth and voice clips go too (the UI on `ui` does not currently say so; see `docs/TECHNICAL_VERBOSE.md` §10).
- **Budget ≤ 33B:** 31.01B distinct deployed parameters with the coordinator counted at its nominal 30B (32.59B with its weight-file count, still within budget; `scripts/verify_weights.py`).

## Results (held-out; see `docs/EVALUATION.md`)

| Area | Result |
|---|---|
| EEG replay, 8 ALS participants, held-out runs | flash AUC **0.850**, letter accuracy **94%** at 10 repetitions (chance 2.8%) |
| Lip reading, unseen GRID speakers | top-1 WER **79.6%** (best-of-5 71%). Readings are therefore candidates only. Still face flagged **8/8** |
| Weak speech | Nemotron Speech 0.6B + forced phrase scoring, 3 speakers, leave-one-speaker-out: right phrase accepted **71.5%**, wrong **0.4%**, **90.2%** resolved within one gaze question, 0/45 near-miss false accepts, 43 ms per clip (`asr_eval/results/loso.md`) |
| Coordinator (NeMo Gym, 72 tasks, Brev, current prompt) | Nemotron raw 0.917, deployed (Nemotron + validator + fallback) 0.940, **0% unsafe**; every wrong move is `ignore_input` on real input (e.g. a person who failed twice) — see `docs/TECHNICAL_VERBOSE.md` §3.3. (Mac, 60 tasks, earlier prompt: 0.933 / 1.000) |
| Safety | **49/49** unit tests (gate, state machine, auto-speak, phrase scoring, optional validator rules, check-in preview) and **7/7** live attacks blocked; TTS text = confirmed string **10/10** (re-transcription matched 10/10 with the earlier Whisper voice agent; today's phrase-only voice agent cannot re-transcribe free text, `docs/EVALUATION.md` §5); unconfirmed text reaching TTS **0** |
| Model budget | **31.01B** of 33B (coordinator at its nominal 30B; 32.59B with its weight-file count) |
| Latency, every model on one RTX PRO 6000 (NVIDIA MPS) | accepted voice message clip → spoken **71–73 ms** p50 on the GPU box (rules decide accepted readings, no model call); lip reading **0.41 s** after the clip ends from the laptop (streamed while mouthing; 0.90 s uploaded whole); voice agent 75 ms with every model busy (244 ms without MPS); voice result 64 ms after the end-of-utterance pause from the laptop, thanks to early decode (`docs/SERVING.md`) |

Run `./scripts/run_evals.sh` to reproduce. Details and known gaps are in `docs/EVALUATION.md`.

## Choosing the voice model on Brev (done: Nemotron Speech is the voice agent)

`asr_eval/` is the audio-only benchmark that picked the voice agent's ASR. Candidates:
- Parakeet-CTC/TDT
- Nemotron Speech Streaming
- Whisper (several sizes)

It runs on an NVIDIA Brev GPU with no other Unspoken model loaded. The task is a **20-phrase closed set with abstention**: output a phrase, `unknown`, or `no_speech`. The data covers your own recordings, TORGO dysarthric speech and real noise negatives.

```bash
cd asr_eval && python run_eval.py --config configs/eval.yaml          # on the GPU box
python run_eval.py --config configs/local_smoke.yaml --limit 6         # CPU plumbing check on a laptop
```

**Outcome (2026-10-17):**
- **Model:** Nemotron Speech Streaming en 0.6B with forced RNNT phrase scoring, now the app's voice agent (`services/audio/` imports the same decision code and frozen thresholds, so it decides exactly as evaluated).
- **Tested:** on 3 recorded speakers (leave-one-speaker-out).
- **Results:** it accepted the right phrase 71.5% of the time with 0.4% wrong, resolved 90.2% within one gaze question, and had 0/45 near-miss false accepts, at 43 ms per clip.
- **Not adopted:** the multilingual Nemotron 3.5 was less accurate and slower on English. Its language selector and draft translations are kept.
- Details: [asr_eval/README.md](asr_eval/README.md), [docs/EVALUATION.md](docs/EVALUATION.md) §6.

- Setup, the CLI on Windows (WSL), costs and the sync/run/fetch loop: [docs/BREV.md](docs/BREV.md).
- Models and licences: [docs/MODEL_CARDS_AND_LICENSES.md](docs/MODEL_CARDS_AND_LICENSES.md#asr-candidates-evaluated-asr_eval).

## Repository

```
web/                    React + Vite UI: glass room, ink + agent-activity engine, always-on sensors, calibration
services/gateway/       state machine, fusion, coordinator calls, policy-gated speech, WebSocket
services/vsr/           Auto-AVSR via Chaplin: personal phrase enrollment (DTW over encoder features), optional free text
services/audio/         Nemotron Speech 0.6B + Silero VAD: forced phrase scoring, asr_eval's decision + thresholds
services/eeg/           BNCI2014-008 loader, xDAWN+Riemann P300 decoder, replay payloads
services/coordinator/   Nemotron tool calling, outside validation, deterministic fallback
services/tts/           Kokoro-82M (MLX on a Mac, PyTorch on CUDA), refuses without a gateway token
shared/                 schemas, confirmation gate, provenance + ports
evaluations/            safety tests, modality baselines, e2e exactness, NeMo Gym env, results
asr_eval/               audio-only ASR benchmark for the voice agent (Brev GPU or CPU): manifest, VAD gate,
                        models, phrase decision, calib-only tuning, summary/plots; scripts/brev/ helpers
scripts/                setup, run, evals, weights ledger, vLLM/MLX serving, smoke test
scripts/brev/           whole backend on a Brev GPU: setup_app, start_app, sync_app, connect, bench_latency
docs/                   TECHNICAL_VERBOSE.md (everything in one place), architecture, model cards & licences,
                        datasets & consent, evaluation, BREV.md, SERVING.md
```

## Product invariant

**No unverified interpretation is ever spoken as the patient's intention.** This is enforced by code, not by prompting. See `shared/confirmation.py` and `evaluations/safety_tests/`.

**Exception on branch `ui`: demo auto-speak.** With `UNSPOKEN_AUTO_SPEAK=1` (currently the default) the gateway speaks an **accepted** reading (the voice agent accepted it, or lips and voice agree) without the person's confirmation; ambiguous readings, lip-only matches and corrections still wait for Yes/No (`evaluations/safety_tests/test_auto_speak.py`). It is audited as `demo_auto_spoken` by the system, never as a user action, and the policy gate requires its own audited event for the exact text. The invariant above holds only with `UNSPOKEN_AUTO_SPEAK=0`. Risks are listed in `docs/TECHNICAL_VERBOSE.md` §10.
