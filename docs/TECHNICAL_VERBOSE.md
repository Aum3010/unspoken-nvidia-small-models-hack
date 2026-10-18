# Unspoken: technical reference (verbose)

Everything about the system as it is on branch `ui` (2026-10-18): what runs where, every agent and its
contract, the decision and safety logic, the browser, how it is served on the NVIDIA Brev GPU, every measured
number, and the known issues. Shorter docs: [README](../README.md), [ARCHITECTURE](ARCHITECTURE.md),
[SERVING](SERVING.md), [BREV](BREV.md), [EVALUATION](EVALUATION.md), [web/README](../web/README.md).

---

## 1. What the system is

An assistive-communication research prototype for people who cannot speak:

- Several small specialist models read the person:
  - lips (silent mouthing);
  - a weak voice;
  - eye gaze, deliberate blinks and head nods/shakes;
  - recorded EEG (replay only).
- A coordinator (NVIDIA Nemotron 3.5 Lightning) decides *how to ask* about what was read.
- The result is spoken by a TTS voice. Code, not a model, decides what may be spoken.

```
browser (laptop)                                  backend (Brev GPU, or a Mac)
camera ─ face tracking (MediaPipe) ─┬─ gaze, blink, nod/shake ─► selection in the page
         mouth motion → webm clip ──┼──────────────────────────► lip agent      ─┐
mic ──── voice detector → wav clip ─┘──────────────────────────► voice agent    ─┤
                                                                 EEG replay     ─┤─► gateway ─► coordinator (Nemotron)
                                                                                 │      │
                     ◄── WebSocket: state, readings, questions, history ─────────┘      ├─► policy gate (code) ─► TTS (Kokoro)
```

**Two delivery modes for speech** (see §6):
- **Confirm mode** (`UNSPOKEN_AUTO_SPEAK=0`): nothing is spoken until the person confirms. This is the product rule
  in the README.
- **Demo auto-speak** (`UNSPOKEN_AUTO_SPEAK=1`, **the current default on `ui`**): the coordinator's top reading is
  spoken without a confirmation. It is audited as `demo_auto_spoken` by `system`, never as a user action.

---

## 2. Processes, ports and environments

| Process | Port | Code | Brev interpreter | Mac interpreter |
|---|---:|---|---|---|
| Web UI (Vite dev server) | 8800 (laptop) | `web/` | — (laptop, Node) | — |
| Gateway | 8000 | `services/gateway/app.py`, `state.py` | `/workspace/venv-app` | `.venv` |
| Lip agent (VSR) | 8101 | `services/vsr/` | `/workspace/venv-app` | `.venv` |
| Voice agent | 8102 | `services/audio/app.py` + `asr_eval/harness/` | `/workspace/venv-asr` (NeMo) | needs NeMo (not wired) |
| EEG replay | 8103 | `services/eeg/` | `/workspace/venv-app` | `.venv` |
| TTS | 8104 | `services/tts/app.py` | `/workspace/venv-app` | `.venv` (MLX) |
| Coordinator client | 8105 | `services/coordinator/app.py` | `/workspace/venv-app` | `.venv` |
| vLLM (Nemotron 3.5 Lightning) | 8001 | `scripts/brev/start_app.sh` | `/workspace/venv-vllm` | Ollama :11434 instead |
| NVIDIA MPS daemon | — | `start_app.sh` (`UNSPOKEN_MPS`, default 1) | system | — |

- **Binding:** every backend port binds `127.0.0.1`. The laptop reaches only the gateway, through an SSH tunnel
  (`scripts/brev/connect.sh`, or `brev port-forward`). Vite proxies `/api` and `/ws` to `127.0.0.1:8000`.
- **Health:** each service has `/health`; the gateway aggregates them at `/api/health`.
  - A service that was ready within the last 2 minutes but times out its 1.5 s probe is reported `ready` with
    `busy=true` instead of offline. The lip agent answers late while it decodes.
- **Versions on Brev:** torch 2.11.0+cu128 (app venv), vLLM 0.31.0 (its own venv, torch 2.13), NeMo 3.0.0
  (asr venv), Python 3.12 via `uv` (the asr venv is the box's Python 3.10).
  - Pinned lists: `~/unspoken/.run/requirements-cuda.lock` and `requirements-vllm.lock`.

---

## 3. The agents

### 3.1 Voice agent: `services/audio/app.py`

- **Models:**
  - NVIDIA Nemotron Speech Streaming en 0.6B (`nvidia/nemotron-speech-streaming-en-0.6b`, 618,084,865 params);
  - Silero VAD 6.2.3 (462,594 params) as a speech gate.
- **Method:** the clip is transcribed, then each of the **20 fixed phrases** (`asr_eval/configs/eval.yaml`) is
  scored with the model's own RNNT loss (`forced_llr`: per-token log-likelihood ratio of the phrase against the
  model's own transcript).
- **Same code as the evaluation:** decisions come from `asr_eval/harness/live.py` (`Bench`), the code
  `asr_eval/serve.py` uses, with thresholds frozen in `asr_eval/results/thresholds.json`.
- **Operating point:** `balanced` (≤5% false accepts per negative type on calibration):
  - `vad_t 0.2`;
  - `abs_t -1.155`;
  - `margin_t 0.405`.

  `strict` is the headline point.
- **Output (typed `Evidence`):**

  | Decision | Evidence status | Hypotheses |
  |---|---|---|
  | candidate (accepted) | `proposed` | the accepted phrase only |
  | unknown (low score, small margin or negation mismatch) | `ambiguous` + `low_word_confidence` | top 3 phrases |
  | no speech (Silero) | `unknown` + `no_speech_detected` | none |
  | the model heard no words | `unknown` + `low_word_confidence` | none |
  | clip < 0.25 s | `unknown` + `too_short` | none |

  Scores are raw `forced_llr` values, not probabilities. Never free text.
- **Evaluated accuracy** (3 recorded speakers, leave-one-speaker-out, strict; `asr_eval/results/loso.md`):
  - 71.5% of real phrases accepted correctly, 0.4% wrong;
  - 90.2% resolved within one question;
  - 0/45 near-miss false accepts.

### 3.2 Lip agent: `services/vsr/`

- **Model:** Auto-AVSR `LRS3_V_WER19.1` via Chaplin (250.4M), plus its subword LM (53.7M). Beam 12, CUDA fp32.
- **Preprocessing:** PyAV at 25 fps, then MediaPipe face detection, then a mean-face similarity warp, then a 96 px
  grey mouth ROI, centre-cropped to 88.
- **Recognition order:**
  1. **Personal enrollment** (`enroll.py`, `phrases.py`).
     - The person's own recordings are stored as encoder features in `data/enroll/<sha1>/NNN.npy` with
       `phrase.txt`: 22 phrases and 90 recordings on `ui`. These are derived from a real person's recordings and
       are committed; see §10.
     - A clip is matched by length-normalised DTW over cosine distance.
     - A match needs distance ≤ `ENROLL_MAX_DIST` 0.30.
     - Up to 3 matches are offered, scored −100 × distance.
     - A clip closest to a recorded **negative** ("I don't need water", `negative_phrases.txt`, stored as
       `[unknown] …`) gets no reading.
  2. **Optional closed phrase list** (`UNSPOKEN_VSR_PHRASES`, off by default): teacher-forced decoder + CTC score
     per phrase.
  3. **Free-text beam search**, only with `UNSPOKEN_VSR_FREE_TEXT=1`. It is off once phrases are enrolled; an
     unmatched clip returns no reading at once.
- **Quality flags:**
  - `too_short` (< 0.6 s);
  - `no_face`;
  - `partial_face` (< 90% of frames);
  - `occlusion` (crop failed);
  - `low_mouth_motion` (mean frame difference < 2.2).
- **CPU threads:** `OPENBLAS_NUM_THREADS=1` is set before numpy loads. Multi-threaded OpenBLAS contended with
  the models and made enrollment matching 10–30 s per clip.
- **Measured:**
  - 391 ms per clip at the service: about 250 ms video decode + face detection + crop, encoder about 10 ms, DTW
    127 ms;
  - with beam search, decoding adds about 280 ms.
  - **Free-text accuracy** (beam 12, 132 GRID clips, 33 unseen speakers): WER 69.4%, best-of-5 60.9%.
  - **Enrollment matcher on strangers:** 5 of 43 GRID clips (unrelated people and sentences) matched an enrolled
    phrase within 0.30 (about 12%).

### 3.3 Coordinator: `services/coordinator/app.py`

- **Model:** NVIDIA Nemotron 3.5 Lightning 30B-A3B, NVFP4 checkpoint, served by vLLM on Brev; Ollama on a Mac.
- **Call settings:** OpenAI-compatible tool calling, `temperature 0`, `max_tokens 200`, thinking off, 20 s timeout.
- **Input:** a structured summary only (candidates, evidence statuses and flags, available inputs, gaze
  calibration, history), never media.
- **Measured cost:** 1,784 prompt tokens and about 73–86 output tokens per call.
- **Tools:**
  - `ask_confirmation(candidate_ids, input_method)`: offer 1–4 existing candidates.
  - `request_retake(modality, reason)`.
  - `switch_input_method(method)`: board, spelling, gaze, blink, button, audio.
  - `propose_correction(source_candidate_ids, text, input_method)`: fix misread words ("i wore my family" →
    "i want my family"). The gateway adds it as a new candidate with origin `correction`.
  - `ignore_input(reason)`: noise, not a message. The gateway drops the readings and asks nothing.
  - `speak_confirmed_text`: always rejected without a confirmation event.
- **Validator** (outside the model):
  - ids exist;
  - the method is available;
  - no gaze unless calibrated;
  - **corrections:** ≤ 12 words, at least half of the words similar (ratio ≥ 0.6) to a word that was read, length
    within ±2 words, only lip/audio evidence readings;
  - **`ignore_input`:** rejected if a candidate is supported by 2 modalities, or is not a lip/audio reading.
  - An invalid call falls back to the deterministic policy.
- **Deterministic policy:** applied instantly to every new evidence (quietly), so options appear at once; Nemotron
  refines it in the background.
- **Evaluation** (NeMo Gym environment, 72 tasks, Brev, 2026-10-18,
  `evaluations/results/coordinator_brev.json`):

  | Policy | Reward | Valid | Right action | Unsafe |
  |---|---:|---:|---:|---:|
  | Nemotron alone | 0.926 | 1.00 | 0.78 | **0** |
  | Deployed (Nemotron + validator + fallback) | 0.931 | 1.00 | 0.79 | **0** |
  | Rules only | 1.000 | 1.00 | 1.00 | 0 |

  Every wrong action of the deployed coordinator is `ignore_input` on real input:
  - after two failed tries (6/6; it should offer another input method);
  - an injection next to a real request (3/6);
  - uncalibrated gaze (2/6);
  - a clear negated reading (3/6);
  - one misread.

  Misreads were handled right (corrected or asked) in 5 of 6 tasks; the miss was an `ignore_input`. **No negation
  was ever dropped.** But the validator would
  accept "I don't need water" → "I need water". The validator is unchanged on purpose (decision of 2026-10-18).

### 3.4 TTS: `services/tts/app.py`

- **Model:** Kokoro-82M (`hexgrad/Kokoro-82M` via `kokoro` on CUDA; `mlx-community/Kokoro-82M-bf16` on a Mac), 24 kHz.
- **Voices:** af_heart, af_bella, am_michael, am_fenrir, bf_emma, bm_george.
- **Gate:** refuses any request without a gateway HMAC token for exactly `(message_id, text)`.
- **Measured:** 22–23 ms per short phrase alone; 30 ms with every agent busy under MPS.

### 3.5 EEG replay: `services/eeg/`

- **Data and decoder:** BNCI2014-008 (8 ALS participants); xDAWN covariances → tangent space → logistic
  regression, trained per participant.
- **Recorded data only:** evidence is labelled `RECORDED EEG` and confirmed by an operator.
- **Accuracy:** flash AUC 0.850; letter accuracy 94% at 10 repetitions.
- **Speed:** 106–117 ms per letter on the CPU.

---

## 4. Gateway: `services/gateway/app.py`, `state.py`

**Session state machine:** collecting → proposed / ambiguous → clarifying → selected → confirmed → spoken
(→ rejected). It is deterministic code.

**Capture paths:**

| Route | What it does |
|---|---|
| `POST /api/{sid}/capture/lip` | upload → lip agent → evidence |
| `POST /api/{sid}/capture/audio` | upload → voice agent → evidence |
| `POST /api/{sid}/capture/audio/early` | starts the voice decode **without touching the session**; returns `early_id` (at most 4 per session) |
| `POST /api/{sid}/capture/audio/commit` | waits for that decode (usually already done) and applies it like a normal capture |
| `POST /api/{sid}/capture/audio/discard` | cancels it |

**Noise screening (code):** evidence flagged `no_face`, `low_mouth_motion`, `too_short` or `no_speech_detected`,
or with status `unknown` / no hypotheses, is logged as `evidence_ignored` and never becomes a candidate.
- The trace reads "Ignored mouth clip" or "Ignored voice clip".
- Early-committed voice clips go through the same rule.

**Utterances:** with always-on capture, an unanswered reading is **replaced** by the next lip or audio clip, not
stitched to it. A lip and a voice clip pair into one message when they are within 10 s
(`UTTERANCE_GAP_MS`) and of different modalities. Identical texts fuse into one candidate with both modalities.

**Coordination:**
1. The deterministic plan is applied first, quietly.
2. Nemotron runs in the background (`_refine_with_model`).
3. Its plan is dropped if the person acted meanwhile.

**Auto-speak** (default on): when an `ask_confirmation` or `propose_correction` plan is applied,
`_auto_speak` is scheduled at once (`asyncio.create_task`). It auto-confirms `shown_candidate_ids[0]` as `system`
and speaks it through the same policy gate with `demo_auto=True`.

**Policy gate:** `shared/confirmation.py:authorize_speech`.
- In confirm mode it requires:
  - status `confirmed`;
  - an explicit person-made confirmation method (gaze dwell, blink, button, keyboard, other);
  - the confirmation event in the audit log by a person;
  - text unchanged since.
- In auto mode it requires the system's own `auto_confirmed` audit event for the exact text.
- The result is an HMAC token over `(message_id, text)`, which TTS verifies.

**Audit log:** raw hypotheses, ignored evidence, tool calls (including rejected ones), selections, corrections,
confirmations, speech.

---

## 5. Browser: `web/` (React 19 + Vite, port 8800)

### Page

One liquid-glass page (`web/src/glass/`) over the live camera:
- **Rail:** Phrases, Clear, Help. **Big Yes / No targets** at the top. There is no Speak button.
- **Agent activity panel:**
  - a canvas nucleus with six input nodes (voice, lips, gaze, blink, brain, head);
  - "N of 6 inputs on";
  - the coordinator's line ("Nemotron is planning…", "Fixed a misread word", "Ignored as noise");
  - the last 3 gateway timeline items.
- **History:** spoken messages with modality dots, method and time; click one to replay it.
- **Ink:** the reading or choice is written as faint ink, then drawn in by the pen once spoken.

### Sensors (`glass/sensors.ts`)

All of these run at once.

**Voice:**
- AudioWorklet, 2048-sample blocks, adaptive noise floor.
- Onset: 3 loud blocks (about 130 ms). Pre-roll: 350 ms.
- **Early send:** at 350 ms of quiet.
- **Utterance end:** at 700 ms of quiet.
- At least 350 ms of voicing; maximum 10 s.
- Sent as a 16 kHz WAV.

**Lips:**
- The standard deviation of the lip gap divided by face height over 600 ms must exceed 0.012.
- Recording starts at the first movement and counts after 200 ms. The clip is sent after 900 ms of stillness;
  minimum 0.9 s, maximum 8 s.
- WebM VP9/VP8 at **2.5 Mbps** (about 1.2 MB for a 3–4 s clip).

**Head:** a nod is pitch past 9° and back under 4° within 1 s. A shake is yaw past ±12° on each side within 1.2 s.

**Blink:** both eyelids above 0.5 for 380–1800 ms (deliberate).

**One clip in flight per agent** (`sendClip`). Capture also pauses while:
- a message is selected or confirmed;
- speech plays, plus 700 ms;
- calibrating;
- the gateway is disconnected;
- the agent is not ready.

### Gaze (`lib/gaze.ts`, `appearance.ts`, `ridge.ts`)

- **Inputs:** MediaPipe iris/eyelid geometry (9 features, no head-pose terms) plus MobileGaze MobileOne-S0
  yaw/pitch. MobileGaze runs on ONNX Runtime Web: WebGPU, then wasm, at most about 12 runs/s.
- **Model:** a per-user ridge regression. One-euro smoothing; a blink holds the last estimate.
- **Head fine-tune:** bounded to 1.2× the measured error.
- **Calibration** (`glass/Calibration.tsx`): an own plain black screen.
  - 11 dots on the **real layout**: Yes/No, 3 rail tiles, 6 phrase-card spots.
  - Per dot: 600 ms settle, then sampling until 12 samples and at least 1.6 s (at most 4.5 s).
  - Accuracy: 3 unseen points, mean of the best 80% of errors. Good < 120 px, fair < 220 px.
- **Saved calibration:** `localStorage` `unspoken.gaze.v6`. It is reused automatically if the window size is
  within 40 px, and the calibration screen is then skipped.
- **Learning from use:** each completed eye dwell labels its frames with the target centre and refits.
- **Targeting:**
  - zone targeting for the rail strip, the Yes/No band and the cards (40 px hysteresis);
  - elsewhere, the nearest target within the gaze error (clamped 50–160 px);
  - dwell 1.2 s, or 1.8 s on confirm targets;
  - after 2.5 s off-screen, a "recalibrate" caption appears.

### Actions

- **Phrase card:** chosen = spoken at once (board → confirm).
- **Yes, a nod, or a blink with no target under the gaze:** confirm the pending or selected reading.
- **No / shake:** undo or reject.
- **Help:** red flash, then speaks "I need help now."
- **Keys:** Space = blink, P, C, E (calibrate), Esc.
- **Confirmation methods audited:**

  | Input | Method |
  |---|---|
  | Eye dwell | `gaze_dwell_confirmed` |
  | Pointer dwell, nod | `other_explicit_user_confirmed` |
  | Blink | `deliberate_blink_confirmed` |
  | Space | `keyboard_confirmed` |
  | Click | `accessible_button_confirmed` |

### Required files and known UI issues

- **Required browser files** (gitignored; `scripts/setup_models.sh` fetches them):
  - `web/public/mediapipe/face_landmarker.task` and `mediapipe/wasm/`;
  - `web/public/models/mobileone_s0_gaze.onnx`.
- **Known UI issues on `ui`:**
  - **Privacy text:** the camera chip always says "Camera · on device", even when clips go to Brev.
  - **Camera opening:** the shared camera-open promise and the single face-tracking loop were reverted
    (`3d3e95c`), so slow USB cameras can stall on parallel `getUserMedia` calls.
  - **Unused dependency:** `webgazer` is listed but unused.
  - **Unreachable action:** the `speak` action has no button.

---

## 6. How it is served (Brev)

### The instance

`marginal-sapphire-whale`:
- RTX PRO 6000 Blackwell Server Edition, 96 GB, driver 580.126.09 (CUDA 13.0);
- 16 vCPU, 141 GB RAM;
- $2.63/hr, not stoppable.

### GPU memory

About 63 GB peak, all agents loaded:

| Process | Memory |
|---|---:|
| vLLM | 53.6 GB (weights 17.8 GB + KV cache, `--gpu-memory-utilization 0.55`) |
| Voice agent | 6–9 GB |
| Lip agent | 2.2 GB |
| Kokoro | 1.4 GB |
| MPS server | 0.06 GB |

### vLLM settings

From `start_app.sh`:
- `--max-model-len 16384`, `--max-num-seqs 8`;
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`;
- `--enable-prefix-caching`;
- `--attention-backend TRITON_ATTN` and `VLLM_USE_FLASHINFER_SAMPLER=0`: FlashInfer JIT-compiles with `nvcc`,
  which the image lacks;
- on sm_120 vLLM picks **Marlin weight-only FP4** (activations in 16 bit).
- **Prefix caching hits 0%** (0 of 222,877 cached-token lookups). Every call re-reads the full prompt; the
  model's Mamba layers are the likely reason (not confirmed).

### NVIDIA MPS

- `start_app.sh` starts `nvidia-cuda-mps-control -d` before any GPU process (`UNSPOKEN_MPS=1`, the default) or
  stops it (`UNSPOKEN_MPS=0`).
- It detects the daemon by its full command line, because `pgrep -x` cannot match the 15-character-truncated name.

### Commands

```bash
bash scripts/brev/sync_app.sh marginal-sapphire-whale                   # ship committed code (services, shared, scripts, asr_eval bits)
git archive --format=tar.gz -o /tmp/enroll.tgz HEAD:data enroll         # enrollment is NOT in sync_app: copy it
brev copy /tmp/enroll.tgz marginal-sapphire-whale:/home/shadeform/enroll.tgz
brev exec marginal-sapphire-whale 'cd ~/unspoken && rm -rf data/enroll && tar -xzf ~/enroll.tgz -C data'
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh'   # start | status | stop
bash scripts/brev/connect.sh marginal-sapphire-whale                    # tunnel :8000
cd web && npm run dev                                                   # http://localhost:8800
```

### Fixes for problems already met

See [BREV.md](BREV.md) §5:
- `mne` pinned to 1.11.0 (newer mne needs numpy ≥ 2);
- the cu128 index needed so the pip resolver honours the torch constraint;
- line endings forced to LF when shipping from Windows;
- vLLM compiler workarounds (no `nvcc` on the image);
- ports given time to free up before services restart;
- `brev port-forward` slow to appear.

---

## 7. Latency (Brev, warm, batch 1)

### On the GPU box (no laptop network)

Branch `ui` with MPS and early decode. Files: `evaluations/results/latency_brev_ui_mps{0,1}.json`; as pushed, before
these were added: `latency_brev_ui_as_pushed.json`.

| p50 / p95 ms | MPS off | MPS on |
|---|---:|---:|
| Voice, every agent busy | 244 / 375 | **75 / 118** |
| Kokoro, every agent busy | 79 / 93 | **30 / 62** |
| Coordinator, every agent busy | 352 / 425 | 336 / 342 |
| Lips, every agent busy | 436 / 468 | 404 / 464 |
| Alone: voice / lips / coordinator / Kokoro | 60 / 409 / 314 / 23 | 61 / 391 / 330 / 22 |
| One message end to end (voice + lip clip, auto-spoken) | 434 / 469 | 435 / 474 |
| Voice result after the 700 ms pause, sent then (old) | 337 / 377 | 109 / 119 |
| Voice result after the 700 ms pause, **early decode** | **3 / 22** | **2 / 2** |

### From the laptop through the tunnel

| Measurement | Result |
|---|---|
| Round trip to the gateway | about 120 ms |
| Upload speed | about 1.75 MB/s (14 Mbps) |
| 64 KB voice clip upload | 0.23 s |
| 1.2 MB lip clip upload | **0.68 s** (the server works 34 ms of it) |
| Voice result after the pause, old way | 211 ms p50 |
| Voice result after the pause, early decode | **64 ms** p50 |

### What a person waits for

| Step | Voice message | Lip message |
|---|---:|---:|
| Pause before the clip is cut | 0.7 s | 0.9 s |
| Upload | — (hidden by early decode) | 0.68 s |
| Agent | — (hidden by early decode) | about 0.4 s |
| Coordinator (auto-speak waits for it) | about 0.33 s | about 0.33 s |
| Kokoro + audio back + WebSocket | about 0.15–0.3 s | about 0.15–0.3 s |
| **About** | **1.2–1.4 s** | **about 2.5 s** |

### Decisions taken for speed, all measured

- **MPS on:** no change in arithmetic.
- **Duplicate Nemotron Speech stopped:** the asr_eval bench copy; frees about 6 GB.
- **Lip beam 6 rejected:** WER 69.4 → 71.1%, best-of-5 60.9 → 64.5%.
- **fp16 for lips not adopted:** same answers, not faster. bf16 cannot run (Chaplin's attention).
- **Early voice decode adopted:** 0 of 304 real recordings changed decision with the 350 ms tail
  (`asr_eval/results/trailing_silence_check.json`).
- **Native FP4 for the coordinator deferred:** less accurate (4-bit activations); do it only after giving vLLM
  its own GPU, and only if NeMo Gym holds.

### Open speed items

- Lip video bitrate: needs an accuracy check first.
- Coordinator output length.
- Prefix caching.
- GPU isolation for vLLM (2 GPUs or MIG).

---

## 8. Model budget (≤ 33B)

31.01B with the coordinator counted at its nominal 30B; 32.59B with its weight-file count of 31.58B. Both are within
budget. Source: `evaluations/results/weights.json`, `scripts/verify_weights.py`.

| Model | Size |
|---|---:|
| Nemotron 3.5 Lightning (nominal) | 30.000B |
| Nemotron Speech en 0.6B | 0.618B |
| Auto-AVSR | 0.250B |
| Auto-AVSR LM | 0.054B |
| Kokoro | 0.082B |
| MediaPipe Face Landmarker | 0.0019B |
| MobileGaze | 0.0012B |
| MediaPipe face detection | 0.0006B |
| Silero VAD | 0.0005B |
| P300 decoder | ~0 |

The lip enrollment adds stored features, not parameters.

---

## 9. Evaluation scripts

| What | Command | Result file |
|---|---|---|
| Safety tests (policy gate) | `pytest evaluations/safety_tests` (18 pass) | — |
| Coordinator (NeMo Gym, 72 tasks) | `run_local.py --policies raw:<model> guarded deterministic` | `coordinator_brev.json` (Brev); `coordinator.json` (Mac, 60 tasks, older prompt) |
| Lip speed vs accuracy | `evaluations/modality_baselines/vsr_speed_accuracy.py` | `vsr_speed_accuracy.json` |
| Early-decode accuracy check | `asr_eval/scripts/trailing_silence_check.py` | `trailing_silence_check.json` |
| Latency | `scripts/brev/bench_latency.py` | `latency_brev_*.json` |
| Voice model selection | `asr_eval/run_eval.py`, `scripts/loso.py` | `asr_eval/results/` |

The NeMo Gym data files (`data/*.jsonl`) embed the coordinator's prompt and tools; regenerate them with
`create_examples.py` after any coordinator change.

---

## 10. Known issues on `ui` (for the team)

1. **Auto-speak is on by default.**
   - Speech happens without the person's Yes. It contradicts the README's product rule, and no safety test
     covers it.
   - It is scheduled immediately, so a No rarely arrives in time.
   - It can speak a reading the voice agent did *not* accept: the top of an "unknown" top 3.
   - With `propose_correction`, an edited sentence can be spoken unconfirmed.
2. **Coordinator overuses `ignore_input`.** A person who failed twice, a genuine request next to an injection,
   and clear negated readings get dropped as noise (§3.3). The validator allows it.
3. **The correction validator does not check negations.** It accepts "I don't need water" → "I need water".
4. **Personal lip data in git.** `data/enroll/` holds encoder features of a real person's recordings, force-added
   past the `/data/` gitignore rule, while `enroll.py` says they "are not uploaded".
5. **Disclosure missing.** The UI no longer tells the person that clips go to Brev.
6. **Enrollment threshold barely measured.** It was set from 10 recordings, and strangers matched in about 12%
   of GRID clips.
7. **`zero_infinity=True`** in `services/vsr/phrases.py` gives impossible alignments a perfect CTC score (only
   with `UNSPOKEN_VSR_PHRASES` set).
8. **Camera stalls.** The camera-open sharing and single tracking loop were reverted (slow USB cameras can
   stall).
9. **`sync_app.sh` does not ship `data/enroll/`.** It must be copied separately (§6).
10. **Smaller items:**
    - the Mac path needs NeMo for the voice agent;
    - `web/README.md` was reset to the Vite template on `ui` (restored by this doc update);
    - 8 lint warnings.
