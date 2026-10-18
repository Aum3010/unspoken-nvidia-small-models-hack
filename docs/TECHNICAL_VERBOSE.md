# Unspoken: technical reference (verbose)

Everything about the system as it is on branch `ui` (2026-10-18): what runs where, every agent and its
contract, the decision and safety logic, the browser, how it is served on the NVIDIA Brev GPU, every measured
number, and the known issues. Shorter docs: [README](../README.md), [ARCHITECTURE](ARCHITECTURE.md),
[SERVING](SERVING.md), [BREV](BREV.md), [EVALUATION](EVALUATION.md), [web/README](../web/README.md). Bounty
write-ups (Nemotron Lightning, NeMo Gym, vLLM): [BOUNTIES](BOUNTIES.md).

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
- **Demo auto-speak** (`UNSPOKEN_AUTO_SPEAK=1`, **the current default on `ui`**): an **accepted** reading is spoken
  without a confirmation. Accepted means the voice agent accepted it, or lips and voice agree on it. Everything
  else waits for Yes/No:
  - ambiguous readings;
  - lip-only matches;
  - lip/voice disagreement;
  - the coordinator's corrections.

  It is audited as `demo_auto_spoken` by `system`, never as a user action.

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

**Flags** (environment variables; the defaults are what runs today):

| Flag | Default | Read by | What it does |
|---|---|---|---|
| `UNSPOKEN_AUTO_SPEAK` | `1` | gateway (`state.py`) | demo auto-speak of **accepted** readings; `0` = nothing is spoken until the person confirms |
| `UNSPOKEN_LLM_FOR_ACCEPTED` | `0` | gateway | `1` = call Nemotron for accepted readings too (no rules-only fast path) |
| `UNSPOKEN_HOST_LABEL` | `this device` (`NVIDIA Brev` from `start_app.sh`) | gateway `/api/health` | where clips are processed; any value other than `this device` makes the page say clips go there |
| `UNSPOKEN_COORD_COMPACT` | `0` | coordinator | `1` = schema-constrained short JSON answers (faster, less accurate; §3.3) |
| `UNSPOKEN_VALIDATOR_NEGATION` | `0` | coordinator | `1` = reject a correction that adds or removes a negation (§3.3) |
| `UNSPOKEN_VALIDATOR_IGNORE_HISTORY` | `0` | coordinator | `1` = reject `ignore_input` after two failed tries, a recent "none of these", or on an accepted clean reading (§3.3) |
| `UNSPOKEN_AUTO_SPEAK_DELAY_MS` | `0` | gateway | > 0 = a countdown before an accepted reading is auto-spoken; No or a shake cancels it (§4) |
| `UNSPOKEN_CHECKIN` | `0` | gateway + page | `1` = vision preview: proactive check-in routes and panel (§4); `0` = none of it exists (routes 404) |
| `UNSPOKEN_LLM_BACKEND`, `_BASE_URL`, `_MODEL`, `_TIMEOUT` | `ollama`, `http://127.0.0.1:11434`, `nemotron-3.5-lightning:30b-a3b`, `20` | coordinator | the model server; `start_app.sh` sets `openai` + vLLM on :8001 |
| `UNSPOKEN_VSR_PHRASES` | empty | lip agent | optional closed phrase list (off) |
| `UNSPOKEN_VSR_FREE_TEXT` | `0` | lip agent | `1` = free-text beam search when no recorded phrase matches |
| `UNSPOKEN_ENROLL_MAX_DIST` | `0.30` | lip agent | enrollment match threshold (DTW distance) |
| `UNSPOKEN_MPS` | `1` | `start_app.sh` | NVIDIA MPS daemon on/off |
| `LLM_MTP` | `0` | `start_app.sh` | MTP speculative decoding for vLLM (§6) |
| `LLM_EXTRA` | empty | `start_app.sh` | extra vLLM flags, for experiments |
| `LLM_MEM`, `LLM_ATTN` | `0.55`, `TRITON_ATTN` | `start_app.sh` | vLLM memory share and attention backend |

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
     - A clip is matched by length-normalised DTW over cosine distance. Every recording is compared in one
       batched pass (`dtw_batch`): the recordings are zero-padded into one array and filled one anti-diagonal at a
       time. On 270 test queries, distances match the per-recording loop to within 3e-8, with 0 top-3 changes.
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
- **Streamed clips** (`/stream/{id}/chunk`, `/finish`, `/discard`):
  - The browser sends 250 ms pieces while the person mouths. A thread per clip decodes them as they arrive (PyAV
    reading a growing buffer) and runs face detection on each 25 fps grid frame once it is final.
  - After the pause only cropping, the encoder and matching remain.
  - The frames are exactly those of a whole-clip decode: 12/12 GRID clips (VP9 and VP8 live webm) gave identical
    readings, scores, face ratio, mouth motion and enrollment distance.
  - The whole-clip `/infer` stays as the fallback.
- **Measured** (3 s 1280×720 clip, Brev):
  - decode 188 ms, face detection 247 ms, crop 29 ms, encoder 20 ms, DTW 34 ms (batched; 127 ms before);
  - whole clip `/infer` 492 ms p50, streamed **97 ms** after the end of the clip;
  - with beam search, decoding adds about 280 ms.
  - **Free-text accuracy** (beam 12, 132 GRID clips, 33 unseen speakers): WER 69.4%, best-of-5 60.9%.
  - **Enrollment matcher on strangers:** 5 of 43 GRID clips (unrelated people and sentences) matched an enrolled
    phrase within 0.30 (about 12%).

### 3.3 Coordinator: `services/coordinator/app.py`

- **Model:** NVIDIA Nemotron 3.5 Lightning 30B-A3B, NVFP4 checkpoint, served by vLLM on Brev; Ollama on a Mac.
- **Call settings:** OpenAI-compatible tool calling, `temperature 0`, `max_tokens 200`, thinking off, 20 s timeout.
- **When it runs:** only for ambiguous, unknown or disagreeing readings. Accepted readings are decided by the rules
  without it (§4).
- **Compact mode** (`UNSPOKEN_COORD_COMPACT=1`, **off**):
  - The model answers with one schema-constrained JSON object (`{"a":"ask","ids":[1],"m":"button"}`) and sees the
    candidates numbered 1..n. Code maps the answer back to the same tools and ids before the validator, and writes
    the note to the person from a template.
  - It is 3× faster: 84 ms p50 and 15 output tokens, against 272 ms and 65.
  - The deployed reward holds (0.949 vs 0.940–0.949), and injection, misread and negation improve. But it offers
    too few options on ambiguous readings (0.89 vs 1.00) and corrects after a rejection (0.78–0.89 vs 1.00).
    Ambiguous readings are exactly what the coordinator still handles, so it stays off.
  - Files: `coordinator_compact_brev.json`, `coordinator_compact_mtp_brev.json`.
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
  - **Optional rules (flags, off by default):**
    - `UNSPOKEN_VALIDATOR_NEGATION`: a correction may not add or remove a negation. It uses asr_eval's
      normalisation and negation words.
    - `UNSPOKEN_VALIDATOR_IGNORE_HISTORY`: no `ignore_input` after two failed tries, after a recent "none of
      these", or on an accepted lip/audio reading with no quality flags.
    - Paired NeMo Gym, deployed: eval **0.954 → 0.995**, held-out **0.977 → 0.985**, 0 unsafe.
    - It fixes `repeated_failure` (0.67 → 1.00) and `negation_kept` (0.89 → 1.00).
    - The negation rule changed nothing measured. See EVALUATION §4.
- **Deterministic policy:** applied instantly to every new evidence (quietly), so options appear at once; Nemotron
  refines it in the background.
- **Evaluation** (NeMo Gym environment, 72 tasks, Brev, 2026-10-18,
  `evaluations/results/coordinator_brev.json`, the latest of six runs):

  | Policy | Reward | Valid | Right action | Unsafe | p50 |
  |---|---:|---:|---:|---:|---:|
  | Nemotron alone | 0.917 | 1.00 | 0.75 | **0** | 272 ms |
  | Deployed (Nemotron + validator + fallback) | 0.940 | 1.00 | 0.82 | **0** | 275 ms |
  | Rules only | 1.000 | 1.00 | 1.00 | 0 | 0 |

  - **Run-to-run spread** over six Brev runs of the default prompt (including the MTP and cache-mode server
    variants): Nemotron alone 0.912–0.931, deployed 0.940–0.949.
  - **Held-out** (44 tasks: seed 2026 without the tasks that also occur in the eval set; 10 categories;
    `coordinator_heldout_brev.json`):

    | Policy | Reward | Right action | Unsafe |
    |---|---:|---:|---:|
    | Nemotron alone | 0.939 | 0.82 | 0 |
    | Deployed | **0.985** | 0.95 | 0 |

    The whole gain is `none_of_these` (0.67 → 1.00), from the validator's history rule and the fallback.
    Nemotron decides 86% of the tasks itself. Compact mode on the same tasks: 0.894 / 0.955
    (`coordinator_heldout_compact_brev.json`).
  - Most wrong actions of the deployed coordinator are `ignore_input` on real input:
    - after two failed tries (6/6; it should offer another input method);
    - an injection next to a real request (2–3/6);
    - uncalibrated gaze;
    - a clear negated reading (2–3/6).

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
| `POST /api/{sid}/capture/lip` | upload → lip agent → evidence (fallback for a broken stream) |
| `WS /ws/{sid}` binary + `POST /api/{sid}/capture/lip/stream` | streamed lip clip: pieces while mouthing, then the reading |
| `POST /api/{sid}/capture/audio` | upload → voice agent → evidence |
| `POST /api/{sid}/capture/audio/early` | starts the voice decode **without touching the session**; returns `early_id` (at most 4 per session) |
| `POST /api/{sid}/capture/audio/commit` | waits for that decode (usually already done) and applies it like a normal capture |
| `POST /api/{sid}/capture/audio/discard` | cancels it |
| `POST /api/{sid}/phrase_audio`, `GET /api/phrase_audio/{id}` | pre-rendered audio of the fixed phrases |

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

**Accepted readings skip Nemotron** (`_accepted_reading`, `UNSPOKEN_LLM_FOR_ACCEPTED=0` by default):
- When the message's only candidate comes from evidence that is all `proposed`, and the voice agent supports it, the
  rules' plan (ask about that one reading) is applied at once and Nemotron is not called.
- Nemotron runs for ambiguous, unknown and disagreeing readings only.
- Measured on Brev: accepted voice message → spoken **413 → 105 ms** p50 in the run right after this change, and
  71–73 ms in later runs (§7; not from phrase audio, cause not isolated). Nemotron ran for 2 of 40 messages (the
  ambiguous ones).
  - Source: Measured with an ad-hoc voice-only bench on the box (40 Kokoro-rendered phrase clips), printed, not saved to a file: 413 / 459 ms before the change (Nemotron planned every message); 106 / 120 ms with accepted readings decided by rules (commit `51df483`); 105 / 115 ms after auto-speak was limited to accepted readings (`09c44c1`); 71 / 89 ms in the final run (same run as `latency_brev_final2.json`); on `demo-final`, 73 / 84 ms with the gateway's phrase cache cold and 71 / 80 ms warm. So the later 105 → 71 ms drop is **not** from pre-rendered phrase audio (worth about 2 ms), and its cause was not isolated.

**Auto-speak** (default on): only for an `ask_confirmation` of exactly the accepted reading. `_auto_speak` is
scheduled at once; it auto-confirms the reading as `system` and speaks it through the same policy gate with
`demo_auto=True`. The snapshot's `auto_speak_candidate_id` tells the page which caption to show ("Understanding
you…" or "Did you mean this?"). Tests: `evaluations/safety_tests/test_auto_speak.py`.

**Optional countdown** (`UNSPOKEN_AUTO_SPEAK_DELAY_MS`, default 0 = speak at once):
- `_auto_speak` waits that long, then speaks only if the same message is still pending with the same accepted
  reading.
- The snapshot carries `auto_speak_in_ms` (relative, so clock skew does not matter), and the page shows
  "Speaking in N s… No or shake to stop it".
- No or a shake (reject) inside the window means nothing is spoken. A Yes inside the window speaks once, by the
  person.
- Tests: `test_auto_speak_delay.py`.

**Pre-rendered phrase audio** (`POST /api/{sid}/phrase_audio`, `GET /api/phrase_audio/{id}`). **Known bug:** the
manifest request returns HTTP 500, so the page never preloads; the gateway-side cache still fills (§10):
- The fixed phrases are rendered once per voice and rate and preloaded by the page: the voice agent's 20, the
  recorded lip phrases, the board and Help.
- The TTS token for pre-rendering comes from `phrase_preload_token` (message id `preload-…`).
- When `authorize_speech` passes for exactly one of these texts, the gateway skips Kokoro. The spoken entry carries
  `phrase_audio_id`, and the page plays its local copy, with no download.
- Edited, spelled and corrected text still goes to Kokoro.

**Streamed lip clips:** pieces arrive as binary WebSocket messages (2-byte header length, JSON header
`{id, seq, mime}` or `{id, discard}`, then the bytes). They are forwarded in order to the lip agent.
`POST /api/{sid}/capture/lip/stream {stream_id, chunks}` waits up to 5 s for all pieces, then asks for the reading.
A broken stream answers 409 and the page uploads the whole clip.

**Vision preview check-in** (`UNSPOKEN_CHECKIN=1`, off by default; `services/gateway/checkin.py`):
- **When it is off:** no route exists (`/api/{sid}/vision/checkin` answers 404), the snapshot has no `checkin`
  key, and the page shows nothing.
- **When it is on:** `POST /api/{sid}/vision/checkin` (or V, or the panel button):
  1. traces "🔔 Nurse alerted (vision preview)" and pushes it;
  2. then shows the fixed question "Your nurse is coming to check on you. Do you feel shaky?" with the usual
     Yes/No (no model).
- **Answering:** Yes, a blink or a nod answers YES; No or a shake answers NO
  (`POST /api/{sid}/vision/checkin/answer`). The answer is logged as `checkin_response` and shown as "Nurse
  notified: patient answered YES/NO".
- **Not speech:** it never goes through `authorize_speech` or TTS, and it creates no candidate or message.
- **Queueing:** while a message is pending (proposed / ambiguous / clarifying / selected / confirmed), or within 3 s
  of speech, the check-in waits and opens once the session is idle.
- **No talking over it:** while it is open, auto-speak is paused.
- **Data:** `data/vision/mimic_event.json` from `scripts/vision/fetch_mimic_event.py`. It holds one charted
  glucose below 70 mg/dL from the MIMIC-IV demo (stay 30757476: serum glucose 64 mg/dL) and its 6 h of heart rate
  and SpO2.
- **Profile:** `data/vision/profile.yaml`, SYNTHETIC ("Raj", family member "Priya"). Both are gitignored.
- **Panel label:** "Recorded ICU data (MIMIC-IV demo) · synthetic profile · vision preview".
- **Tests:** `test_checkin.py`.

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
- WebM VP9/VP8 at **2.5 Mbps** (about 1.2 MB for a 3–4 s clip), recorded in 250 ms pieces that are **sent while
  the person mouths**, in order, over the session WebSocket. At the stop, the page asks for the reading; if the
  stream broke it uploads the whole clip.

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
  - **Privacy text:** fixed on `demo-final`; the chip and calibration screen name the remote host.
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
- **Prefix caching hits 0%, and cannot hit with our prompt.**
  - Even 3 identical requests get 0 hits.
  - The model's Mamba state per sequence is large, so vLLM raises the cache block to **4,224 tokens** ("Setting
    attention block size to 4224 tokens to ensure that attention page size is >= mamba page size").
  - Only whole blocks are cached, and the coordinator prompt is about 1,700 tokens.
  - `--mamba-cache-mode all` (deprecated) does not change this.
- **MTP speculative decoding** (`LLM_MTP=1`, **off**):
  - The checkpoint has one next-token layer (`num_nextn_predict_layers: 1`, vLLM `NemotronHMTP`). With
    `{"method":"mtp","num_speculative_tokens":1}`, 88% of drafts are accepted.
  - The coordinator goes from 272 to **229 ms** p50, and NeMo Gym stays within run-to-run noise
    (`coordinator_mtp_brev.json`).
  - **Off because of one crash:** in the benchmark's "every agent at once" phase, vLLM with MTP failed
    (`cudaErrorLaunchFailure`). Under MPS this took every GPU process down (TTS, voice, lips: illegal memory
    access).
  - MTP alone, 8 concurrent requests × 30 rounds, passed 240/240. Not reproduced, not explained.
  - It also fails (HTTP 500) together with `--mamba-cache-mode all`.

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

The "Now" column is MPS on, MTP off, after the speedups below (`latency_brev_final2.json`). The "MPS off" and
"MPS on" columns are branch `ui` with MPS and early decode (`latency_brev_ui_mps{0,1}.json`). Before any of these:
`latency_brev_ui_as_pushed.json`.

| p50 / p95 ms | MPS off | MPS on | **Now** |
|---|---:|---:|---:|
| Voice, every agent busy | 244 / 375 | 75 / 118 | 78 / 106 |
| Kokoro, every agent busy | 79 / 93 | 30 / 62 | 32 / 56 |
| Coordinator, every agent busy | 352 / 425 | 336 / 342 | 340 / 366 |
| Lips (whole clip), every agent busy | 436 / 468 | 404 / 464 | **332 / 349** |
| Alone: voice / lips / coordinator / Kokoro | 60 / 409 / 314 / 23 | 61 / 391 / 330 / 22 | 59 / **265** / 333 / 23 |
| One message end to end (voice + lip clip) | 434 / 469 | 435 / 474 | **309 / 322** |
| Accepted voice message: clip → spoken (¹) | — | 413 / 459 | **71 / 89** |
| Lip reading after the clip ends: whole clip / **streamed** | — | — | 477 / **104** |
| Voice result after the 700 ms pause, sent then (old) | 337 / 377 | 109 / 119 | 70 / 77 |
| Voice result after the 700 ms pause, **early decode** | 3 / 22 | 2 / 2 | 2 / 3 |

(¹) Measured with an ad-hoc voice-only bench on the box (40 Kokoro-rendered phrase clips), printed, not saved to a file: 413 / 459 ms before the change (Nemotron planned every message); 106 / 120 ms with accepted readings decided by rules (commit `51df483`); 105 / 115 ms after auto-speak was limited to accepted readings (`09c44c1`); 71 / 89 ms in the final run (same run as `latency_brev_final2.json`); on `demo-final`, 73 / 84 ms with the gateway's phrase cache cold and 71 / 80 ms warm. So the later 105 → 71 ms drop is **not** from pre-rendered phrase audio (worth about 2 ms), and its cause was not isolated. The streamed-lip row is from `stream_equiv.py` in the same run (12 GRID clips, printed).

### From the laptop through the tunnel

| Measurement | Result |
|---|---|
| Round trip to the gateway | about 120 ms |
| Upload speed | about 1.75 MB/s (14 Mbps) |
| 64 KB voice clip upload | 0.23 s |
| 1.2 MB lip clip upload | **0.68 s** (the server works 34 ms of it) |
| Voice result after the pause, old way | 211 ms p50 |
| Voice result after the pause, early decode | **64 ms** p50 |
| Lip reading after the clip ends, whole clip uploaded | 898 ms p50 |
| Lip reading after the clip ends, **streamed** while mouthing | **407 ms** p50 (8/8 same reading) |
| Accepted voice message: clip sent → spoken (2026-10-18, `demo-final` baseline, 10 clips) | **252 ms** p50, 393 ms p95 |
| Lip reading after the clip ends, streamed (same baseline, 8 clips) | 415 ms p50 (whole clip 922 ms) |

### What a person waits for

Now, added up from the measured parts (not one end-to-end measurement):

| Step | Accepted voice message (spoken) | Lip message (asked) |
|---|---:|---:|
| Pause before the clip is cut | 0.7 s | 0.9 s |
| Upload and agent | 0.064 s (early decode) | 0.41 s (streamed) |
| Plan | rules, about 0 (no Nemotron) | rules at once; Nemotron refines in the background |
| Speech | preloaded phrase audio plays when the WebSocket says spoken (about 0.07 s on the box, plus one way) | — (waits for Yes/No) |
| **About** | **0.85–0.9 s** (was 1.2–1.4 s) | **about 1.3 s to the question** (was about 2.5 s) |

A lip-only reading is never auto-spoken; it waits for the person's answer.

### Decisions taken for speed, all measured

- **MPS on:** no change in arithmetic.
- **Duplicate Nemotron Speech stopped:** the asr_eval bench copy; frees about 6 GB.
- **Lip beam 6 rejected:** WER 69.4 → 71.1%, best-of-5 60.9 → 64.5%.
- **fp16 for lips not adopted:** same answers, not faster. bf16 cannot run (Chaplin's attention).
- **Early voice decode adopted:** 0 of 304 real recordings changed decision with the 350 ms tail
  (`asr_eval/results/trailing_silence_check.json`).
- **Native FP4 for the coordinator deferred:** less accurate (4-bit activations); do it only after giving vLLM
  its own GPU, and only if NeMo Gym holds.

- **Accepted readings skip Nemotron:** the rules already ask exactly what Nemotron is asked to; 413 → 105 ms
  (71–73 ms in later runs).
- **Batched DTW:** same distances (to 3e-8), 0 of 270 top-3 changes; the lip agent alone 391 → 265 ms.
- **Pre-rendered phrase audio:** the gate still decides in the gateway; edited text still goes to Kokoro.
  Measured worth about 2 ms on the box (cold 73 vs warm 71 ms); the page's preload request fails (§10).
- **Streamed lip clips:** identical readings (12/12 on the box, 8/8 from the laptop); 898 → 407 ms from the
  laptop.
- **Compact coordinator not adopted:** 3× faster, but worse on ambiguous readings (§3.3).
- **MTP not adopted:** 16% faster, same quality, but one unexplained crash under full load (§6).
- **Prefix caching:** cannot hit at this prompt size (§6).

### Open speed items

- **Early lip send** (at 450 ms of stillness, commit at 900 ms): not done.
  - The early clip has a shorter still tail than the committed one, so it is a different input.
  - Verifying "0 decisions change" needs the person's own videos, and only features are stored.
  - After streaming, at most about 0.1 s is left to hide.
- **Face-region crop before upload:** not done.
  - Streaming already hides the upload and most decoding.
  - A crop changes the frames the lip agent sees, so it needs the GRID check plus the person's own clips.
- **Network:** 120 ms round trip to the instance. Lip pieces already ride the session WebSocket. An East Coast
  instance would save about 100 ms per round trip.
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
| Safety tests (policy gate, auto-speak, phrase scoring, validator flags, check-in) | `pytest evaluations/safety_tests`: **49 tests** = 18 `test_confirmation.py` + 7 `test_auto_speak.py` + 4 `test_auto_speak_delay.py` + 3 `test_vsr_phrase_scoring.py` + 11 `test_validator_flags.py` + 6 `test_checkin.py` | — |
| Validator flags, paired A/B | `evaluations/nemo_gym/validator_ab.py [--seed 2026 --distinct-from 7] --out <file>` | `coordinator_validator_v2_paired_{eval,heldout}.json`; unpaired: `coordinator_validator_v2_{eval,heldout}_{off,on}.json` |
| Live attacks + exactness (needs the running stack) | `python evaluations/test_cases/e2e_exactness.py`: 7 attacks (5 unconfirmed-speech probes, a forged TTS token, a non-explicit method): 7/7 blocked on `demo-final`. Its exactness half (10 phrases spoken and re-transcribed) needs a free-text ASR; today's phrase-only voice agent cannot do it (EVALUATION §5) | `e2e.json` (2026-10-17, Whisper), `e2e_demo_final.json` |
| Coordinator (NeMo Gym, 72 tasks; held-out 44) | `run_local.py --policies raw:<model> guarded deterministic` (held-out: `--seed 2026 --distinct-from 7 --out <file>`; `UNSPOKEN_COORD_COMPACT=1` for compact mode) | `coordinator_brev.json` (Brev); `coordinator_heldout_brev.json`, `coordinator_heldout_compact_brev.json` (held-out); `coordinator_mtp_brev.json`, `coordinator_compact_brev.json`, `coordinator_compact_mtp_brev.json` (A/B); `coordinator.json` (Mac, 60 tasks, older prompt) |
| Lip speed vs accuracy | `evaluations/modality_baselines/vsr_speed_accuracy.py` | `vsr_speed_accuracy.json` |
| Early-decode accuracy check | `asr_eval/scripts/trailing_silence_check.py` | `trailing_silence_check.json` |
| Latency | `scripts/brev/bench_latency.py` | `latency_brev_*.json` |
| Voice model selection | `asr_eval/run_eval.py`, `scripts/loso.py` | `asr_eval/results/` |

The NeMo Gym data files (`data/*.jsonl`) embed the coordinator's prompt and tools; regenerate them with
`create_examples.py` after any coordinator change.

---

## 10. Known issues on `ui` (for the team)

1. **Auto-speak is on by default.**
   - Speech happens without the person's Yes, which contradicts the README's product rule.
   - It now covers accepted readings only. Ambiguous readings, lip-only matches and corrections wait for Yes/No
     (`test_auto_speak.py`).
   - It is scheduled immediately, so a No rarely arrives in time.
2. **Coordinator overuses `ignore_input`.** A person who failed twice, a genuine request next to an injection,
   and clear negated readings get dropped as noise (§3.3).
   - The validator allows it by default.
   - `UNSPOKEN_VALIDATOR_IGNORE_HISTORY=1` fixes the failed-twice and negated cases (eval 0.954 → 0.995), but not
     the injection case.
   - Off pending the team's decision.
3. **The correction validator does not check negations** by default; `UNSPOKEN_VALIDATOR_NEGATION=1` adds the
   check (no measured change: the model never dropped a negation in these runs).
4. **Personal lip data in git, pending a consent decision.** `data/enroll/` holds encoder features of a real
   person's recordings, force-added past the `/data/` gitignore rule, while `enroll.py` says they "are not
   uploaded". History is not rewritten; the note and the removal commands are in
   [DATASETS_AND_CONSENT.md](DATASETS_AND_CONSENT.md).
5. **Disclosure: fixed on `demo-final`.** With a remote gateway the camera chip says "Camera · clips to NVIDIA
   Brev" and calibration says clips are processed on that GPU server, not stored; local mode keeps "on device".
6. **Enrollment threshold barely measured.** It was set from 10 recordings, and strangers matched in about 12%
   of GRID clips.
7. **`zero_infinity=True`: fixed on `demo-final`.** Impossible CTC alignments in the phrase-list path now score
   `IMPOSSIBLE` (−1e6, asr_eval's convention) and never rank first (`test_vsr_phrase_scoring.py`).
8. **Camera stalls.** The camera-open sharing and single tracking loop were reverted (slow USB cameras can
   stall).
9. **`sync_app.sh` does not ship `data/enroll/`** (nor `data/vision/`). They must be copied separately (§6).
10. **Smaller items:**
    - the Mac path needs NeMo for the voice agent;
    - `web/README.md` was reset to the Vite template on `ui` (restored by this doc update);
    - 8 lint warnings.
11. **Phrase-audio preload request fails** (found on `demo-final`, not fixed: the preload is frozen for this
    branch).
    - `POST /api/{sid}/phrase_audio` renders and caches the phrases, then logs with `actor="gateway"`, which
      `AuditEvent` rejects, so it answers HTTP 500.
    - The page never gets the list and downloads each phrase's audio when spoken. The gateway-side cache works.
    - The fix is one word (`actor="system"`) and needs approval.