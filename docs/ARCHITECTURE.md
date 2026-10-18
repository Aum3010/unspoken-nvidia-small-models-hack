# Architecture

```text
 Browser (web/, React + Vite)                  ── camera/mic frames stay in the browser; clips go to the gateway ──
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Glass room (web/src/glass): ink + agent-activity canvases, rail, bubbles, black calibration    │
 │ MediaPipe Face Landmarker (wasm/GPU), one loop: mouth motion, deliberate blink, nod / shake    │
 │ Gaze: MediaPipe iris + MobileGaze (ONNX Runtime Web) → per-user ridge → dwell (error-scaled)   │
 │ Always-on capture: voice VAD → 16 kHz WAV clip · mouth motion → webm clip (paused while a      │
 │   message waits for confirmation or speech plays) · click · keyboard → audited method          │
 └──────────────┬─────────────────────────────────────────────────────────────▲──────────────────┘
                │ REST (clips, selections, confirm)                           │ WebSocket: state +
                ▼                                                             │ agent timeline
 ┌──────────────────────────── gateway :8000 (FastAPI) ──────────────────────────────────────────┐
 │ Session state machine (services/gateway/state.py) — deterministic                             │
 │   collecting → proposed/ambiguous → clarifying → selected → confirmed → spoken                │
 │ Cross-modal fusion: identical text → one candidate with supporting modalities;                │
 │   different text → separate candidates, status=ambiguous (never merged)                       │
 │ Policy gate (shared/confirmation.py): HMAC token only for exact confirmed text                │
 │ Audit log: raw hypotheses, tool calls, selections, edits, confirmations, speech               │
 └───┬──────────────┬──────────────┬──────────────┬──────────────────────┬──────────────────────┘
     ▼              ▼              ▼              ▼                      ▼
  vsr :8101     audio :8102     eeg :8103     coordinator :8105       tts :8104
  Auto-AVSR     Nemotron Speech P300 replay   Nemotron 3.5 Lightning  Kokoro-82M
  (Chaplin)     0.6B + Silero   xDAWN+Riemann (Ollama/MLX/vLLM)       (MLX/torch)
  torch MPS/    forced_llr over BNCI2014-008  typed tools + validator verifies token
  CUDA, n-best  20 phrases      recorded only deterministic fallback  before speaking
```

## Agents and what each one is allowed to do

| Agent | Input | Output | May not |
|---|---|---|---|
| Lip-reading (vsr) | mouth video clip | up to 3 of the person's enrolled phrases (DTW distance ≤ 0.30 over encoder features), or nothing; free-text n-best only with `UNSPOKEN_VSR_FREE_TEXT=1`; face/mouth-motion quality | rewrite or "correct" text (Chaplin's LLM correction is deliberately not used) |
| Voice (audio) | microphone clip | one of 20 fixed phrases with its raw `forced_llr` score (confident accept), the top 3 (uncertain), or nothing (no speech / no words heard); transcript and decision reason in `quality` | propose free text, or a phrase when Silero finds no speech or the model heard no words |
| EEG replay (eeg) | recorded BNCI2014-008 trials | per-flash P300 scores, decoded row/column/letter | read ground-truth labels before predicting; claim live brain input |
| Gaze / blink / head (browser) | webcam | selection events with method + dwell; nod / shake | count as confirmation unless the person dwells on a confirm target (1.5× longer), blinks, or nods while a message waits |
| Coordinator | structured evidence summary (no media) | exactly one typed tool call: `ask_confirmation`, `request_retake`, `switch_input_method`, `propose_correction` (fix misread words, validated to stay anchored to the reading), `ignore_input` (treat as noise), (`speak_confirmed_text`, always rejected without a confirmation event) | invent a new message: a correction must keep at least half the words read and the length within ±2 words (negations are not checked: [TECHNICAL_VERBOSE](TECHNICAL_VERBOSE.md) §10) |
| Policy gate (code) | message + audit log | TTS token or `PolicyViolation` | — |
| TTS | token + text | audio of exactly that text | speak anything without a matching token |

## Coordination timeline for one message

1. A specialist returns `Evidence` (shared/schemas.py). Evidence the specialist marked unusable (no face, no mouth motion, too short, no speech, no reading) is logged and dropped, never shown. Otherwise the gateway adds candidates (a new lip/audio clip replaces an unanswered reading unless it pairs with it within 10 s), records raw hypotheses in the audit log, and runs the **deterministic policy instantly**, so options appear at once.
2. The coordinator model is called in the background with a summary. Its tool call is validated outside the model (ids exist, method available, gaze calibrated, no speech). A valid call refines what is shown, unless the person has already acted. An invalid call is logged as rejected and the fallback stays.
3. **Confirm mode** (`UNSPOKEN_AUTO_SPEAK=0`): the person confirms with Yes, a blink or a nod (a phrase card is its own confirmation). **Demo auto-speak** (`UNSPOKEN_AUTO_SPEAK=1`, the default on `ui`): only for an **accepted** reading (the voice agent accepted it, or lips and voice agree), the rules' plan is applied at once, without Nemotron, and the gateway confirms that reading itself, audited as `auto_confirmed` by `system`. Ambiguous readings, lip-only matches, disagreements and corrections go to Nemotron and then wait for Yes/No.
4. `authorize_speech` checks status, the explicit method, that the confirmation event is in the audit log and was made by a person, and that the text has not changed since confirmation (in auto-speak: its own audited `auto_confirmed` event for the exact text). It then signs `(message_id, text)`.
5. The TTS service verifies the signature and speaks the exact string.

## Graceful degradation

- Any specialist down → its capture returns 503, the timeline shows it, and the phrase/spelling boards keep working.
- Coordinator model down or slow → the deterministic policy is used. It is also the evaluation baseline.
- TTS down → the browser's speech synthesis speaks the same confirmed string.
- Gaze calibration poor (≥220 px mean error) → gaze selection stays off; click, hover-dwell and blink-scan remain available.

## Offline evaluation harness: choosing the voice agent's ASR (`asr_eval/`)

`asr_eval/` is separate from the running app. It answers one question before the voice agent changes: **which ASR model, gated how, can propose one of 20 fixed phrases without ever proposing one on silence?**

```text
manifest.csv ─► Silero VAD ─► model (own process) ─► rankers ─► decide() ─► metrics ─► summary.md
 own / TORGO /   speech s @    transcript +           text (CMU       no_speech / unknown /
 LibriSpeech /   thr 0.2-0.6   forced phrase          phonetic),      candidate, thresholds
 noise / TTS /                 log-probs (CTC)        ctc_sum/norm/   tuned on calib only
 degradations                                         llr
```

- **Model interface** (`harness/models.py`): `load()`, `transcribe(wav)`, `score_phrases(wav, texts)`, `info()`.
  - Forced scoring reuses the transcription's encoder pass.
  - CTC alignments that can't exist keep an impossible score; `zero_infinity` would turn them into perfect ones.
- **Decision** (`harness/decide.py`): the same three gates for every model (VAD, absolute score + margin, negation mismatch). The best configuration is chosen on calib.
- **Where it runs:** on a Brev GPU via `run_eval.py` (see [BREV.md](BREV.md)), or on CPU with `configs/local_smoke.yaml`.
- **In the app:** the winning configuration (Nemotron en 0.6B + forced_llr, balanced point) is the voice agent. `services/audio/` imports `harness/live.py`, the same recognizer `serve.py` uses, so the app decides exactly as evaluated. It still emits typed `Evidence` (candidate → proposed; unknown → ambiguous with the top 3; no speech → unknown), so the gateway, coordinator and policy gate are unchanged. The patient's confirmation stays the final say.

## Ports & processes

`scripts/run_all.sh` starts gateway 8000, vsr 8101, audio 8102, eeg 8103, tts 8104, coordinator 8105 and the web app on 8800. Logs go to `.run/logs/`. Each service exposes `/health`; the gateway aggregates them at `/api/health`.

## Two deployments

```text
 Mac (scripts/run_all.sh)                         Brev GPU (scripts/brev/start_app.sh)
 browser :8800 → gateway :8000 → agents (MPS/MLX)  browser :8800 (laptop) ══ SSH tunnel ══► gateway :8000 (Brev)
                 coordinator → Ollama :11434                    agents on CUDA, all on 127.0.0.1:
                                                                vsr 8101 · audio 8102 (NeMo venv) · eeg 8103 · tts 8104
                                                                coordinator 8105 → vLLM 8001 (Nemotron NVFP4)
```

- **Brev:** one RTX PRO 6000 holds everything (about 63 GB peak), with NVIDIA MPS so the GPU processes run concurrently, and early voice decode (the clip is sent at 350 ms of silence and committed at 700 ms). The gateway reports `host: "NVIDIA Brev"` in `/api/health`; the UI on `ui` does not currently show it.
- **Every detail** of the system and its serving: [TECHNICAL_VERBOSE.md](TECHNICAL_VERBOSE.md).
- **Same code on both.** Device selection goes through `shared.service.pick_device` (cuda > mps > cpu). The ASR and TTS backends are picked by platform: MLX on a Mac, PyTorch/NeMo on CUDA.
- **Latency** with every model running together, and how to serve it faster: [SERVING.md](SERVING.md).

## Optional behaviour on `demo-final` (flags, all off by default)

| Flag | Effect | Where |
|---|---|---|
| `UNSPOKEN_VALIDATOR_NEGATION` | the coordinator validator rejects a correction that adds or removes a negation | `services/coordinator/app.py` |
| `UNSPOKEN_VALIDATOR_IGNORE_HISTORY` | the validator rejects `ignore_input` after two failed tries, a recent "none of these", or on an accepted clean reading | `services/coordinator/app.py` |
| `UNSPOKEN_AUTO_SPEAK_DELAY_MS` | a cancellable countdown before an accepted reading is auto-spoken | `services/gateway/app.py`, `GlassApp.tsx` |
| `UNSPOKEN_CHECKIN` | vision preview: a proactive check-in question on recorded MIMIC-IV demo data. The answer is logged for the nurse and never spoken | `services/gateway/checkin.py`, `VisionPreview.tsx` |

Measured effects and the full list of flags: [TECHNICAL_VERBOSE.md](TECHNICAL_VERBOSE.md) §2–§4 and
[EVALUATION.md](EVALUATION.md) §4. Open issues: TECHNICAL_VERBOSE §10.
