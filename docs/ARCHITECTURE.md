# Architecture

```text
 Browser (web/, React + Vite)                  ── camera/mic frames stay in the browser; clips go to the gateway ──
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MediaPipe Face Landmarker (wasm/GPU): mouth ROI overlay, face-quality hints, deliberate blink  │
 │ Gaze: MediaPipe iris + MobileGaze (ONNX) → per-user ridge → dwell (error-scaled slack, confirm) │
 │ Selection engine: click · keyboard · hover dwell · gaze dwell · blink scan → audited method    │
 │ MediaRecorder: silent mouth clip (webm) · voice clip (webm/opus)                               │
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
| Lip-reading (vsr) | mouth video clip | n-best raw transcripts, beam scores, face/mouth-motion quality | rewrite or "correct" text (Chaplin's LLM correction is deliberately not used) |
| Voice (audio) | microphone clip | raw-audio and enhanced-audio transcripts with word probabilities and a word diff | merge the two, or emit words when no voice activity is detected |
| EEG replay (eeg) | recorded BNCI2014-008 trials | per-flash P300 scores, decoded row/column/letter | read ground-truth labels before predicting; claim live brain input |
| Gaze / blink (browser) | webcam | selection events with method + dwell | count as confirmation unless the person dwells on the separate confirm target |
| Coordinator | structured evidence summary (no media) | exactly one typed tool call: `ask_confirmation`, `request_retake`, `switch_input_method`, (`speak_confirmed_text`, always rejected without a confirmation event) | write, rephrase or choose the patient's message |
| Policy gate (code) | message + audit log | TTS token or `PolicyViolation` | — |
| TTS | token + text | audio of exactly that text | speak anything without a matching token |

## Coordination timeline for one message

1. A specialist returns `Evidence` (shared/schemas.py). The gateway adds candidates, records raw hypotheses in the audit log, and runs the **deterministic policy instantly**, so options appear at once.
2. The coordinator model is called in the background with a summary. Its tool call is validated outside the model (ids exist, method available, gaze calibrated, no speech). A valid call refines what is shown, unless the person has already acted. An invalid call is logged as rejected and the fallback stays.
3. The person selects (click, keyboard, dwell, blink), optionally corrects the text (the original is kept), then confirms with a separate action. Confirm targets need a longer dwell than select targets.
4. `authorize_speech` checks status, the explicit method, that the confirmation event is in the audit log and was made by a person, and that the text has not changed since confirmation. It then signs `(message_id, text)`.
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
