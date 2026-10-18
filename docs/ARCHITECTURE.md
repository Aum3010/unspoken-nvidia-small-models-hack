# Architecture

```text
 Browser (web/, React + Vite)                         ── all camera/mic frames stay on this Mac ──
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MediaPipe Face Landmarker (wasm/GPU): mouth ROI overlay, face-quality hints, deliberate blink  │
 │ Gaze: MediaPipe iris + MobileGaze (ONNX Web) → ridge calibration → dwell or look+blink         │
 │ Gaze cards: fixed 3×3 cards, calibrated on the cards, snap to nearest + 250 ms gaze lock       │
 │ My phrases: record own phrases for lip-reading enrollment (sent to /api/enroll)                │
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
 │ Enrollment proxy: /api/enroll GET/POST/DELETE → vsr /enroll (features only)                   │
 └───┬──────────────┬──────────────┬──────────────┬──────────────────────┬──────────────────────┘
     ▼              ▼              ▼              ▼                      ▼
  vsr :8101     audio :8102     eeg :8103     coordinator :8105       tts :8104
  Auto-AVSR     Whisper-base    P300 replay   Nemotron 3.5 Lightning  Kokoro-82M
  (Chaplin)     + RNNoise       xDAWN+Riemann (Ollama/MLX/vLLM)       (MLX)
  torch MPS     MLX Metal       BNCI2014-008  typed tools + validator verifies token
  n-best raw    raw ∥ enhanced  recorded only deterministic fallback  before speaking
  + enrollment
  + phrase list
```

## Agents and what each one is allowed to do

| Agent | Input | Output | May not |
|---|---|---|---|
| Lip-reading (vsr) | mouth video clip; the person's enrolled phrase features, if any | one candidate list from three modes sharing one encoder pass: free-text n-best with beam scores, closest enrolled phrases (DTW distance), optional fixed phrase list (log-likelihood); face/mouth-motion quality | rewrite or "correct" text (Chaplin's LLM correction is deliberately not used); store video (enrollment keeps encoder features only, on this machine) |
| Voice (audio) | microphone clip | raw-audio and enhanced-audio transcripts with word probabilities and a word diff | merge the two, or emit words when no voice activity is detected |
| EEG replay (eeg) | recorded BNCI2014-008 trials | per-flash P300 scores, decoded row/column/letter | read ground-truth labels before predicting; claim live brain input |
| Gaze / blink (browser) | webcam (MediaPipe iris + MobileGaze eye-image model, per-person ridge calibration) | selection events with method + dwell | count as confirmation unless the person dwells, or holds a look and blinks deliberately, on the separate confirm target |
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
- Nothing enrolled in **My phrases** → lip reading uses free text (and the phrase list, if configured) only.

## Known integration gap: three phrase lists

Voice (`asr_eval/configs/eval.yaml`, 20 phrases), the board and Gaze cards (`web/src/components/Board.tsx`) and lip reading (`services/vsr/patient_phrases.txt`, 44 phrases) each keep their own list, with different wording ("I'm in pain" vs "I am in pain"). Cross-modal fusion compares normalised text, so two modalities agreeing on the same request can be shown as a disagreement. The planned fix is one shared list keyed by phrase id ([HANDOFF.md](../HANDOFF.md), step 3).

## Offline evaluation harness: choosing the voice agent's ASR (`asr_eval/`)

`asr_eval/` is separate from the running app. It answers one question before the voice agent changes: **which ASR model, gated how, can propose one of 20 fixed phrases without ever proposing one on silence?**

```text
manifest.csv ─► Silero VAD ─► model (own process) ─► rankers ─► decide() ─► metrics ─► summary.md
 own / TORGO /   speech s @    transcript +           text (CMU       no_speech / unknown /
 LibriSpeech /   thr 0.2-0.6   forced phrase          phonetic),      candidate, thresholds
 noise / TTS /                 log-probs (CTC/RNNT)   forced_sum/     tuned on calib only
 degradations                                         norm/llr
```

- **Model interface** (`harness/models.py`): `load()`, `transcribe(wav)`, `score_phrases(wav, texts)`, `info()`.
  - Forced scoring reuses the transcription's encoder pass: `ctc_loss` for CTC models, NeMo's pure-PyTorch RNNT loss for transducers.
  - CTC alignments that can't exist keep an impossible score; `zero_infinity` would turn them into perfect ones.
- **Decision** (`harness/decide.py`): the same gates for every model (VAD, absolute score + margin, negation mismatch, empty transcript → `unknown`). The best configuration is chosen on calib.
- **Where it runs:** on a Brev GPU via `run_eval.py` (see [BREV.md](BREV.md)), or on CPU with `configs/local_smoke.yaml`.
- **Result:** Nemotron Speech Streaming en 0.6B with `forced_llr` ([EVALUATION.md](EVALUATION.md) §6).
- **What happens next:** it will replace Whisper base + text candidates in `services/audio/`. It will still emit typed `Evidence`, so the gateway, coordinator and policy gate don't change. The patient's confirmation stays the final say.

## Ports & processes

`scripts/run_all.sh` starts gateway 8000, vsr 8101, audio 8102, eeg 8103, tts 8104, coordinator 8105 and the web app on 5317. Logs go to `.run/logs/`. Each service exposes `/health`; the gateway aggregates them at `/api/health`.
