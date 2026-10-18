# Handoff: glass UI, the whole backend on Brev, Nemotron voice, MobileGaze

Branch `ui`, 2026-10-18. It builds on `asr-eval` (the voice model selection, summarised at the end).

## TL;DR

- **The app is one liquid-glass page** (`web/src/glass/`) at http://localhost:8800, following the design
  `unspoken_liquid_glass_3.html` without its bottom dock.
  - Every input runs at once: voice, lips, gaze, blink, head nod/shake.
  - Readings appear as faint ink and bubbles; nothing is spoken until the person confirms.
  - Top right: live agent activity, then history.
  - Eye calibration has its own plain black screen.
- **The whole backend runs on the Brev GPU** (`scripts/brev/`). The laptop runs only the UI and reaches the gateway
  through an SSH tunnel. All six services plus vLLM run on one RTX PRO 6000.
- **Voice agent = Nemotron Speech Streaming en 0.6B** with forced phrase scoring. It uses the evaluated
  decision code and frozen thresholds (`asr_eval/harness/live.py`). Whisper and RNNoise are gone.
- **Gaze = MediaPipe iris + MobileGaze** (from `yash-gaze-v3`) with per-user ridge calibration. WebGazer is gone.
- **Budget 31.01B of 33B.** The coordinator is counted at its nominal 30B; with its weight-file count of 31.58B the
  total is 32.59B, also within budget.
- **Latency on Brev:** one message end to end in 0.97 s p50 with NVIDIA MPS; the voice result is ready 2 ms after
  the end-of-utterance pause (early decode). Numbers and decisions: [docs/SERVING.md](docs/SERVING.md).

## What was built (branch `ui`)

| Piece | Where | Notes |
|---|---|---|
| Glass UI | `web/src/glass/` ([web/README.md](web/README.md)) | Ink engine, agent nucleus, bubbles, gaze dwell ported from the design; React controller wired to the gateway; old views removed |
| Always-on sensors | `web/src/glass/sensors.ts`, `web/src/lib/face.ts` | Voice VAD → 16 kHz WAV clips; mouth-motion → webm clips; nod/shake; one face-tracking loop; capture pauses while a message waits or speech plays |
| Calibration | `web/src/glass/Calibration.tsx` | Black screen; 18 dots in two passes + 5 unseen test dots; saved calibration; live status; measured failure reason |
| Gaze engine | `web/src/lib/gaze.ts`, `ridge.ts`, `appearance.ts` | From `yash-gaze-v3`; one TypeScript fix (constructor parameter properties are not allowed by this tsconfig); frame counters added (`gazeDebug`) |
| CUDA port | `services/{audio,tts,vsr}`, `shared/service.py` | `pick_device` (cuda > mps > cpu); Kokoro via `kokoro` on CUDA; ASR/TTS backend chosen by platform |
| Voice agent | `services/audio/app.py`, `asr_eval/harness/live.py` | Candidate → proposed (one phrase); unknown → ambiguous (top 3); no speech / no words → unknown |
| Brev scripts | `scripts/brev/` | `setup_app.sh` (uv venvs, torch cu128, vLLM), `start_app.sh` (start / status / stop), `sync_app.sh`, `connect.sh`, `bench_latency.py` |
| Ledger | `scripts/verify_weights.py`, `evaluations/results/weights.json`, `shared/provenance.py` | Nemotron Speech + Silero VAD, MobileGaze; coordinator at the nominal 30B |
| Docs | README, ARCHITECTURE, BREV (§4b, §5), SERVING (new), EVALUATION (§6, §7), MODEL_CARDS, DATASETS_AND_CONSENT, web/README, asr_eval/README | |

## How to run

```bash
# laptop (Windows: brev runs inside WSL Ubuntu)
bash scripts/brev/sync_app.sh marginal-sapphire-whale                          # after every commit
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh'   # restart; `status` / `stop`
bash scripts/brev/connect.sh marginal-sapphire-whale                           # tunnel the gateway to localhost:8000
cd web && npm run dev                                                          # http://localhost:8800
```

- **First time on a new instance:** run `scripts/brev/setup_app.sh` once (about 15–30 min). The voice agent also
  needs the NeMo venv from `asr_eval/scripts/setup_brev.sh`.
- **On a Mac:** `./scripts/run_all.sh` still works, but the voice agent now needs NeMo in the backend venv
  (not wired or tested yet; see open items).

## Where things are

| Item | Location |
|---|---|
| Code | branch `ui` on `origin` |
| Backend on Brev | `~/unspoken` on `marginal-sapphire-whale`; logs in `~/unspoken/.run/logs/`; pinned versions in `~/unspoken/.run/requirements-*.lock` |
| Venvs on Brev | `/workspace/venv-app` (agents), `/workspace/venv-vllm` (vLLM), `/workspace/venv-asr` (NeMo: voice agent + asr_eval) |
| Latency results | `evaluations/results/latency_brev.json`, [docs/SERVING.md](docs/SERVING.md) |
| Browser model files | `web/public/mediapipe/`, `web/public/models/mobileone_s0_gaze.onnx` (gitignored; `setup_models.sh` fetches them) |

## Open items

1. **Gaze calibration with a real webcam is not verified.** Two causes found and fixed:
   - Parallel camera requests stalled on a slow USB camera (12 s first open), so calibration hung on "Starting camera".
   - A camera held by another app or tab was reported as a permission problem.

   Still to do: run a full calibration with a person in front of the camera and read the reason line, or the
   `window.__gaze` counters, if it fails.
2. **Negations.** "I don't need water" is correctly *not* accepted, but as specified (unknown → offer the top 3) the
   offered list can include "I need water". Decide whether a negation mismatch should offer nothing instead.
3. **Mac path for the voice agent.** It needs NeMo; `run_all.sh` should start `services/audio` from a NeMo venv
   (as `start_app.sh` does on Brev).
4. **Serve faster.** Done: NVIDIA MPS (on by default), the duplicate Nemotron Speech stopped, early voice decode.
   Tested and rejected: beam 6 for lip reading (accuracy drop), fp16 (no gain). Remaining: GPU isolation for vLLM
   (2 GPUs or MIG), to do last; native FP4 for the coordinator is not recommended before that (it is the less
   accurate option and the coordinator is not the bottleneck). Details in [docs/SERVING.md](docs/SERVING.md).
5. **Not ported from `yash-gaze-v3`:** the Gaze cards page (card snap, gaze lock), the look-then-blink setting,
   the head-assist settings UI, and the diagnostics export button.
6. **The phrase list.** The voice agent knows 20 phrases (`asr_eval/configs/eval.yaml`); the rail's Phrases
   bubbles show 6 of their own. Unify them into one list.

## Voice model selection (from `asr-eval`, still valid)

- Nemotron Speech Streaming en 0.6B with forced RNNT phrase scoring (`forced_llr`).
  - 3 recorded speakers, leave-one-speaker-out: 71.5% accepted correctly, 0.4% wrong, 90.2% resolved within one
    gaze question, 0/45 near-miss false accepts, 43 ms per clip.
- **Not adopted:** Parakeet-CTC (3–5× more wrong and false accepts), Whisper (hallucinates on silence, lower
  one-question resolution), multilingual Nemotron 3.5 (less accurate and slower on English).
- Thresholds: `asr_eval/results/thresholds.json`. The app uses the balanced point because every candidate is
  confirmed; strict is the headline.
- **Next for the data:** more noisy/muffled/whisper recordings and negatives from p02/p03 and new speakers, then
  `run_stage.sh all`. Full details: `asr_eval/README.md`, [docs/EVALUATION.md](docs/EVALUATION.md) §6.

## Housekeeping (please do)

- **Billing.** `marginal-sapphire-whale` bills $2.63/hr until deleted; it is not stoppable.
  - `start_app.sh stop` frees the GPU but does not stop billing.
  - Before deleting, back up the recordings (`brev copy marginal-sapphire-whale:/workspace/asr_eval_data/own/ asr_eval/data/own/`).
- **Rotate the Brev API key** that was pasted into an earlier chat session (it is not in any file or commit).
- **Recordings are personal voice data.** Keep them out of git, and delete both copies (Brev and laptop) after the hackathon.
- **Privacy.** With the backend on Brev, mouth and voice clips leave the laptop ([docs/DATASETS_AND_CONSENT.md](docs/DATASETS_AND_CONSENT.md)).
  The UI's camera chip says so; tell the person and get their OK first.
