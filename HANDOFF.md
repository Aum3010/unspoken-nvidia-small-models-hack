# Handoff: voice agent model selection on NVIDIA Brev

Branch `asr-eval`, 2026-10-17. This covers the work to choose and validate the **voice (residual speech) agent's ASR**. The rest of Unspoken (gateway, coordinator, lips, gaze, EEG, TTS) is unchanged on this branch.

## TL;DR

- **Decision:** use **Nemotron Speech Streaming en 0.6B** (`nvidia/nemotron-speech-streaming-en-0.6b`) with **forced RNNT phrase scoring** (`forced_llr`). It replaces Whisper base + transcript matching for the 20-phrase task.
- **Numbers.** On 3 recorded speakers, each tested by thresholds tuned on the other two (strict):
  - the right phrase was accepted **71.5%** of the time, with **0.4%** wrong;
  - **90.2%** were resolved within one gaze question;
  - **0/45** near-miss false accepts;
  - **43 ms** median per clip on the GPU.
- **Everything is reproducible** with one command (`asr_eval/run_eval.py`), on CPU locally or on a Brev GPU.
- **The live test UI is running on Brev.** Open it on a laptop through `brev port-forward`.
- **Not done yet:** wiring the chosen model into `services/audio/` (see *Next steps*).

## What was built

| Piece | Where | What it is |
|---|---|---|
| Evaluation harness | `asr_eval/` ([README](asr_eval/README.md)) | Manifest → Silero VAD gate → model (own process) → rankers → `decide()` → metrics/plots. Data: own recordings, TORGO (dysarthric/control), LibriSpeech, MUSAN/Speech Commands/ESC-50, generated silence, Kokoro TTS (sanity only), degradation grid. Calib/test split by speaker |
| Model backends | `asr_eval/harness/models.py` | Parakeet-CTC (transformers, `ctc_loss` scoring), NeMo transducers (Parakeet-TDT, Nemotron en, Nemotron 3.5 multilingual; RNNT scoring), faster-whisper |
| Speaker rotation | `asr_eval/scripts/loso.py` | Every recorded speaker tested by thresholds tuned without them; reports the "resolved within one gaze question" view |
| Live test bench | `asr_eval/serve.py`, `asr_eval/static/index.html` | Push-to-talk page. **Record** tab (guided, auto-save, redo/skip/resume) and **Test** tab (decision, gate, top 3, timings); sensitivity and language selectors |
| Brev tooling | `asr_eval/scripts/setup_brev.sh`, `brev_prepare.sh`, `scripts/brev/*` | Environment (CUDA 12.8 torch, `requirements.lock`), dataset download, sync/run/status/wait/fetch helpers, CLI install for WSL |
| Docs | [docs/BREV.md](docs/BREV.md), [docs/EVALUATION.md](docs/EVALUATION.md) §6, [docs/MODEL_CARDS_AND_LICENSES.md](docs/MODEL_CARDS_AND_LICENSES.md), [docs/DATASETS_AND_CONSENT.md](docs/DATASETS_AND_CONSENT.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How to use Brev; protocol and results; model licences and status; datasets and recording consent; where the harness fits |
| Porting skill | `.claude/skills/apple-silicon-to-cuda/SKILL.md` | How to move Apple Silicon (MPS/MLX) code to CUDA, with the pitfalls found here |

## Results and decisions (details in `asr_eval/results/`)

| Question | Answer | Evidence |
|---|---|---|
| Which ASR? | Nemotron en 0.6B + forced RNNT scoring (`forced_llr`) | `results/loso.md` |
| Why not Parakeet-CTC (highest raw accept rate)? | 3–5× more wrong and false accepts | `results/loso.md` |
| Why not Whisper? | Lower one-question resolution; Whisper hallucinates on silence ("you", "Thank you."), so the VAD gate is mandatory anyway. large-v2 is over budget | `results/summary.md` |
| Multilingual Nemotron 3.5? | **Not adopted.** On English speakers: 43.8% accepted vs 71.5%, dysarthric WER 39% vs 28%, p95 263 vs 94 ms. Language selector and draft translations are kept for later | `docs/MODEL_CARDS_AND_LICENSES.md` |
| Two models combined? | No. The coordinator already does cross-modal agreement; a second ASR adds latency and breaks the 33B budget | README / plan.md |
| Fine-tune weights? | Not with 3 speakers; revisit as per-patient enrollment with more data | — |
| Operating point | Strict (≤1% false accepts per negative type) for reported numbers; balanced (≤5%) as the live default, since every candidate is confirmed by gaze | `results/thresholds.json` |

Safety rules in `decide()`: VAD gate; minimum score and margin; negation mismatch → `unknown`; **empty transcript → `unknown`** ("never infer words absent from all usable input", plan.md).

## Where things are

| Item | Location |
|---|---|
| Code + committed results | branch `asr-eval` on `origin` |
| Recordings (3 speakers, 307 clips, 17 MB) | Brev `/workspace/asr_eval_data/own/`; laptop backup `asr_eval/data/own/` (gitignored, **never commit**) |
| Per-clip outputs (contain transcripts) | Brev only, `~/asr_eval/results/raw/` (`fetch_results.sh --raw` to copy) |
| Datasets, HF cache, venv | Brev `/workspace/asr_eval_data`, `/workspace/hf`, `/workspace/venv-asr` |
| Brev instance | `marginal-sapphire-whale`, RTX PRO 6000 96 GB, $2.63/hr, **not stoppable**, **left running** at handoff |
| Live bench | Brev 127.0.0.1:8800 → laptop `http://localhost:8800` via `brev port-forward marginal-sapphire-whale --port 8800:8800` |

## How to resume

```bash
# laptop (Windows: inside WSL Ubuntu; CLI install: asr_eval/scripts/brev/install_cli_wsl.sh)
brev login                                            # use your own key; never paste keys into chats or files
brev ls

# ship committed code, run everything (+ speaker rotation), bring results home
bash asr_eval/scripts/brev/sync.sh marginal-sapphire-whale
brev exec marginal-sapphire-whale 'bash ~/run_stage.sh all'
brev exec marginal-sapphire-whale 'bash ~/wait_for.sh ~/asr_eval/run.log "[s]cripts/loso.py" 90'
bash asr_eval/scripts/brev/fetch_results.sh marginal-sapphire-whale
brev copy marginal-sapphire-whale:/workspace/asr_eval_data/own/ asr_eval/data/own/   # back up recordings

# live bench
brev exec marginal-sapphire-whale 'bash ~/serve_ui.sh nemotron-streaming-0.6b forced_llr balanced'
brev port-forward marginal-sapphire-whale --port 8800:8800                            # open http://localhost:8800

# laptop-only plumbing test (CPU, no GPU cost)
cd asr_eval && python run_eval.py --config configs/local_smoke.yaml --limit 6
```

On a fresh instance, run `scripts/brev_prepare.sh` once first (about 20–40 min; MUSAN is slow from OpenSLR, and the ELDA mirror is tried first). Troubleshooting: [docs/BREV.md](docs/BREV.md) §5.

## Next steps (in priority order)

1. **Record the hard conditions.** Only p01 has noisy, muffled and whispered clips, which is where the losses are. Have p02 and p03 (and new speakers) record those in the Record tab, plus more negatives (only 19 silence/noise and 7 unrelated so far). p03 should lower the mic gain, since most of their clips clipped. Then `run_stage.sh all`.
2. **Integrate into `services/audio/`.** Replace mlx-whisper with the Nemotron backend from `asr_eval/harness/models.py` and the decision code in `harness/decide.py` + `harness/pipeline.py`:
   - **Emit `Evidence`** (shared/schemas.py) with the top-3 phrase hypotheses and their raw `forced_llr` scores (not probabilities).
   - **Map decisions to status:** candidate → `proposed`; unknown → `ambiguous`, so the coordinator offers the top 3 for gaze; no speech → `unknown` + `no_speech_detected` flag; empty transcript → `unknown`.
   - **Thresholds:** load them from `asr_eval/results/thresholds.json` (`nemotron-streaming-0.6b/forced_llr|phrases20`, strict or `|balanced`).
   - **Phrase list:** the 20-phrase list lives in `asr_eval/configs/eval.yaml`. Unify it with the app's board (`web/src/components/Board.tsx`, 24 + yes/no) as one shared list.
3. **GPU co-location with the coordinator.**
   - The voice model uses about 5–6 GB on the device.
   - Lower vLLM's `--gpu-memory-utilization` in `scripts/serve_vllm_brev.sh` (currently 0.90) before sharing the GPU.
   - Re-measure p95 latency under load.
4. **Budget ledger.** Nemotron en (0.618B) replaces Whisper base (0.072B), for a total of about 32.59B of 33B.
   - Update `scripts/verify_weights.py`.
   - It must count NVFP4 packed weights as 2 per byte, or the coordinator gets undercounted by about 2×.
5. **Multilingual, when needed.**
   - Native-speaker review of the drafts in `configs/eval.yaml` → `languages`.
   - Native recordings (the Record tab is English-only for now).
   - Run `scripts/multilingual_sanity.py`.
   - Deploy Nemotron 3.5 *instead of* the English model at sites that need it, since both together exceed the budget.
6. **Optional:**
   - forced TDT scoring for Parakeet-TDT, which needs libNVVM for NeMo's Numba kernels;
   - real streaming endpointing (Nemotron's cache-aware chunks), which isn't built since capture is push-to-talk.

## Known limits

- **Small data:** 3 speakers, 235 real phrase clips, 71 real negatives. Gaps of 2–3 points are within noise, and negative-set confidence intervals are wide.
- **English only:** all recordings are English, so non-English behaviour is untested.
- **TORGO words are a surrogate task:** command words from TORGO prompts. Use it for dysarthric-vs-control trends, not absolute accuracy.
- **Synthetic noise is not like your room:** degraded copies help, but real noisy/muffled/whisper recordings performed much worse than the synthetic degradations.

## Housekeeping (please do)

- **Rotate the Brev API key.** It was pasted into a chat session during this work. It is not in any file or commit (checked before push).
- **Delete the instance when done** (`brev delete marginal-sapphire-whale`), *after* backing up recordings and results. It bills $2.63/hr until deleted, about $16 by handoff.
- **Recordings are personal voice data.** Keep them out of git, and delete both copies (Brev and laptop) after the hackathon, as stated in the consent text.
