# Models, weights and licences

Parameter counts below come from `scripts/verify_weights.py`, which reads the weight files on disk. Quantised tensors are counted at their logical (unpacked) size, and all MoE experts are counted. Full output is in `evaluations/results/weights.json`.

| Role | Model / checkpoint | Params (counted) | Runtime | Licence |
|---|---|---:|---|---|
| Coordinator (all logical agent roles share it) | NVIDIA Nemotron 3.5 Lightning 30B-A3B — Ollama `nemotron-3.5-lightning:30b-a3b` / `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` | 31.578 B | Ollama (Metal) · MLX · vLLM on Brev (NVFP4) | OpenMDW-1.1 |
| Visual speech | Auto-AVSR `LRS3_V_WER19.1` via Chaplin (`huggingface.co/Amanvir/LRS3_V_WER19.1`) | 0.250 B | PyTorch MPS | code Apache-2.0 / MIT (Chaplin); weights follow LRS3 terms |
| VSR beam-search LM | Auto-AVSR `lm_en_subword` | 0.054 B | PyTorch MPS | as above |
| ASR | Whisper base, `mlx-community/whisper-base-mlx` | 0.072 B | MLX | MIT |
| Speech enhancement | RNNoise (xiph) via `pyrnnoise` | ≈0.0001 B | CPU | BSD-3 |
| TTS | Kokoro-82M, `mlx-community/Kokoro-82M-bf16` | 0.082 B | MLX | Apache-2.0 (voices per model card) |
| Face / blink (browser) | MediaPipe Face Landmarker float16 | ≤0.002 B | wasm/WebGL | Apache-2.0 |
| Gaze (browser) | MobileGaze MobileOne-S0 eye-image model (`mobileone_s0_gaze.onnx`, trained on Gaze360) + per-user ridge calibration on MediaPipe iris/head-pose features | 0.00124 B (ONNX initializers) | ONNX Runtime Web (WebGPU / WebAssembly) | MIT |
| Mouth detector (VSR preprocessing) | MediaPipe face detection 0.10.21 | ≈0.001 B (estimate) | CPU | Apache-2.0 |
| P300 decoder | xDAWN covariances → tangent space → logistic regression, trained per participant | ~50 weights | scikit-learn / pyRiemann | BSD-3 |
| **Total distinct deployed** | | **≈32.04 B ≤ 33 B** | | |

Not deployed and not counted: LipVoicer and VoiceBridge, which are research references only (no verified checkpoint for VoiceBridge). `granite4.2:8b` appears in the coordinator evaluation only as a comparison policy.

## ASR candidates under evaluation (`asr_eval/`, not deployed yet)

These models are benchmarked on Brev to choose a replacement for Whisper base in the voice agent (see [BREV.md](BREV.md) and `asr_eval/README.md`).
- Licences were read from each Hugging Face model card on 2026-10-17.
- "Measured" parameter counts come from the loaded weights (`results/raw/{model}_info.json`). The others are model-card figures until measured.
- Revisions are pinned at run time and logged.

| Model | Checkpoint (revision) | Params | Runtime in harness | Licence | Phrase scoring | Status |
|---|---|---:|---|---|---|---|
| Parakeet-CTC 0.6B | `nvidia/parakeet-ctc-0.6b` (`ad09ba1cc627`) | 608.8 M (measured) | transformers `ParakeetForCTC` | CC-BY-4.0 | Forced CTC (`ctc_loss`) + text | Milestone 1: running on Brev |
| Parakeet-TDT 0.6B v2 | `nvidia/parakeet-tdt-0.6b-v2` (`ae9ad07059`) | 617.8 M (measured) | NeMo | CC-BY-4.0 | Text (forced TDT scoring too slow without NVVM: 3.5–7 s/clip) | Evaluated; tied with Nemotron on text matching |
| **Nemotron Speech Streaming en 0.6B (selected)** | `nvidia/nemotron-speech-streaming-en-0.6b` (`ebe59e5a81`) | 618.1 M (measured) | NeMo, default lookahead; 160/560 ms variants evaluated | NVIDIA Open Model License | Forced RNNT scoring + text | **Chosen for the voice agent**: best on recorded speakers (LOSO 71.5% accepted, 90.2% resolved within one gaze question, 0/45 near-miss false accepts, 43 ms p50) |
| Nemotron 3.5 ASR streaming 0.6B (multilingual) | `nvidia/nemotron-3.5-asr-streaming-0.6b` (`ea30d66deb`) | 638.0 M (measured) | NeMo `EncDecRNNTBPEModelWithPrompt`, language via `prompt_indices` | OpenMDW-1.1 | Forced RNNT scoring + text | **Not adopted (yet)**: with the en-US prompt it scored lower on our English speakers (43.8% accepted, 78.3% one-question) and dysarthric TORGO (39% vs 28% WER), and scoring is slower (13k-token vocabulary; p95 263 vs 94 ms). Multilingual code is in place (`languages` in the config, draft translations) for a deployment that needs it |
| Whisper large-v3-turbo | faster-whisper → `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | ~0.81 B (card) | faster-whisper fp16, beam 3, T=0 | MIT (OpenAI and conversion) | Text + `no_speech_prob`/`avg_logprob` | Milestone 2 |
| Whisper large-v2 | faster-whisper → `Systran/faster-whisper-large-v2` | ~1.55 B (card) | faster-whisper | Apache-2.0 (OpenAI) / MIT (conversion) | Text | Milestone 2, reference only (does not fit the budget) |
| Whisper small | faster-whisper → `Systran/faster-whisper-small` | ~0.24 B (card) | faster-whisper | Apache-2.0 (OpenAI) / MIT (conversion) | Text | Milestone 2 |
| Canary-1B-Flash | `nvidia/canary-1b-flash` | ~1 B (card) | NeMo | CC-BY-4.0 | Text | Optional |
| Parakeet-CTC 0.6B SAPC1 (dysarthric fine-tune) | `dys-asr/parakeet-ctc-0.6b-sapc1` | ~0.6 B | transformers | Speech Accessibility Project DUA | — | **Skipped**: needs the SAP data use agreement. Also outputs upper-case text |
| Parakeet-TDT 0.6B unconstrained (dysarthric) | `dys-asr/parakeet-tdt-0.6b-unconstrained` | ~0.6 B | NeMo | SAP DUA, gated (manual approval) | — | **Skipped**: same reason |

Supporting components in the harness:

| Component | Use | Licence |
|---|---|---|
| Silero VAD 6.x (`silero-vad`) | Speech gate before any phrase is proposed | MIT |
| Kokoro-82M (`hexgrad/Kokoro-82M`, all English voices) | Generates synthetic phrases for **sanity checks only**, never headline numbers | Apache-2.0 |
| `pronouncing` + `cmudict` (CMU Pronouncing Dictionary) | Phonetic similarity for transcript → phrase matching | `pronouncing`: BSD-style (licence file); the `cmudict` Python package declares GPL-3.0-or-later. Evaluation-only dependency: keep it out of shipped app code, or swap in a permissive phonetic source before deployment |
| DeepFilterNet | Enhancement ablation, off by default | MIT / Apache-2.0 |

**Budget.** The deployed ledger is 32.04 B of 33 B, and Whisper base accounts for 0.07 B of it. Replacing Whisper base leaves about 1.03 B for the new ASR:
- a 0.6 B Parakeet/Nemotron fits;
- Whisper large-v3-turbo (0.81 B) fits;
- Whisper large-v2 (1.55 B) does not, and two candidates together do not.

## Pinned upstream code

| Repo | Commit | Use |
|---|---|---|
| amanvirparhar/chaplin | `7aee1f8fca776ce4f63690063310b53573b7d804` | VSR pipeline: `pipelines/` + `espnet/`. Patched `espnet/nets/ctc_prefix_score.py` for MPS (setup script applies it). |
| weichkai/p300-speller-bci | `6319bb86e51a514d8987564d6e93fc64a600e3b3` | Reference for the xDAWN/Riemann pipeline and epoching. Our adapter reads the original BNCI `.mat` files directly, including row/column flash codes. |
| NVIDIA-NeMo/Gym | `f3c422998b2f5b98dadb977f4cce55301f980b15` | Resources-server conventions for `evaluations/nemo_gym/unspoken_communication`. |

## Known model limitations

- **Auto-AVSR** was trained on LRS3 broadcast speech. Its benchmark WER (~19–20%) is not webcam or patient accuracy; see EVALUATION.md for unseen-speaker GRID results. A still face makes it hallucinate fluent sentences, so mouth-motion gating is mandatory.
- **Whisper** is not validated on dysarthric speech, and it hallucinates on silence. The voice agent returns no hypotheses when RNNoise finds no voice activity.
- **RNNoise + gain** can make ASR *worse* in heavy noise (measured). Raw and enhanced transcripts are always shown side by side.
- **WebGazer** gives an estimate of screen position with tens to hundreds of pixels of error. It is not clinical eye tracking.
- **Nemotron** is used only to pick the next interaction step. Every call is validated by code, and it cannot produce message text.
