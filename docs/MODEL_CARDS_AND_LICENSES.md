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
| Gaze (browser) | WebGazer.js 3.5.3 (TF.js facemesh + ridge) | ≈0.001 B (estimate) | browser | GPL-3.0-or-later |
| Mouth detector (VSR preprocessing) | MediaPipe face detection 0.10.21 | ≈0.001 B (estimate) | CPU | Apache-2.0 |
| P300 decoder | xDAWN covariances → tangent space → logistic regression, trained per participant | ~50 weights | scikit-learn / pyRiemann | BSD-3 |
| **Total distinct deployed** | | **≈32.04 B ≤ 33 B** | | |

Not deployed and not counted: LipVoicer and VoiceBridge, which are research references only (no verified checkpoint for VoiceBridge). `granite4.2:8b` appears in the coordinator evaluation only as a comparison policy.

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
