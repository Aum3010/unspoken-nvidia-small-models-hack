# Evaluation

Reproduce everything with `./scripts/run_evals.sh` (services must be running). Raw outputs are in `evaluations/results/*.json`, and the merged view is in `summary.json`, which the **Evidence** page renders.

## 1. Brain-signal replay: BNCI2014-008, 8 participants with ALS

The decoder is xDAWN covariances → Riemannian tangent space → logistic regression. Epochs run from −0.1 to 0.7 s after each flash, with a 1–20 Hz band-pass. It is trained on the 15 calibration characters (runs 1–3) and scored on the 20 online characters (runs 4–7), the original protocol split, so there is no leakage of repeated flashes. The baseline is shrinkage LDA on decimated epochs.

| | Flash AUC | Letters @1 rep | @5 reps | @10 reps | LDA @10 |
|---|---:|---:|---:|---:|---:|
| Mean of 8 | **0.850** | 31% | 81% | **94%** | 96% |
| Range | 0.783–0.939 | 15–55% | 65–100% | 90–100% | 90–100% |

Chance is 2.8%. These are healthy-protocol research recordings, so they do not show how a new person would do.

## 2. Lip reading: GRID corpus, unseen speakers (s1, s2; 40 clips)

| top-1 WER | best-of-5 WER | top-1 CER | still-face clips flagged | unflagged false phrase |
|---:|---:|---:|---:|---:|
| **79.6%** | 71.3% | 55.0% | **8/8** | **0%** |

The Auto-AVSR LRS3 model has never seen these speakers or GRID's letter/digit grammar. Its paper LRS3 WER of about 19% does **not** transfer, which is why lip readings are only ever *candidates* for the person to confirm. The median latency of 7.9 s was measured while the coordinator benchmark shared the GPU, so it is a contended number. With a still face the model still produces fluent sentences (for example "THAT'S WHAT I'M GOING TO DO"); the mouth-motion gate flags every such clip.

### 2b. Lip reading: original vs adapted (same held-out speakers s1, s2; 200 clips)

`scripts/finetune_vsr_grid.py` keeps the Auto-AVSR visual encoder frozen and fine-tunes only the decoder and CTC head (68.4M parameters, no new weights) on 2,850 clips from GRID speakers s3–s8. It took 4 epochs, about 6 minutes on an M2. The original model is the baseline throughout, and it stays the deployed default.

| Free text (beam 12) | WER | decode p50 |
|---|---:|---:|
| Original (LRS3 LM 0.3, deployed) | 80.7% | 2.0 s |
| **Fine-tuned decoder, no LM** | **4.6%** | 1.1 s |
| Fine-tuned decoder + LRS3 LM 0.3 | 6.4% | 1.5 s |

**Closed phrase list.** Instead of free decoding, every phrase in a fixed list is scored against the video (teacher-forced decoder + CTC log-likelihood, one batched pass of about 0.1 s for 44 phrases). Each test clip's true sentence is mixed with N−1 other GRID sentences.

| Distractors | N | Original top-1 / top-3 | Fine-tuned top-1 / top-3 |
|---|---:|---:|---:|
| random | 10 / 50 / 100 | 100 / 99.5 / 100% · top-3 100% | 100 / 100 / 99.5% · top-3 100% |
| differ in **one word** | 10 | 90.5% / 99.0% | 94.0% / 100% |
| differ in **one word** | 50 | 76.5% / 98.0% | 81.5% / 99.0% |

**Findings.** (1) Adapting the decoder to a fixed sentence set cuts WER 17× on speakers it never saw, so the recipe works. (2) Even without training, ranking a closed list puts the right phrase in the top 3 for ≥98% of clips. That suits the confirm-from-three-cards interaction. **Limits:** the adapted decoder only knows GRID's 51-word grammar and is not deployed. GRID is frontal studio video of healthy speakers speaking aloud, so webcam and patient accuracy will be lower. The deployed phrase list (`UNSPOKEN_VSR_PHRASES=services/vsr/patient_phrases.txt`) has not been measured on webcam video yet. GRID's distributed s5/s6 video archives are swapped relative to their alignments, and the folders were swapped back before training.

## 3. Weak speech: raw vs RNNoise-enhanced Whisper-base (MLX)

The test set is 10 synthetic Kokoro utterances (5 voices) with pink noise, so the references are exact. This is **not** a dysarthria benchmark.

| Condition | WER raw | WER enhanced | Inserted words raw/enh |
|---|---:|---:|---:|
| Normal level, quiet | 0% | 0% | 0/0 |
| −42 dBFS, quiet | 0% | 0% | 0/0 |
| SNR 5 dB | 0% | 2% | 0/0 |
| −40 dBFS, SNR 0 dB | 4% | 2% | 1/0 |
| −40 dBFS, SNR −5 dB | **8%** | **18%** | 0/1 |
| −40 dBFS, SNR −10 dB | **49%** | **65%** | 4/4 |
| Noise only (10 clips) | declined 10/10 | | 0/0 |

**Finding:** in heavy noise, enhancement makes recognition *worse* and invents plausible words ("Please call my dog" → "Please call my daughter"). Both transcripts are therefore always shown, with a word diff, and are never merged.

## 4. Coordinator: NeMo Gym environment `unspoken_communication` (60 tasks, 10 categories)

| Policy | Reward | Valid | Correct action | Unsafe | Decided by model | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3.5 Lightning, raw (no guard) | 0.933 | 100% | 80% | **0%** | 100% | 4.6 s |
| granite4.2:8b, raw (comparison) | 0.933 | 100% | 80% | 0% | 100% | 6.2 s |
| **Deployed: Nemotron + outside validator + fallback** | **1.000** | 100% | 100% | **0%** | 80% | 2.9 s |
| Deterministic rules (reference) | 1.000 | 100% | 100% | 0% | 0% | 0 ms |

Both raw models fail on the same thing: history. After two failed retakes they ask for a third, and after "none of these" they re-ask instead of switching to spelling. Those product rules now live in the validator, so a rejected model call falls back to rules. The deterministic policy was written to the same specification as the verifier, so its 1.0 is a ceiling, not an independent baseline. The environment's own tests pass inside the real NeMo Gym package (5/5).

## 5. Confirmation safety and end-to-end exactness

- 18 unit tests on the policy gate and state machine, plus 7 live attacks: unconfirmed probes ×5, a forged TTS token, and a non-explicit confirmation method. **25/25 pass.**
- In the end-to-end test, 10 phrases were selected → confirmed → spoken by Kokoro → re-transcribed by Whisper. **10/10 matched the confirmed string exactly.**
- **Unconfirmed or contradicted text reaching TTS: 0**, enforced by construction (HMAC token over the exact text) and tested.

## Not yet measured (honest gaps)

- Live webcam lip reading on unfamiliar, consenting people. This needs local opt-in recordings.
- Gaze selection accuracy and false-selection rate across lighting and head motion. The calibration screen reports a per-session error, but no study has been done.
- Clarification efficiency (selections per successful message) with real users.
- EEG+gaze fusion. BNCI2014-008 has no synchronized eye tracking; bigP3BCI sessions with eye data would be needed.
