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

The Auto-AVSR LRS3 model has never seen these speakers or GRID's letter/digit grammar. Its paper LRS3 WER of about 19% does **not** transfer, which is why lip readings are only ever *candidates* for the person to confirm. The median latency of 7.9 s is **historical**: measured on the Mac while the coordinator benchmark shared the GPU. On Brev a clip takes about 0.4 s (§7). With a still face the model still produces fluent sentences (for example "THAT'S WHAT I'M GOING TO DO"); the mouth-motion gate flags every such clip.

**Larger run on Brev (132 clips, 4 from each of 33 unseen GRID speakers, beam 12, `vsr_speed_accuracy.json`):** WER 69.4%, best-of-5 60.9%, CER 47.3%. Beam 6 is ≈50 ms faster but worse (71.1% / 64.5%), so beam 12 stays; fp16 gives identical answers and is not faster.

**Personal enrollment (current default on `ui`).** The lip agent now matches clips against the person's own recordings (22 phrases, 90 recordings, DTW over encoder features, threshold 0.30) instead of free text. Measured only incidentally: 5 of 43 GRID clips from strangers matched an enrolled phrase within the threshold (≈12% false matches). The threshold was set from 10 recordings; a held-out evaluation is still needed.

## 3. Weak speech: raw vs RNNoise-enhanced Whisper-base (MLX), the previous voice agent

*Historical: this was the voice agent before Nemotron Speech replaced it (§6). Kept because the finding still applies: enhancement is not used in front of the new model either.*

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

## 4. Coordinator: NeMo Gym environment `unspoken_communication`

**Current (Brev, vLLM, `ui` branch prompt with `propose_correction` and `ignore_input`, 72 tasks, 12 categories, 2026-10-18, `coordinator_brev.json`):**

| Policy | Reward | Valid | Correct action | Unsafe | Decided by model | p50 latency (Brev) |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3.5 Lightning, raw | 0.917 | 100% | 75% | **0%** | 100% | 272 ms |
| **Deployed: Nemotron + outside validator + fallback** | 0.940 | 100% | 82% | **0%** | 92% | 275 ms |
| Deterministic rules (reference) | 1.000 | 100% | 100% | 0% | 0% | 0 ms |

Across six Brev runs of the default prompt (including the MTP and cache-mode server variants): raw 0.912–0.931, deployed 0.940–0.949.

**Held-out** (44 tasks: seed 2026, without the tasks that also occur in the eval set; 10 categories; `coordinator_heldout_brev.json`): Nemotron alone **0.939**, deployed **0.985**, rules 1.000, 0% unsafe. The whole gain is `none_of_these` (0.67 → 1.00); Nemotron decides 86% of the tasks itself. Compact mode: 0.894 / 0.955 (`coordinator_heldout_compact_brev.json`). Write-up: [BOUNTIES.md](BOUNTIES.md) §2. Most wrong actions of the deployed coordinator are `ignore_input` on real input: a person who failed twice (6/6; should switch input method), an injection next to a genuine request, uncalibrated gaze, a clear negated reading.

**Optional validator rules (flags off by default; `coordinator_validator_v2_*.json`, Brev, 2026-10-18):**
- `UNSPOKEN_VALIDATOR_NEGATION=1`: reject a `propose_correction` that adds or removes a negation (the voice
  agent's own check, `asr_eval/harness/text.py`).
- `UNSPOKEN_VALIDATOR_IGNORE_HISTORY=1`: reject `ignore_input` after two failed tries, after a "none of these" in
  the last 3 turns, or when a specialist accepted a lip/audio reading with no quality flags.
- A rejected call falls back to the rules.

Paired measurement (`validator_ab.py`): one model call per task, then each setting re-validates that same call,
so the differences are the validator's alone. Deployed reward:

| Setting | Eval (72) | Held-out (44) | Unsafe |
|---|---:|---:|---:|
| Nemotron alone (no validator) | 0.926 | 0.932 | 0 |
| Validator as today (flags off) | 0.954 | 0.977 | 0 |
| + negation rule | 0.954 | 0.977 | 0 |
| + ignore-history rule | **0.995** | **0.985** | 0 |
| both | **0.995** | **0.985** | 0 |

Per category, deployed, flags off → both on:

| Category | Eval | Held-out |
|---|---|---|
| `repeated_failure` | **0.67 → 1.00** | — (not in held-out) |
| `none_of_these` | 1.00 → 1.00 | 1.00 → 1.00 |
| `negation_kept` | **0.89 → 1.00** | 1.00 → 1.00 |
| `misread` | 0.94 → 1.00 | 0.94 → 1.00 |
| `prompt_injection` | 0.94 → 0.94 | 0.83 → 0.83 |
| `gaze_uncalibrated` | 1.00 → 1.00 | 1.00 → 1.00 |

- **The negation rule changed nothing measured:** in these runs the model never dropped a negation. The
  `negation_kept` misses were `ignore_input` on a negated reading, which the ignore-history rule catches.
  It stays as a guard.
- **Nemotron decides less with the ignore-history rule:** 92% → 79% of eval tasks (86% → 84% held-out); the rest
  fall back to the rules.
- **Unpaired runs** (`run_local.py`, flags set in the environment; each policy calls the model again), deployed:
  - eval 0.944 → 0.982;
  - held-out 0.977 → 0.985;
  - the one category that dropped (`gaze_uncalibrated` 1.00 → 0.94, eval) is model run-to-run variation: the paired
    run shows 1.00 for both settings.

**Speed variants (same 72 tasks, Brev):**

| Variant | Raw | Deployed | p50 | Output tokens | Adopted |
|---|---:|---:|---:|---:|---|
| Default (tool calls, verbose prompt) | 0.912–0.917 | 0.940–0.949 | 272 ms | 65 | yes |
| MTP speculative decoding | 0.912 | 0.940 | 229 ms | 65 | no: one crash under full load |
| Compact JSON, numbered candidates (v2) | 0.907 | 0.949 | 84 ms | 15 | no: ambiguous 0.89, after a rejection 0.78–0.89 |
| Compact v2 + MTP | 0.907 | 0.963 | 78 ms | 15 | no |
| Compact v1 (looser "fix" rule) | 0.806 | 0.931 | 97 ms | 18 | no: corrected readings that were fine | Misreads were handled right in 5 of 6 tasks and no correction ever dropped a negation, although the validator would accept one. The validator's history rules (retake, none of these) do not apply to `ignore_input`, which is how the old failure mode comes back. The scorer now knows the two new tools; ignoring is accepted only for failed captures with no reading.

**Earlier (Mac, Ollama, previous prompt without the two tools, 60 tasks, 10 categories, `coordinator.json`):**

| Policy | Reward | Valid | Correct action | Unsafe | Decided by model | p50 latency (Mac/Ollama) |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3.5 Lightning, raw (no guard) | 0.933 | 100% | 80% | **0%** | 100% | 4.6 s |
| granite4.2:8b, raw (comparison) | 0.933 | 100% | 80% | 0% | 100% | 6.2 s |
| **Deployed: Nemotron + outside validator + fallback** | **1.000** | 100% | 100% | **0%** | 80% | 2.9 s |
| Deterministic rules (reference) | 1.000 | 100% | 100% | 0% | 0% | 0 ms |

Both raw models fail on the same thing: history. After two failed retakes they ask for a third, and after "none of these" they re-ask instead of switching to spelling. Those product rules now live in the validator, so a rejected model call falls back to rules. The deterministic policy was written to the same specification as the verifier, so its 1.0 is a ceiling, not an independent baseline. The environment's own tests pass inside the real NeMo Gym package (5/5); re-run 2026-10-18 on Brev with the current 72-task environment (NVIDIA-NeMo/Gym `b3103ea`, nemo-gym 0.7.0rc0, Python 3.13.16): 5/5.

## 5. Confirmation safety and end-to-end exactness

- **Unit tests** (`pytest evaluations/safety_tests`): **49 pass** (2026-10-18, `demo-final`):
  - 18 in `test_confirmation.py`: policy gate and state machine;
  - 7 in `test_auto_speak.py`: auto-speak only for accepted readings;
  - 4 in `test_auto_speak_delay.py`: with the countdown flag, No inside the window speaks nothing, no answer
    speaks once, and a Yes is not doubled;
  - 3 in `test_vsr_phrase_scoring.py`: an impossible phrase-list alignment never ranks first;
  - 11 in `test_validator_flags.py`: the optional validator rules, both on and off;
  - 6 in `test_checkin.py`: the vision-preview check-in.
    - It is absent when off.
    - The nurse alert comes first.
    - The answer never reaches the gate or TTS.
    - It waits while a message is pending or speech plays.
    - Nothing is auto-spoken over it.
- **Live attacks** (`evaluations/test_cases/e2e_exactness.py`, against the running stack): **7/7 blocked**:
  5 unconfirmed-speech probes, a forged TTS token, and a non-explicit confirmation method (`e2e.json`).
- **Exactness** (same script): 10 phrases are put on the board → confirmed → spoken by Kokoro → re-transcribed by
  the voice agent.
  - 2026-10-17, with the earlier Whisper voice agent: **10/10** matched the confirmed string exactly (`e2e.json`).
  - 2026-10-18, `demo-final` on Brev (`e2e_demo_final.json`): the text the gateway sent to TTS equalled the
    confirmed string **10/10**, but only **3/10** re-transcriptions matched.
  - That check no longer works: today's voice agent only maps audio onto its 20 fixed phrases, so it cannot
    transcribe free text. The 3 matches are the 3 phrases on its list; "Thank you" came back as "I can't
    breathe". An independent free-text ASR is needed to repeat it.
- **Unconfirmed or contradicted text reaching TTS: 0**, enforced by construction (HMAC token over the exact text) and tested.

## 6. Voice agent model selection: 20-phrase task (`asr_eval/`; adopted in the app)

**Question.** Which open ASR model can map a weak or impaired utterance onto **one of 20 request phrases**, and refuse otherwise? The voice agent's output is a phrase candidate, `unknown`, or `no_speech`.
- Proposing a phrase on silence, noise or unrelated speech is the worst failure.
- The wrong phrase is next worst.
- Abstaining is acceptable.

**Protocol** (full settings in `asr_eval/configs/eval.yaml`, one command to reproduce):
- **Data:**
  - Own consented recordings in six conditions (normal, soft, whisper, slow, muffled, noisy).
  - TORGO dysarthric and control speakers.
  - LibriSpeech, MUSAN, Speech Commands and ESC-50 negatives.
  - Generated silence.
  - Hard negatives such as "I'm not in pain" and "water the plants".
  - Kokoro TTS as a sanity check only.
  - A degradation grid (gain, SNR, telephone band, low-pass, tempo, reverb), one effect at a time.
- **Split:** by speaker into calib and test. Augmented copies inherit their source's split, and overlap is asserted.
- **Speech gate:** Silero VAD with a 100 ms minimum, so "yes" and "no" survive.
- **Decision:**
  - Every model: phonetic matching of the transcript.
  - CTC models only: forced phrase scoring.
  - `unknown` on a low score, a small margin, or a negation mismatch.
  - Thresholds are tuned on calib only. Objective: the most correct accepts with a false-accept rate ≤ 1% for every negative type.
- **Reporting:** every rate comes with k/n and a Wilson interval, plus the rule-of-three bound when k = 0. TTS and degraded rows are kept apart from real recordings. The best configuration is selected on calib and reported on test.

**Result (2026-10-17, 3 recorded speakers, leave-one-speaker-out, `asr_eval/results/loso.md`).** Nemotron Speech Streaming en 0.6B with forced RNNT phrase scoring (`forced_llr`) is the voice agent's model.
- It accepted the right phrase for 168/235 real recordings (71.5%), with 1 wrong phrase.
- It resolved 212/235 (90.2%) within one gaze question (candidate, or the right phrase in the coordinator's top 3).
- False accepts: 0/45 near-miss sentences, 1/19 silence/noise/cough, 0/7 unrelated speech.
- Latency: 43 ms median, 94 ms p95, per clip on the GPU.
- Remaining losses are noisy, muffled and whispered speech.
- The multilingual Nemotron 3.5 was tested with an English prompt and was less accurate and slower, so it was not adopted.

**In the app (2026-10-18).** `services/audio/` runs this configuration at the balanced operating point through `asr_eval/harness/live.py`, the same recognizer as the live test bench, so the app decides exactly as evaluated. Spot check on Brev through the gateway (Windows SAPI voice):

| Clip | Decision | Time |
|---|---|---:|
| "I need water" | proposed "I need water" (margin 3.4 over the runner-up) | 71 ms |
| "I'm in pain" | proposed "I'm in pain" | 239 ms |
| "I don't need water" | not accepted (low score); top 3 offered as a question | 145 ms |
| 2 s of room noise | no speech, no hypotheses | 206 ms |

## 7. Latency with every model running (Brev, 2026-10-18)

Full method, tables and serving advice: [SERVING.md](SERVING.md) and [TECHNICAL_VERBOSE.md](TECHNICAL_VERBOSE.md) §7.

**Now** (MPS on, MTP off, after the speedups; `latency_brev_final2.json` unless noted):

| Measurement | p50 / p95 |
|---|---:|
| One message end to end on the box (voice + lip clip → spoken) | **309 / 322 ms** |
| Accepted voice message, clip → spoken, on the box (ad-hoc bench, same run, printed) | **71 / 89 ms** (73 / 84 cold phrase cache, 71 / 80 warm on `demo-final`) |
| Streamed lip clip: reading after the clip ends, on the box / from the laptop | **104 ms** / **407 ms** (whole clip: 477 / 898 ms) |
| Voice result after the 700 ms pause, early decode, on the box / from the laptop | 2 ms / 64 ms |

What a person waits for (added up from the measured parts, not one end-to-end measurement): **0.85–0.9 s** for
an accepted voice message (0.7 s of it is the deliberate pause), **about 1.3 s** to the question for a lip message
(0.9 s of it is the pause).

**Earlier: branch `ui` before the speedups** (`latency_brev_ui_mps{0,1}.json`):

- **One message end to end** (voice + lip clip → evidence → coordinator → auto-speak): 0.43 s p50, 0.47 s p95, with or without MPS. The lip agent (≈0.4 s) sets it; the coordinator runs alongside it.
- **With every agent busy:** MPS cuts the voice agent from 244 to 75 ms p50 and Kokoro from 79 to 30 ms.
- **Early voice decode:** the voice result is in the session 2–3 ms after the 700 ms end-of-utterance pause (109–337 ms when sent at that moment); from the laptop through the tunnel, 64 ms instead of 211 ms. 0 of 304 real recordings changed decision (`trailing_silence_check.json`).
- **From the laptop:** a 1.2 MB lip clip (2.5 Mbps webm) takes 0.68 s to upload through the tunnel; round trips cost ≈120 ms.

Earlier runs (before the `ui` changes: free-text lip reading, confirm step): 1.03 s p50 end to end without MPS, 0.97 s with MPS (`latency_brev*.json`).

## Not yet measured (honest gaps)

- Live webcam lip reading on unfamiliar, consenting people. This needs local opt-in recordings.
- Gaze selection accuracy and false-selection rate across lighting and head motion, now with MediaPipe iris + MobileGaze. The calibration screen reports a per-session error on unseen dots, but no study has been done, and calibration with a real webcam is still being debugged (HANDOFF.md).
- Clarification efficiency (selections per successful message) with real users.
- EEG+gaze fusion. BNCI2014-008 has no synchronized eye tracking; bigP3BCI sessions with eye data would be needed.
