# Results: what each agent gains over its baseline

**The metric.** A wrong word spoken for a patient is worse than a pause. So every agent is scored the same way, on
people it has never seen: **correct commits, wrong commits, abstentions ("not sure, ask again") and time**. Raw accuracy
alone would reward an agent that guesses.

Every number below links to the file it comes from and the command that reproduces it. Caveats are stated in the row
they apply to, not in a footnote.

| Agent | Baseline → ours | Held-out data (open) | Caveat |
|---|---|---|---|
| [Voice](#1-voice-score-the-phrases-dont-transcribe) | Dysarthric speakers: **40.0% → 61.1%** command words accepted correctly, wrong accepts 1.1% → 0% | TORGO; 3 speakers of our own, leave-one-speaker-out | TORGO here is 10 single command words; our phrase set has 3 speakers |
| [Lips, fine-tuned](#2-lips-specialise-the-decoder) | Word error rate on unseen speakers: **80.7% → 4.6%** | GRID, speakers s1 and s2 never trained on | GRID is a fixed 51-word grammar: proof of the recipe, not deployed |
| [Lips, personal](#2-lips-specialise-the-decoder) | Top-1 phrase: **72.0% → 96.1%** (phrase list → own recordings) | MIRACL-VC1, 15 speakers | MIRACL-VC1 is licensed for research use only |
| [EEG, ALS patients](#3-eeg-alone-als-patients-study-f) | Wrong characters: **24 → 1** per 120, same decoder; throughput not lower (2.51 → 3.13 correct/min) | bigP3BCI Study F, 10 ALS participants, session 3 | Yields on 36% of selections; 4 of 10 participants cannot be served reliably by EEG; throughput CIs overlap |
| [EEG + gaze](#4-eeg--gaze-study-h) | Correct within 3 s at ~1% wrong: **2.9% → 29.1%** (+26.1 points, 95% CI 18.2 to 35.3) | bigP3BCI Study H, 16 participants, held-out runs | Participants do not have ALS: supports the mechanism, not ALS speed |
| [Lips + voice](#6-lips--voice-on-the-same-utterances) | Babble at 0 dB, correct commits: voice **8.0%**, lips **53.5%**, together **76.0%**; at every noise level together beats both. Latency = the slower agent (348 ms p50), not the sum | GRID s1+s2, 200 utterances, real paired audio and video, neither speaker trained on | Fixed 51-word grammar; candidates differ in one word; wrong commits 2.9% overall vs a 2% target |
| [Coordinator](#5-coordinator-nemo-gym-unspoken_communication) | Right action **82% → 95%**, unsafe 0% → 0%, 278 ms p50 (Nemotron alone → Nemotron + validator) | NeMo Gym `unspoken_communication`, 44 held-out tasks | The gain is validator code, not a retrained model |

---

## 1. Voice: score the phrases, don't transcribe

**Question.** A patient with weak or slurred speech says one of 20 request phrases. Can the voice agent name the right
phrase, and refuse on silence, noise or unrelated speech?

**Method.** Same model throughout: NVIDIA Nemotron Speech Streaming en 0.6B. The baseline transcribes, then matches the
text to the phrase list. Ours scores every listed phrase directly against the audio (forced RNNT alignment, normalised
log-likelihood ratio, `forced_llr`). Thresholds are tuned on other speakers than the one tested.

| | Transcribe, then match | Score phrases (ours) |
|---|---:|---:|
| TORGO dysarthric speakers: accepted correct | 36/90 (40.0%) | **55/90 (61.1%)** |
| TORGO dysarthric speakers: wrong accept | 1/90 (1.1%) | **0/90** |
| TORGO control speakers: accepted correct | 63/118 (53.4%) | **89/118 (75.4%)** |
| Own recordings: resolved within one gaze question | 196/235 (83.4%) | **212/235 (90.2%)** |
| Own recordings: wrong phrase accepted | 1/235 | 1/235 |
| Own recordings: false accepts on near-miss sentences | 1/45 | **0/45** |
| Latency (VAD + ASR + scoring, GPU) | | 43 ms p50, 94 ms p95 |

Model choice: 5 open ASR models × 4 decision methods were compared on the same split (`asr_eval/results/summary.md`).
Nemotron Streaming with phrase scoring had the best balance of one-question resolution and wrong accepts.

Files: `asr_eval/results/loso.md`, `asr_eval/results/summary.md`. Reproduce: `python asr_eval/run_eval.py --config asr_eval/configs/eval.yaml`.

## 2. Lips: specialise the decoder

**Fine-tuning (GRID).** Auto-AVSR visual-only (LRS3) with the encoder frozen; only the decoder and CTC head are trained
(68.4M parameters, no new ones). Test speakers s1 and s2 are never seen in training.

| Decoder | Training speakers | WER (200 test clips) | Decode, median |
|---|---|---:|---:|
| Original (LRS3) | | 80.7% | 289 ms |
| Fine-tuned, no LM | 6 (Mac) | **4.6%** | |
| Fine-tuned, no LM | 31 (Brev) | **5.8%** | 143 ms |

More speakers did not help: the decoder learns the grammar. Remaining errors are mostly single letters, which are nearly
invisible on the lips. The data pipeline also found that Zenodo's GRID transcripts do not match the videos for 17 of the
31 training speakers; the setup script checks every speaker and fetches correct alignments.

**Personal enrollment (deployed).** The clip is matched against the person's own recordings of their phrases (dynamic
time warping over the frozen encoder's features). MIRACL-VC1, 10 phrases, one take per phrase enrolled, 2,700 test
clips: top-1 **96.1%** vs 72.0% for scoring a phrase list, top-3 99.2% vs 89.7%. On one person's 83 webcam takes of our
20 phrases (leave-one-out): top-1 73%, top-3 83%. A clip farther than 0.30 from every recording gives no reading.

Files: `evaluations/results/vsr_finetune.json`, `evaluations/results/brev/vsr_finetune.json`,
`evaluations/results/brev/vsr_enrollment_miracl.json`, `docs/LIP_READING.md`. Reproduce: `scripts/finetune_vsr_grid.py`,
`scripts/eval_enrollment_miracl.py`.

## 3. EEG alone, ALS patients (Study F)

**Question.** From P300 EEG of people with ALS, how quickly can a character be picked without picking wrong ones?

**Method.** Decoders fitted per participant on sessions 1-2 only. Session 3 (120 selections) scored once, after all
choices were written to disk. Baseline: the original Bayesian stopping rule at repetition boundaries, which always
commits when the recording ends. Ours: ensemble decoder, flash-level stopping, threshold and deadline chosen on
calibration data at a 2% risk target, and **yield** ("keep looking") instead of a forced guess. The table holds the
decoder fixed (ensemble), so the difference is the stopping policy alone.

| Ensemble decoder | Baseline (Bayes, tau 0.99, always commit) | Ours (flash-level, 2% risk, yield) |
|---|---:|---:|
| Wrong characters committed (of 120) | 24 | **1** |
| Correct | 80.0% | 63.3% (yields on 35.8%) |
| Correct characters per minute, pooled | 2.51 | 3.13 |
| Correct characters per minute, mean of participants [95% CI] | 3.93 [2.17, 5.90] | 4.39 [1.99, 7.14] |
| Median time per attempt / per correct character | 16.5 s / 12.0 s | **9.2 s / 8.5 s** |

Against the original xDAWN baseline the change is 29 → 1 wrong and 2.11 → 3.13 correct/min, but part of that is the
decoder. The claim this supports: **nearly no wrong characters, at no loss of throughput**. The throughput difference is
within noise.

Under 3 s nothing is reliable: no policy gets correct characters within 3 s without about half its commits being
wrong. Five strong participants get 0-1 wrong in 12 at a median of 4.6-13.4 s; F_08 gets 12/12 within 5 s. F_07 has no
usable P300, and the policy never commits for them.

Files: `evaluations/results/eeg_stopping/`, `evaluations/results/eeg_fast_selection/`, `docs/EEG_STOPPING_BENCHMARK.md`,
`docs/EEG_FAST_SELECTION.md`. Reproduce: `scripts/benchmark/run_study_f_baselines.py`, `run_study_f_fast_selection.py`.

## 4. EEG + gaze (Study H)

**Question.** Does a co-recorded eye tracker, fused with the same EEG evidence, make a selection reliable within 1-3 s?

**Method.** 16 participants, leave-one-block-out. EEG evidence as in Study F; gaze evidence from a per-participant map
of gaze to the 72-key grid with a lapse model; fused as independent evidence. Policies commit when confident and yield
otherwise; thresholds chosen on calibration blocks at a 2% risk target. Budgets fixed at 1-10 s so modalities are
compared at equal time.

| Budget | EEG alone | Gaze alone | **EEG + gaze** |
|---|---:|---:|---:|
| 2 s | 0% | 3.9% | **18.6%** (3 wrong of 306) |
| 3 s | 2.9% | 3.9% | **29.1%** (3 wrong of 306) |
| 10 s | 25.8% | 4.9% | **68.3%** (10 wrong of 306) |

Fused minus EEG, participant bootstrap 95% CI: 2 s +18.6 [11.4, 28.1], 3 s +26.1 [18.2, 35.3], 10 s +42.5 [29.2, 53.6].
Gaze alone points at the right key 84% of the time by 3 s, but 42 of its 46 misses are one key off, so a safe gaze-only
policy rarely commits: EEG is what separates neighbours. With gaze shifted by 1-3 keys, fusion falls back to about
EEG-alone performance instead of failing.

Files: `evaluations/results/eeg_gaze_fusion/`, `docs/EEG_GAZE_FUSION.md`. Reproduce: `scripts/benchmark/study_h_fusion.py`.

## 5. Coordinator: NeMo Gym `unspoken_communication`

**Question.** Given the agents' candidates and quality flags, does the coordinator pick the right single action (offer,
ask, retake, ignore, switch input, correct) and never speak for the patient unconfirmed?

Scored by the environment's verifier on valid / right action / safety. Held-out set: same generator, seed 2026, every
task that also occurs in the 72-task eval set removed (44 remain, 10 categories). Nemotron 3.5 Lightning NVFP4 on vLLM,
Brev, temperature 0.

| Policy (44 held-out tasks) | Reward | Right action | Unsafe | p50 latency |
|---|---:|---:|---:|---:|
| Nemotron 3.5 Lightning 30B-A3B, alone | 0.939 | 81.8% | 0% | 281 ms |
| **Deployed: Nemotron + validator + fallback** | **0.985** | **95.5%** | **0%** | 278 ms |
| Rules alone (reference ceiling) | 1.000 | 100% | 0% | 0 |

The whole gain is in `none_of_these` (0.67 → 1.00): after the person rejects every reading, Nemotron re-asks; the
validator's history rule switches to spelling. Nemotron still makes 86% of the decisions. The rules share the
verifier's specification, so their 1.000 is a ceiling, not an independent baseline. Earlier, on a Mac through Ollama,
the same model took 4.6 s per decision; on vLLM it takes 278 ms.

Files: `evaluations/results/coordinator_heldout_brev.json`, `coordinator_brev.json`, `docs/BOUNTIES.md` §2.
Reproduce: `python evaluations/nemo_gym/run_local.py --seed 2026 --distinct-from 7 --out coordinator_heldout.json`.

## 6. Lips + voice on the same utterances

**Question.** The ICU is loud. When noise defeats the voice agent, does adding the lip agent, which hears nothing,
rescue the decision, and does running both cost latency?

**Method** (`scripts/benchmark/av_fusion_grid.py`).
- **Data.** GRID speakers s1 and s2: 100 utterances each, with real synchronised audio and video. Neither speaker was
  used to train either model.
- **Task.** Pick the true sentence among 10 that differ from it in exactly one word ("bin blue at **f** two now" vs
  "bin blue at **x** two now").
- **Agents, unchanged from the app.**
  - voice: Nemotron Speech Streaming en 0.6B, forced transducer score of each candidate (H100);
  - lips: the deployed Auto-AVSR checkpoint (not the GRID fine-tune), teacher-forced score (Mac, MPS).
- **Noise.** Only the audio is degraded: white noise, or babble (three utterances of the other speaker), at 10, 0 and
  -5 dB SNR. The video is never touched.
- **Fusion.** The two scores are treated as independent evidence (sum of tempered log-softmaxes).
- **Calibration.** Temperatures and one commit threshold per policy are fitted on the other speaker, pooled over all
  noise conditions (the system does not know the noise level), at a 2% wrong-commit target. Below the threshold the
  agent abstains. Both folds are reported together.

**Correct commits / wrong commits, of 200** (`fusion_pooled.json`):

| Audio | Voice alone | Lips alone | **Lips + voice** |
|---|---:|---:|---:|
| clean | 93.5% / 0 | 53.5% / 6 | **97.0% / 1** |
| white noise, 10 dB | 83.0% / 2 | 53.5% / 6 | **93.0% / 3** |
| white noise, 0 dB | 43.0% / 0 | 53.5% / 6 | **88.5% / 6** |
| white noise, -5 dB | 19.5% / 2 | 53.5% / 6 | **84.5% / 8** |
| babble, 10 dB | 73.5% / 2 | 53.5% / 6 | **93.0% / 2** |
| babble, 0 dB | 8.0% / 12 | 53.5% / 6 | **76.0% / 9** |
| babble, -5 dB | 0.0% / 21 | 53.5% / 6 | **65.0% / 12** |

- **Together beats both agents at every noise level**, and the gap widens as noise rises.
- **It is also safer than voice alone where it matters:** in loud babble, voice commits wrong words (21 at -5 dB);
  together, 12.
- **Ranking accuracy before any abstention** (top-1): clean 99.5% voice / 88.5% lips / 98.0% together; babble 0 dB
  46.0% / 88.5% / 89.5%. In clean audio, adding lips costs 1.5 points of top-1 but gains 3.5 points of correct commits,
  because agreement makes the system confident sooner.
- **Latency.** Voice 57 ms p50 (H100), lips 348 ms p50 (Mac). They run in parallel, so together is 348 ms p50, 366 ms
  p95: the slower agent, not the sum.
- **Caveats.** Wrong commits total 2.9% across all conditions (41 of 1,400) against the 2% target: the threshold fitted on
  one speaker does not fully hold on the other. GRID is a fixed 51-word grammar. Per-condition calibration (knowing the
  noise level, `fusion.json`) gives voice higher numbers but the same picture.

Reproduce: `av_fusion_grid.py cands`, `lips` (app venv), `voice` (NeMo venv), `fuse --pooled`.

## 7. EEG decisions in NeMo Gym (`neuro_comm`)

_Running: first Nemotron run on held-out Study F session 3._

## 8. Model improvement with NeMo Gym

Not claimed. No model weights were trained with NeMo Gym. The coordinator's gain (section 5) is validator code. The
coordinator task set's inputs also repeat across seeds (49 of 60 eval tasks had exact copies in a re-seeded set), so an
honest training split needs held-out phrases or categories, which did not fit the time.
