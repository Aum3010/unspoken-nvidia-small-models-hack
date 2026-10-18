# Final evaluation plan

> **Note for the team (P J, Yash).** `eval-consolidation` = `ui-speedups` (merged as is, no conflicts, 92 tests pass)
> plus evidence from `eeg-dataset-foundation` and `yash-lipreading-v2`, plus two new experiments: **E1** (lips + voice
> fused on the same GRID utterances) and **E4** (Nemotron on `neuro_comm`, the ALS EEG Gym environment). The coordinator
> Gym runs (72 eval + 44 held-out) and the serving/latency work on `ui-speedups` are used as they are, not re-run.
> - `docs/RESULTS.md` is the one-page results summary; its coordinator row uses the held-out Brev numbers from
>   `coordinator_heldout_brev.json`.
> - **Bounty wording to check:** `BOUNTIES.md` §2 ("Best Model Improvement With NeMo Gym") says, correctly, that no
>   weights changed: the gain is the validator. A judge reading "model improvement" may expect training. We have no
>   training run, so we should not claim one; keep the wording as it is.
> - Copied code needed two small fixes: `scripts/data/audit_eeg_datasets.py --skip-bnci` (the Study F audit crashed
>   with BNCI files present) and `evaluations/dataset_manifests/` (missing from the first copy).


**The claim to support.** Each agent was specialised with task-specific data, and that improved its accuracy and/or
latency on held-out people. Combining the agents beats the best single agent, without making the system slower.

**One metric everywhere.** A wrong word spoken for a patient is worse than a pause, so every row reports correct
commits, wrong commits, abstentions and latency, on people or sessions the agent never saw. Results go to
`docs/RESULTS.md`, one row each, with the dataset, split, baseline and caveat.

## Part 1. Each agent: task-specific data → measured gain (done; consolidated on this branch)

| Agent | Task-specific data | Accuracy | Latency | Source |
|---|---|---|---|---|
| Voice | Own 20-phrase recordings (3 speakers, 235 clips in 4 conditions, 71 negatives); TORGO | Dysarthric 40.0% → 61.1% accepted, 0 wrong | 43 ms p50 | `asr_eval/results/` |
| Lips (fine-tune) | GRID, 6-31 training speakers | WER 80.7% → 4.6% on unseen speakers | 289 → 143 ms decode | `vsr_finetune.json` |
| Lips (personal) | 90 webcam takes of 22 phrases; MIRACL-VC1 | top-1 72.0% → 96.1% (MIRACL) | enrollment match 0.45 s per clip (Mac) | `vsr_enrollment_miracl.json` |
| EEG (ALS) | Frozen per-flash evidence, bigP3BCI Study F | Wrong 24 → 1 per 120, throughput not lower | 16.5 → 9.2 s median per attempt | `eeg_fast_selection/` |
| Coordinator | Synthetic NeMo Gym task set (60 tasks) | Correct action 80% → 100%, 0 unsafe | 4.6 s (Ollama, Mac) | `coordinator.json` |
| Gaze | none | not measured | | stated as a gap |

## Part 2. Combined agents beat the best single one (new runs today)

### E1. Lips + voice on the same utterances (critical path)
`scripts/benchmark/av_fusion_grid.py`
- **Data:** GRID s1+s2, 200 clips, real synchronised audio and video. Neither speaker trained either model.
- **Task:** pick the true sentence from 10 candidates that differ from it in exactly one word.
- **Conditions:** the audio clean, with white noise, or with babble (three other-speaker clips) at 10, 0 and -5 dB SNR.
  The video is never degraded.
- **Agents:**
  - voice: Nemotron Speech Streaming 0.6B, forced phrase scoring (Brev, NeMo venv)
  - lips: deployed Auto-AVSR, phrase scoring (Mac MPS)
  - fused: independent evidence, temperatures fitted on the other speaker
- **Decision:** commit at a 2% wrong-commit target fitted on the other speaker, otherwise abstain. Two folds.
- **Report:** correct / wrong / abstain per condition for voice, lips and fused. Latency p50/p95 per agent; fused is the
  slower of the two running in parallel.
- **Expected shape, not assumed:** voice falls in noise, lips do not move, fused at or above both. Reported as measured.

### E2. EEG + gaze (done)
Study H: correct within 3 s at ~1% wrong, 2.9% (EEG) and 3.9% (gaze) → 29.1% fused, CI [18.2, 35.3]. Non-ALS.

### E3. Coordinator: accuracy and latency in NeMo Gym (done on `ui-speedups`)
44 held-out tasks (duplicates of eval tasks removed), Nemotron 3.5 Lightning NVFP4 on vLLM, Brev: Nemotron alone right
action 0.82 → deployed (validator + fallback) 0.95, 0 unsafe, 278 ms p50; Nemotron decides 86% itself
(`coordinator_heldout_brev.json`). Not re-run.

### E4. EEG decisions in NeMo Gym, first Nemotron run (Brev)
`scripts/benchmark/run_neuro_comm_eval.py` on Study F session 3 (120 selections, 10 ALS participants):
Bayes, Bayes + abstain, Nemotron 30B, hybrid (Nemotron called only when Bayes is unsure). Report: reward, correct,
wrong, abstain, model calls, end-to-end latency, participant bootstrap CIs.

### E5. Whole system under load (done on `ui-speedups`)
NVIDIA MPS on one GPU: with every agent busy, voice 244 → 75 ms p50, Kokoro 79 → 30 ms, vLLM 352 → 336 ms
(`latency_brev_ui_mps{0,1}.json`). Not re-run.

## Box layout (2x H100 80GB, `breakable-tomato-hippopotamus`)

| GPU | Process |
|---|---|
| 0 | vLLM: Nemotron 3.5 Lightning 30B-A3B BF16, port 8001 (E4) |
| 1 | NeMo: Nemotron Speech Streaming 0.6B (E1 voice) |

Lips (E1) ran on the Mac: 200 clips, top-1 88.5%, 348 ms p50 (MPS).

## Not claimed

- NeMo Gym model improvement (no training run: no time for an honest held-out design; the coordinator task set's
  inputs repeat across seeds, 49 of 60 eval tasks).
- Gaze accuracy (no measurement).
- Study H as ALS evidence.
- The GRID fine-tune as deployed.
