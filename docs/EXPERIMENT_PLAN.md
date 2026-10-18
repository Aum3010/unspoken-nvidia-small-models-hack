# Final evaluation plan

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

### E3. Coordinator: accuracy and latency in NeMo Gym (Brev)
`evaluations/nemo_gym/run_local.py` with `UNSPOKEN_LLM_BACKEND=openai`, 60 tasks:
- Nemotron 3.5 Lightning 30B-A3B BF16 on vLLM (GPU 0): raw and deployed (with validator)
- Nemotron 3 Nano 4B BF16 on vLLM (GPU 1): raw and deployed
- Report: reward, correct action, unsafe, p50/p95 latency.

The question: does the validator let a model 7x smaller hold 100% at lower latency?

### E4. EEG decisions in NeMo Gym, first Nemotron run (Brev)
`scripts/benchmark/run_neuro_comm_eval.py` on Study F session 3 (120 selections, 10 ALS participants):
Bayes, Bayes + abstain, Nemotron 30B, hybrid (Nemotron called only when Bayes is unsure). Report: reward, correct,
wrong, abstain, model calls, end-to-end latency, participant bootstrap CIs.

### E5. Whole system under load (done, reported as measured)
`latency_brev.json`: each agent alone vs all together on one GPU. Voice 55 → 304 ms p50, coordinator 241 → 304 ms,
lips p95 0.5 → 2.8 s. Two GPUs (LLM on one, agents on the other) is the fix. Re-measure on the 2x H100 if time allows.

## Box layout (2x H100 80GB)

| GPU | Process |
|---|---|
| 0 | vLLM: Nemotron 3.5 Lightning 30B-A3B BF16, port 8001 |
| 1 | vLLM: Nemotron 3 Nano 4B BF16, port 8002; then NeMo voice scoring (E1) |

## Timeline (minutes from box ready)

| Min | Box | Mac |
|---|---|---|
| 0-15 | uv venvs (vLLM, NeMo), model downloads, sync code + 200 GRID clips + Study F evidence | E1 lips (running) |
| 15-30 | E1 voice; E3 on both models | write the E1 section |
| 30-45 | E4 (Nemotron, hybrid) | E1 fuse; RESULTS.md |
| 45-60 | E5 re-measure if time; copy results back; **delete the instance** | slide |

## Not claimed

- NeMo Gym model improvement (no training run: no time for an honest held-out design; the coordinator task set's
  inputs repeat across seeds, 49 of 60 eval tasks).
- Gaze accuracy (no measurement).
- Study H as ALS evidence.
- The GRID fine-tune as deployed.
