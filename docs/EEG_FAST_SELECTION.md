# EEG fast selection: how quickly can a character be extracted from the recorded EEG?

Study F (bigP3BCI, 10 ALS participants, 36-character speller). All numbers are from real recordings. Sessions 1-2 were
used for every fit and choice. Session 3 Static Test (120 selections, 12 per participant) was scored once, after all
choices had been written to disk. No synthetic EEG, no language prior, no other modality.

## Answer

- **Under 3 s, nothing is reliable.** No policy commits a correct character within 1, 2 or 3 s unless it accepts roughly
  half of its commits being wrong. At the lowest swept threshold (tau 0.2), 14% of characters are correct by 3 s with
  58 wrong commits per 120. Even with hindsight, the argmax is already settled on the target by 3 s for only 16-20% of
  selections, and a policy cannot tell which ones (`oracle_information_times.csv`).
- **At about 1 wrong commit per 120 attempts, the best calibration-chosen configuration gets a correct character in a
  median of 8.5 s.** The configuration is the ensemble decoder with flash-level stopping, participant-calibrated alpha,
  threshold and deadline, at a 2% risk target. It is correct on 63% of attempts and yields on 36%. The median is 9.2 s
  per attempt and 17% of characters are correct within 5 s. Throughput is 3.13 correct characters/min against 2.11 for
  the original 21.6 s baseline, with 1 wrong commit instead of 29.
- **It depends on the person.**
  - Five strong participants (F_05, F_08, F_20, F_21, F_24) get 0-1 wrong commits in 12 at a median of 4.6-13.4 s per
    character. F_08 gets 12/12 correct within 5 s, at a median of 4.6 s.
  - Four weak participants (F_06, F_07, F_23, F_25) cannot be served reliably by this EEG. The policy yields on most of
    their selections. For F_07 (no usable P300, AUC 0.48-0.51) it never commits and yields at 0 s.
  - A single "N seconds per character" claim for ALS users would be false.
- **Where the gain comes from.**
  - (1) Yielding instead of committing on exhaustion removes most wrong characters and the 31.9 s worst cases.
  - (2) A better early-evidence decoder, selected on sessions 1-2, helps every policy. With the original stopping rule
    it alone moves accuracy from 0.758 to 0.800 and mean latency from 21.6 s to 19.1 s.
  - (3) Flash-level instead of repetition-level stopping saves 1-4 s of median latency in commit policies. At matched
    risk it is not consistently better (xdawn: flash 2.18 vs repetition 2.43 correct/min; ensemble: 3.13 vs 3.03).

## Setup

| Item | Choice |
|---|---|
| Evidence | Frozen per-flash classifier scores and LLRs (`data/evidence/study_f/flash_evidence.csv.gz`). Calibration rows are out-of-fold: train on one calibration session, score the other. |
| Accumulation | `L_c(k) = sum of LLRs of flashes 1..k that lit c`; `p = softmax(alpha * L)`, uniform prior over all 36 characters. A character not yet lit keeps `L = 0`, so a decision is possible after any flash, including partial flash groups (`services/eeg/study_f/flash_stopping.py`). |
| Decision time | `t_dec(k) = onset_k - onset_1 + 0.7 s`, from recorded stimulus timestamps. Flash k's full 0.7 s response window must exist before it can be used. A yield at a deadline costs the deadline. |
| CPU time | Kept separate from `t_dec`. The posterior update is 8 µs per flash. Decoder scoring per epoch is 1.5 ms (LDA), 2.7 ms (xDAWN+Riemann) and 2.7 ms (ensemble) on this container's CPU. Both are negligible next to the 250 ms stimulus interval. |
| Policy | Commit the argmax at the first flash with `pmax >= tau`. If none occurs before `deadline`, or the recording runs out, **abstain/yield**. Granularity is every flash, or repetition ends only (for comparison). |
| Calibration (sessions 1-2) | `alpha` fitted by NLL over every flash prefix, globally or per participant. `(tau, deadline)` chosen from a grid (17 tau values, deadlines 1-25 s and none) to maximise calibration correct/min, subject to calibration wrong commits <= floor(risk x n). n = 60 per participant, 600 pooled. Ties go to the faster setting. If no feasible setting gets any correct character, the policy never commits and yields at 0 s. |
| Families | `flash_participant` (participant alpha, tau, deadline), `rep_participant` (same at repetition ends), `flash_global` (pooled alpha, tau, deadline), `flash_hybrid` (participant alpha, pooled tau and deadline). Risk targets 0, 2%, 5%, 10% wrong commits per attempt. |
| Baseline | The original `bayes_tau0.99` (repetition ends, alpha 1, commit on exhaustion), recomputed from the frozen repetition evidence. It reproduces 0.758 accuracy, 29 wrong and 21.6 s mean latency (test `test_baseline_reproduces_original_benchmark`). |
| Throughput | Correct characters / total decision time of all 120 attempts, including wrong commits and yields. The earlier doc averaged per participant: baseline 3.22 by that definition (`mean_participant_ccpm` column). Both are in `aggregate.csv`. |

### Experiment 2: early-information decoders (selected on sessions 1-2 only)

Candidates were small regularized linear ERP models on the same epochs (`scripts/benchmark/study_f_early_decoders.py`):
- xDAWN spatial filters + shrinkage LDA
- shrinkage LDA on a compact 0.1-0.6 s, 32 Hz window
- the mean of the two existing decoders' fold-standardized scores

The pre-registered criterion was the mean log posterior of the target over flash prefixes 1-36 (the first four
repetitions, at most 9.5 s), out-of-fold on calibration (chance is -3.58):

| Decoder | Out-of-fold flash AUC | Early log posterior |
|---|---:|---:|
| ens_lda_riemann (new) | 0.808 | **-2.169** |
| compact_slda (new) | 0.798 | -2.274 |
| shrinkage_lda (existing) | 0.798 | -2.308 |
| xdawn_riemann_lr (existing, designated primary) | 0.773 | -2.328 |
| xdawn_slda (new) | 0.769 | -2.469 |

The ensemble beat both existing decoders on calibration, so it was the only new model scored on session 3
(`early_decoder_selection.json`). It is not a new architecture: it combines the two existing decoders.

## Results: session 3 Static, 120 attempted selections

Latency is in seconds per attempted selection (commit or yield). "≤x s" is the fraction of the 120 attempts that are
correct commits made within x s.

| Decoder | Policy | Acc | Wrong /120 | Abstain /120 | Selective acc | Lat mean | median | p90 | p95 | ≤1 s | ≤2 s | ≤3 s | ≤5 s | Correct/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xdawn_riemann_lr | **baseline_rep_bayes_tau0.99_commit** | 0.758 | 29 | 0 | 0.758 | 21.6 | 25.4 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.07 | 2.11 |
| xdawn_riemann_lr | flash_bayes_tau0.99_commit | 0.758 | 29 | 0 | 0.758 | 20.9 | 21.8 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.07 | 2.18 |
| xdawn_riemann_lr | rep_participant_risk0.02 | 0.592 | 3 | 46 | 0.959 | 14.6 | 11.9 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.14 | 2.43 |
| xdawn_riemann_lr | flash_participant_risk0 | 0.558 | 1 | 52 | 0.985 | 11.1 | 9.1 | 27.6 | 31.9 | 0.00 | 0.00 | 0.00 | 0.10 | 3.00 |
| xdawn_riemann_lr | flash_participant_risk0.02 | 0.567 | 1 | 51 | 0.986 | 15.6 | 10.4 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.12 | 2.18 |
| xdawn_riemann_lr | flash_participant_risk0.05 | 0.592 | 8 | 41 | 0.899 | 14.1 | 9.4 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.17 | 2.51 |
| xdawn_riemann_lr | flash_global_risk0.02 | 0.408 | 0 | 71 | 1.000 | 8.5 | 10.0 | 10.0 | 10.0 | 0.00 | 0.00 | 0.00 | 0.17 | 2.89 |
| xdawn_riemann_lr | flash_hybrid_risk0.02 | 0.433 | 0 | 68 | 1.000 | 8.4 | 10.0 | 10.0 | 10.0 | 0.00 | 0.00 | 0.00 | 0.15 | 3.08 |
| ens_lda_riemann | baseline_rep_bayes_tau0.99_commit | 0.800 | 24 | 0 | 0.800 | 19.1 | 16.4 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.10 | 2.51 |
| ens_lda_riemann | flash_bayes_tau0.99_commit | 0.792 | 25 | 0 | 0.792 | 18.3 | 15.4 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.10 | 2.60 |
| ens_lda_riemann | rep_participant_risk0.02 | 0.658 | 1 | 40 | 0.988 | 13.0 | 9.4 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.23 | 3.03 |
| ens_lda_riemann | flash_participant_risk0 | 0.592 | 1 | 48 | 0.986 | 9.2 | 8.3 | 20.0 | 20.0 | 0.00 | 0.00 | 0.00 | 0.14 | 3.85 |
| ens_lda_riemann | **flash_participant_risk0.02** | 0.633 | 1 | 43 | 0.987 | 12.1 | 9.2 | 20.0 | 31.7 | 0.00 | 0.00 | 0.00 | 0.17 | 3.13 |
| ens_lda_riemann | flash_participant_risk0.05 | 0.625 | 3 | 42 | 0.962 | 11.3 | 8.6 | 20.0 | 29.0 | 0.00 | 0.00 | 0.00 | 0.23 | 3.33 |
| ens_lda_riemann | flash_participant_risk0.1 | 0.675 | 10 | 29 | 0.890 | 12.4 | 7.7 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.24 | 3.26 |
| ens_lda_riemann | flash_global_risk0.02 | 0.467 | 0 | 64 | 1.000 | 8.1 | 10.0 | 10.0 | 10.0 | 0.00 | 0.00 | 0.00 | 0.22 | 3.45 |
| ens_lda_riemann | flash_hybrid_risk0.02 | 0.433 | 0 | 68 | 1.000 | 8.4 | 10.0 | 10.0 | 10.0 | 0.00 | 0.00 | 0.00 | 0.17 | 3.09 |
| shrinkage_lda | baseline_rep_bayes_tau0.99_commit | 0.792 | 25 | 0 | 0.792 | 20.8 | 18.4 | 31.9 | 31.9 | 0.00 | 0.00 | 0.00 | 0.06 | 2.28 |
| shrinkage_lda | flash_participant_risk0.02 | 0.592 | 3 | 46 | 0.959 | 13.3 | 10.8 | 25.0 | 25.0 | 0.00 | 0.00 | 0.00 | 0.11 | 2.66 |

All rows, both decoders, every risk target and the CBCol secondary session are in `aggregate.csv` and `per_participant.csv`.

### Per participant: baseline vs flash participant policy at a 2% risk target (12 selections each)

| Participant | cal AUC (xdawn / ens) | Baseline xdawn: acc, wrong, median s | xdawn flash p 2%: acc, wrong, yield, median s | Ensemble flash p 2%: acc, wrong, yield, median s, correct ≤5 s | Correct/min baseline → ens p 2% |
|---|---|---|---|---|---|
| F_03 | 0.81 / 0.87 | 0.92, 1, 28.7 | 0.50, 0, 6, 17.6 | 0.83, 1, 1, 10.8, 1/12 | 2.2 → 4.5 |
| F_05 | 0.87 / 0.89 | 1.00, 0, 9.4 | 1.00, 0, 0, 7.6 | 0.92, 0, 1, 9.1, 0/12 | 5.3 → 5.7 |
| F_06 | 0.60 / 0.62 | 0.33, 8, 31.9 | 0.08, 0, 11, 31.9 | 0.00, 0, 12, 20.0, 0/12 | 0.6 → 0.0 |
| F_07 | 0.51 / 0.48 | 0.00, 12, 31.9 | 0.00, 0, 12, 0.0 | 0.00, 0, 12, 0.0, 0/12 | 0.0 → 0.0 |
| F_08 | 0.98 / 0.99 | 1.00, 0, 5.0 | 1.00, 0, 0, 4.7 | 1.00, 0, 0, 4.6, 12/12 | 9.8 → 13.6 |
| F_20 | 0.86 / 0.90 | 1.00, 0, 13.9 | 0.92, 1, 0, 7.5 | 0.92, 0, 1, 7.1, 4/12 | 3.7 → 6.9 |
| F_21 | 0.90 / 0.91 | 1.00, 0, 9.4 | 1.00, 0, 0, 6.8 | 1.00, 0, 0, 7.1, 2/12 | 5.1 → 7.7 |
| F_23 | 0.71 / 0.75 | 0.83, 2, 31.9 | 0.42, 0, 7, 31.9 | 0.17, 0, 10, 20.0, 0/12 | 1.7 → 0.5 |
| F_24 | 0.86 / 0.90 | 0.92, 1, 19.7 | 0.75, 0, 3, 18.9 | 0.92, 0, 1, 13.4, 1/12 | 2.7 → 3.6 |
| F_25 | 0.64 / 0.78 | 0.58, 5, 31.9 | 0.00, 0, 12, 31.9 | 0.58, 0, 5, 30.3, 0/12 | 1.1 → 1.3 |

## What this does and does not show

- **Speed below about 5 s costs errors the data cannot avoid.** The Pareto figure's swept lines show that halving the
  median latency from about 10 s to 5 s raises wrong commits from about 10-15 to about 40-65 per 120. That is not a usable
  operating point for communication.
- **Matched risk is honoured only approximately on session 3.** At the 2% target (2.4 wrong per 120), the realised
  wrong counts were 0-3. At 5% (6 per 120), the xdawn participant policy made 8, and at 10% the ensemble made 10. With
  60 calibration selections per participant, per-participant thresholds are noisy, and some participants' session 3
  differed from sessions 1-2. F_23 was weak on calibration (26/60 correct) but better on session 3, so the policy
  yielded on selections the baseline got right (0.83 → 0.17).
- **Risk 0 had higher throughput than risk 2% for the ensemble (3.85 vs 3.13).** That is noise on 120 selections, not a
  reason to prefer it. The participant-bootstrap 95% intervals for correct/min overlap heavily: for example, 2.0-7.1
  for the ensemble at 2% risk, and 1.7-4.9 for the baseline (`aggregate.csv`, mean over participants).
- **Calibration at partial flash groups is acceptable.** ECE of the alpha-1 posterior is 0.03-0.06 within the first
  repetition, 1-4 repetitions and later bands (`calibration_ece_by_band.csv`). Participant-specific alpha did not improve
  it. The thresholds are risk-chosen on calibration anyway, so they do not rely on exact calibration.
- **Not shown.**
  - Live closed-loop speed: when the speller stops early, the stimulus sequence would change, and replay cannot test
    that.
  - Display and amplifier latency, which are unknown for Study F.
  - The cost of whatever modality takes over after a yield.
  - Inter-selection pauses (median 3.5 s in the recordings), which are not added here.

## Reproduce

```bash
PY=/home/claude/venv-eeg/bin/python
$PY scripts/benchmark/study_f_early_decoders.py cache      # epochs from data/raw (once, ~40 s)
$PY scripts/benchmark/study_f_early_decoders.py select     # sessions 1-2 decoder comparison -> early_decoder_selection.json
$PY scripts/benchmark/study_f_early_decoders.py finalize   # fit adopted decoder, score session 3 -> data/evidence/study_f_early/
$PY scripts/benchmark/run_study_f_fast_selection.py --extra-evidence data/evidence/study_f_early/flash_evidence_ens_lda_riemann.csv.gz
$PY scripts/benchmark/plot_study_f_fast_selection.py
$PY -m pytest -q evaluations/dataset_tests/test_flash_stopping.py
```

Outputs are in `evaluations/results/eeg_fast_selection/`:
- `aggregate.csv` and `per_participant.csv`
- `per_selection.csv.gz`
- `chosen_params_<decoder>.json` (all calibration choices)
- `descriptive_frontier_sweep.csv` (one tau swept on session 3, for the shape of the frontier only, not a deployable choice)
- `oracle_information_times.csv` (hindsight reference, not a policy)
- `prefix_accuracy_per_participant.csv` and `calibration_ece_by_band.csv`
- Figures: `pareto_latency_errors_accuracy.png`, `latency_cdf.png`, `prefix_accuracy_per_participant.png`, `per_participant.png`

The earlier benchmark (`docs/EEG_STOPPING_BENCHMARK.md`, `evaluations/results/eeg_stopping/`) is unchanged and is the
baseline.
