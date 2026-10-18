# EEG x gaze fusion on Study H (bigP3BCI): time to a reliable selection

**Question.** Study F showed P300 EEG alone is not reliable within 1-3 s. Does fusing a co-recorded eye tracker with the
same flash-level Bayesian EEG evidence change that? And what happens as the gaze gets worse?

**Answer (16 non-ALS participants, held-out runs, real co-recorded data).**
- With gaze, about 1 in 5 selections commits correctly within 2 s, and 1 in 3.5 within 3 s, at about 1% wrong (risk
  target 2%). EEG alone commits almost none within 3 s.
- Gaze alone points at the right key 68% of the time after 0.5 s and 84% after 3 s. Almost every miss lands on an
  adjacent key, so a risk-controlled gaze-only policy commits rarely. EEG is what separates neighbouring keys.
- When gaze is degraded with a synthetic per-selection offset of 1-3 key widths, fusion falls back to roughly EEG-alone
  performance, staying at or slightly above it. At small offsets it makes a few more wrong commits than EEG at short
  budgets.
- **Caveat:** Study H participants are not ALS. This supports the fusion mechanism, not ALS performance. Study F remains
  the ALS evidence for EEG alone ([EEG_FAST_SELECTION.md](EEG_FAST_SELECTION.md)).

## Data

bigP3BCI Study H (Kalika, Collins, Caves and Throckmorton, *J. Neural Eng.* 14:056010, 2017), CC-BY 4.0. It has
16 participants and one session each. The setup:
- 9x8 checkerboard grid (72 keys), 6 keys per flash, 125 ms SOA.
- 16 EEG channels plus binocular eye-tracker gaze (normalised screen X/Y with validity flags) in the same EDF at 256 Hz.

**Used for the offline experiment:** the Train (copy-spelling, full-length) runs.
- Most participants have 3 runs × 6 selections × 168 flashes (14 repetitions).
- H_08 recorded 6 runs.
- 306 evaluated selections in total.
- One H_04 Train trial has no `CurrentTarget` label (audit ERROR); it is skipped and listed in the cache.

**Not used for offline policies:** the Test runs (CBGazeNo, CBGaze01, CBGaze10, CBGazeReal).
- They were stopped online, so their evidence is truncated.
- Their recorded online outcomes are reported below as the original study's reference.
- What Gaze01 and Gaze10 mean is not stated in the dataset description; unverified.

Fetch: `scripts/data/fetch_bigp3bci.py --study H --participant H_xx` (checksummed against the mirror's SHA256SUMS).

## Method (no label or gaze-model leakage)

**Splitting.** The outer loop is leave-one-block-out over 3 blocks of Train runs: one run per block, or two for H_08.
Everything is fitted on the other two blocks.
- **EEG:** the `ens_lda_riemann` decoder from the Study F work.
  - Out-of-fold scores between the two training blocks give the Gaussian LLR and the evidence temperature alpha.
  - The final decoder is fitted on both training blocks and scores the held-out block.
  - Evidence per key is the summed LLR of the flashes that lit it; a flash counts once its 0.7 s window has ended.
- **Gaze:** a per-participant affine map from mean screen gaze to grid (column, row), fitted on the training blocks.
  - Evidence is the log-likelihood of the cumulative mean valid gaze since the first flash, under each key's location.
  - The likelihood is an isotropic Gaussian mixed with a uniform lapse over the grid.
  - sigma and the lapse are fitted by negative log posterior on the inner (cross-block) calibration trajectories.
- **Fused:** alpha·L_EEG + L_gaze, i.e. conditionally independent evidence, softmax over 72 keys.

**Policies.** Commit when the max posterior ≥ tau; otherwise yield at the deadline.
- tau and the deadline are chosen on the calibration trajectories: wrong ≤ floor(risk·n), maximise correct/min.
- They are pooled across participants per fold.
- The "budget" frontier fixes the deadline (1, 2, 3, 5, 10 s) and chooses only tau, so modalities are compared at
  equal time.

**Timing.** Decision time runs on a 125 ms grid from the first flash, using recorded sample timestamps. Gaze uses only
valid samples up to t.

**Stress test.** A random offset (SD 0.5, 1, 1.5, 2 and 3 key widths, one draw per selection) is added to the real mapped
gaze in both calibration and evaluation. It is a synthetic perturbation of real gaze, standing in for drifting or poorly
calibrated gaze. It is not ALS data.

**Disclosure.**
- The first run used a pure Gaussian gaze likelihood (lapse 0).
- Its gaze-alone policies committed almost nothing, and fused commits were less efficient. That run is kept in
  `evaluations/results/eeg_gaze_fusion/gaussian_only_first_run/`.
- The lapse model was then added after seeing that run. It nests the Gaussian (lapse 0 is in the grid) and is selected
  only by calibration likelihood.
- The fixed-budget frontier was also added after the first runs, as a fairer comparison than correct/min, which rewards
  yielding fast.

## Results

### Argmax accuracy at fixed decision times (no stopping rule; information, not a usable policy)

| t (s) | 0.5 | 1 | 1.5 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|---|
| EEG | 6.5% | 8.2% | 13.4% | 19.9% | 25.8% | 41.5% | 66.3% |
| gaze | 67.6% | 74.2% | 78.4% | 80.4% | 83.7% | 85.0% | 85.3% |
| fused | 67.6% | 74.2% | 80.1% | 82.4% | 85.0% | 88.2% | 93.5% |

Chance is 1.4% (1/72). Gaze plateaus at about 85%. Of the 46 selections whose final gaze sits on the wrong key, 42 are
one key off. Gaze misses by participant range from 0% (H_01, H_03, H_09, H_11, H_12) to 67% (H_07).

### Correct and wrong commits by time budget (risk target 2%, real gaze; 306 selections; the rest yield)

Each cell gives correct commits as a share of selections, then the count of wrong commits.

| budget | 1 s | 2 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|---|
| EEG | 0% / 0 | 0% / 0 | 2.9% / 6 | 12.1% / 9 | 25.8% / 11 |
| gaze | 3.3% / 5 | 3.9% / 5 | 3.9% / 5 | 4.6% / 5 | 4.9% / 5 |
| **fused** | 6.2% / 3 | 18.6% / 3 | **29.1% / 3** | 50.3% / 7 | 68.3% / 10 |

**Participant-level bootstrap.** These are 95% CIs from 2000 resamples of the 16 participants, for fused minus EEG
correct commits:
- 2 s: +18.6 points [11.4, 28.1]
- 3 s: +26.1 points [18.2, 35.3]
- 10 s: +42.5 points [29.2, 53.6]

The risk target held on held-out data at budgets up to 3 s (3/306 wrong). At 10 s it was slightly exceeded: 10/306,
3.3%.

### Degraded gaze (fused, risk target 2%; synthetic offset added to real gaze)

| offset SD (keys) | 1 s | 2 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|---|
| 0 (real) | 6.2% / 3 | 18.6% / 3 | 29.1% / 3 | 50.3% / 7 | 68.3% / 10 |
| 0.5 | 4.2% / 8 | 11.8% / 9 | 13.7% / 6 | 27.8% / 9 | 46.1% / 12 |
| 1 | 1.3% / 9 | 3.3% / 7 | 8.5% / 7 | 19.9% / 9 | 34.3% / 10 |
| 2 | 0.3% / 0 | 2.0% / 8 | 4.2% / 2 | 15.0% / 7 | 30.1% / 6 |
| 3 | 0% / 0 | 0.7% / 4 | 5.2% / 7 | 14.1% / 8 | 27.8% / 8 |
| EEG alone | 0% / 0 | 0% / 0 | 2.9% / 6 | 12.1% / 9 | 25.8% / 11 |

As gaze degrades, correct commits fall smoothly toward EEG-alone. The cost is a few extra wrong commits at short budgets
when gaze is moderately wrong (0.5-1 key). A per-selection offset makes gaze confidently wrong in a way the pooled
calibration does not fully absorb.

### Original study's online results (Test runs, recorded; reference only)

| condition | online accuracy | flashes to decision | stimulus time |
|---|---|---|---|
| CBGazeNo | 84.6% | 65.9 | 8.2 s |
| CBGaze01 | 91.2% | 30.9 | 3.9 s |
| CBGaze10 | 88.9% | 49.8 | 6.2 s |
| CBGazeReal | 86.0% | 12.5 | 1.6 s |

Means are over participants, and the stimulus time excludes the final 0.7 s response window. These are the 2017 study's
own online fusion and stopping results. They are not produced by this pipeline and do not use the same risk control, so
they are not a like-for-like comparison. They do show independently that real gaze cut online selection time by about
5x at similar accuracy.

## What this means for Unspoken

- The "everything at once" design is supported. Gaze proposes the neighbourhood in under a second, and EEG settles
  which neighbour.
- When gaze becomes unreliable, the same shared belief degrades toward EEG alone instead of failing. No mode switch is
  needed.
- At about 1% wrong, this mechanism still yields on most selections within 3 s. In the product, a yield should mean
  "keep looking": evidence keeps accumulating rather than restarting. That is not measured here.

## Reproduce

```
/home/claude/venv-eeg/bin/python scripts/data/fetch_bigp3bci.py --study H --participant H_01   # ... H_16
/home/claude/venv-eeg/bin/python scripts/benchmark/study_h_fusion.py build      # ~25 s per participant
/home/claude/venv-eeg/bin/python scripts/benchmark/study_h_fusion.py evaluate   # ~15 min
/home/claude/venv-eeg/bin/python scripts/benchmark/plot_study_h_fusion.py
```

Outputs are in `evaluations/results/eeg_gaze_fusion/`. Tests are in `evaluations/dataset_tests/test_study_h_fusion.py`.
