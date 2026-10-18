# Study F EEG stopping benchmark

Question: on recorded P300 data from ALS participants, how much evidence does a classical stopping rule need, and what is
left for a learned policy to improve? This document covers the **frozen-evidence foundation and the classical baselines
only**. No System 1, NeMo Gym training or Nemotron orchestration is involved. All numbers come from real bigP3BCI Study F
recordings (10 ALS participants); nothing is simulated.

## 1. What is evaluated

- **Primary evaluation ("Static Test", session 3):** 2 runs x 6 selections x 10 participants = **120 selections**, fixed length (14 repetitions, 126 flashes). One of them (F_21) was recorded with only 4 repetitions.
- **Secondary ("CBCol", session 3):** 18 selections x 10 = **180 selections**, reported separately (training-style runs, no feedback shown to the participant).
- **Calibration:** sessions 1-2 (CBCol + Static runs, 60 selections per participant, 600 total). Nothing from session 3 is used to fit any filter, feature transform, classifier, score calibration, temperature or threshold.
- **Not used:** Dyn / DynBigram. Their evidence was truncated online and DynBigram used a language prior, so they cannot serve as counterfactual stopping benchmarks.

Replay constraint: a policy can stop earlier than the recording but can never request flashes that were not recorded. If it has not stopped when the recording ends, the evidence is *exhausted* and it must commit its best guess or abstain.

## 2. Pipeline (all in `services/eeg/study_f/`, scripts in `scripts/benchmark/`)

1. **Stimulus semantics** (see `docs/EEG_DATASET_AUDIT.md` section 9). A flash lights 3-5 of 36 characters; a repetition is a random partition of all 36 (10 flashes, then 8); 14 repetitions = 7 paper-sequences of 18 flashes = 126. The character-to-flash mapping is *measured from the 36 per-flash character flags*, never derived from `StimulusCode` or row/column ids, and is validated against the recorded target indicator in all 153,202 flashes.
2. **Preprocessing (fixed, no fitted parameters):** causal 4th-order Butterworth 1-20 Hz, epoch -0.1..+0.7 s around `StimulusBegin`, baseline-correct, average in blocks of 4 samples (64 Hz). 16 EEG channels.
3. **Flash decoder, fitted per participant on sessions 1-2 only.** Primary (named before looking at results): `xdawn_riemann_lr`, the repo's existing XdawnCovariances -> Riemannian tangent space -> logistic regression (`services/eeg/p300.py::_riemann`). Comparison: shrinkage LDA on the flattened epoch.
4. **Out-of-fold calibration scores:** train on one calibration session, score the other; the final decoder (both sessions) scores session 3 once and is then frozen. Pooled-variance Gaussians fitted to the out-of-fold scores turn a score into a log-likelihood ratio `l_j = log p(s_j | target flash) - log p(s_j | non-target)`.
5. **Frozen evidence** (`data/evidence/study_f/`, gitignored, rebuilt by one command, SHA-256s in `manifest.json`):
   - `flash_evidence.csv.gz`: per flash `score`, `llr`, lit characters, recorded onset time; **no labels**.
   - `rep_evidence_<decoder>.npz`: per selection and repetition, a 36-vector of character log-evidence increments, last-flash timestamp and flash count (NaN where not recorded).
   - `ground_truth.csv`: verifier-only target index per selection. Policies never read it; the scorer does, after a decision.
6. **Baselines** (`run_study_f_baselines.py`) on the identical frozen evidence.

### Bayesian update rule

Under hypothesis "character c is the target", flash j is a target flash iff c was lit in it, so
`log p(data | c) = const + sum over flashes j with c lit of l_j`. After r repetitions `L_c(r)` is the sum of the per-repetition increments and
`posterior(c) = softmax(alpha * L(r))` with a uniform prior over the 36 characters. Flashes are treated as conditionally independent (naive Bayes). `alpha = 1` is the plain Bayesian rule; the *calibrated* rule uses an `alpha` fitted per participant by minimising character NLL on out-of-fold calibration selections (at every repetition count). Stopping is evaluated at repetition boundaries (the frozen unit); flash-level increments are in `flash_evidence.csv.gz` for finer policies later.

### Policies

| Policy | Rule |
|---|---|
| `fixed_k` | consume k repetitions (k = 1..14), commit argmax; fewer if recording shorter (exhausted) |
| `bayes_tau<t>` | stop at first repetition with max posterior >= t (alpha = 1); on exhaustion commit argmax |
| `calibrated_tau<t>` | same with calibrated alpha |
| `*_abstain_tau<t>` | as above, but if evidence is exhausted below t, **abstain** |
| `fixed_k_abstain_cal` | consume k repetitions, abstain if calibrated confidence is below the calibration-chosen threshold |
| `SELECTED_*` | threshold = smallest grid value whose pooled out-of-fold calibration error <= 5%; none reaches it (weak participants), so the largest value 0.9999 was chosen. The curves over all thresholds are the real result. |

### Metrics

Per selection: committed, correct, abstained, exhausted, flashes and repetitions consumed, **acquisition time** (recorded last-flash onset minus first onset, plus one 0.25 s SOA), **decision latency** (last-flash onset minus first onset plus the 0.7 s epoch tail; compute time is not included), correct characters per minute = correct / total decision latency (`ccpm`) and with the recorded median 3.5 s inter-selection pause (`ccpm_with_pause`). Accuracy counts abstentions as not correct; incorrect commits and abstentions are counted separately.

Uncertainty: metrics are computed per participant, then averaged over the 10 participants; the 95% interval is a percentile bootstrap **over participants** (2,000 resamples). Selections from one participant are not treated as independent. With 10 participants and wide between-participant spread the intervals are wide, and differences between policies should be read from the paired per-participant table, not from overlapping bands.

## 3. Results (real data; `evaluations/results/eeg_stopping/`)

Mean over 10 participants, Static Test session 3, decoder `xdawn_riemann_lr` (95% participant-bootstrap interval for accuracy):

| Policy | Accuracy | 95% interval | Repetitions | Flashes | Latency (s) | Correct chars/min | Incorrect commits | Abstentions |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fixed 4 reps | 0.558 | 0.34-0.78 | 4.0 | 36 | 9.5 | 3.55 | 53 | 0 |
| fixed 7 reps (paper's static default) | 0.608 | 0.38-0.83 | 7.0 | 64 | 16.4 | 2.23 | 47 | 0 |
| fixed 10 reps | 0.692 | 0.48-0.88 | 9.95 | 90 | 22.8 | 1.82 | 37 | 0 |
| fixed 14 reps | 0.758 | 0.55-0.94 | 13.9 | 125 | 31.8 | 1.44 | 29 | 0 |
| Bayesian, tau 0.9 | 0.717 | 0.52-0.89 | 8.2 | 74 | 19.0 | 3.98 | 34 | 0 |
| Bayesian, tau 0.99 | 0.758 | 0.54-0.93 | 9.4 | 85 | 21.6 | 3.22 | 29 | 0 |
| calibrated, tau 0.99 | 0.758 | 0.55-0.94 | 9.2 | 83 | 21.3 | 3.23 | 29 | 0 |
| calibrated + abstain, tau 0.9 | 0.600 | 0.35-0.83 | 8.1 | 73 | 18.8 | 3.71 | **6** | 42 |
| calibrated + abstain, tau 0.99 | 0.575 | 0.32-0.82 | 9.2 | 83 | 21.3 | 2.84 | **0** | 51 |

CBCol (secondary) and the LDA decoder give the same picture (LDA is a little better: Static fixed-14 0.792, Bayesian tau 0.99 0.792 in 9.0 repetitions); see `aggregate.csv`, `per_participant.csv`.

What this shows (and what it does not):
- **Bayesian stopping reaches the fixed-14 accuracy with about a third fewer repetitions** (9.4 vs 13.9, same 0.758; latency 21.6 s vs 31.8 s) and gives 2.2x the throughput. At equal repetitions it is more accurate than fixed (8.2 reps: 0.717 vs fixed-10: 0.692 at 9.95). This is the headroom a learned policy has to beat, and it is a real but modest one.
- **Calibration adds almost nothing here**: the fitted `alpha` is about 1 for most participants (0.77-1.40), the naive-Bayes posterior is already roughly calibrated out-of-fold (ECE 0.03) and slightly under-confident on session 3 (mean confidence 0.713 vs accuracy 0.758, ECE 0.07). Calibrated and plain Bayes curves overlap.
- **Abstention can buy zero or near-zero wrong characters** (6 incorrect commits in 120 at tau 0.9; 0 at 0.99) but at the cost of abstaining on roughly 35-45% of selections, concentrated in weak participants.
- **The aggregate hides a two-population structure.** Per-participant fixed-14 accuracy: F_08, F_05, F_20, F_21 = 1.00; F_03 0.92; F_24 0.92; F_23 0.83; F_25 0.58; F_06 0.33; **F_07 0.00** (flash AUC 0.51: no usable P300). Policies cannot fix F_07; they can only abstain or escalate. A learned policy's value is likely in deciding *who/when to escalate*, not in squeezing the strong participants.
- **Uncertainty is large.** Only 12 Static selections per participant in the evaluation set and 10 participants: the participant-bootstrap intervals for accuracy span ~0.4 wide. No claim here is statistically established beyond "Bayesian stopping is not worse than fixed at lower evidence", which holds for most individual participants.

Figures: `curves_evaluation_<decoder>.png` (accuracy vs flashes, incorrect commits vs correct characters/min, accuracy vs latency), `curves_evaluation_secondary_<decoder>.png`, `per_participant_fixed_<decoder>.png`.

## 4. Known limitations

- Truncated recordings: one primary-evaluation selection (F_21 SE003 Static01 trial 0) has 4 repetitions; others are in calibration/secondary. They are kept and counted as exhausted.
- `SELECTED_*` thresholds degenerate to 0.9999 because the pooled out-of-fold error never reaches 5% (F_06, F_07, F_25 dominate the error). A per-participant or cost-weighted selection rule would be a design choice for the next phase, not made here.
- The flash-to-flash independence assumption inflates raw posterior confidence; the calibrated `alpha` is a single scalar per participant.
- Stopping is at repetition boundaries; real-time compute latency, display-to-EEG latency and the 0.7 s epoch tail are the only latency terms (the last is an assumption: the epoch window).
- No BNCI2014-008 results (files unreachable from this environment).
- The decoder comparison was not used for model selection; `xdawn_riemann_lr` was designated primary in advance, and LDA happens to be slightly more accurate.

## 5. Reproduction

```bash
uv pip install --python .venv -r requirements.txt pytest      # pandas, pyedflib, pyriemann, matplotlib included
for p in F_03 F_05 F_06 F_07 F_08 F_20 F_21 F_23 F_24 F_25; do
  .venv/bin/python scripts/data/fetch_bigp3bci.py --study F --participant $p; done          # ~1.4 GB, resumable, SHA-256 checked
.venv/bin/python scripts/data/audit_eeg_datasets.py --out data/processed/audit              # ~40 s
.venv/bin/python scripts/benchmark/build_study_f_evidence.py                                # ~3.5 min -> data/evidence/study_f/
.venv/bin/python scripts/benchmark/run_study_f_baselines.py                                 # ~3 min  -> evaluations/results/eeg_stopping/
.venv/bin/python scripts/benchmark/plot_study_f.py
.venv/bin/python -m pytest -q                                                               # 48 tests
```
