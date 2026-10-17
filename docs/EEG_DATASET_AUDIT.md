# EEG dataset audit: BNCI2014-008 and bigP3BCI Study F

Scope: data foundation only. No model, policy, Gym or Nemotron work was done. Everything below was measured from files
on disk unless it is under "Unresolved". Audit run 2026-10-17.

## 1. Summary

| | BNCI2014-008 | bigP3BCI Study F |
|---|---|---|
| Files audited | **none** (not on disk; source host `bnci-horizon-2020.eu` unreachable from the audit container) | **270 EDF+ files, all 10 participants (F_03 F_05 F_06 F_07 F_08 F_20 F_21 F_23 F_24 F_25), 3 sessions each, 1,468,685,108 bytes, SHA-256 verified against `SHA256SUMS.txt`** |
| Audit status | adapter written and unit-tested on a hand-built recording only; **real-file audit not done** | real files opened and inspected |
| Grid | 6x6 row/column (existing code, unverified here) | **36 per-character channels labelled `<char>_<row>_<col>`, rows/cols 1-6**; the PhysioNet overview table says 9x8: **unresolved** |
| Stimulus unit | row or column of 6 chars | **a subset of 3-5 characters per flash** (no row/column decoding is possible) |
| EEG | 8 ch, 256 Hz | **16 ch, 256 Hz** (F3 Fz F4 T7 C3 Cz C4 T8 CP3 CP4 P3 Pz P4 PO7 PO8 Oz) |
| Eye tracking | no | **no** (no `ET*` channels in any Study F file) |

Two corrections to earlier plans:
- **There is no participant `F_01`.** Study F contains `F_03 F_05 F_06 F_07 F_08 F_20 F_21 F_23 F_24 F_25` (from `SHA256SUMS.txt`, 27 files each). All ten are now on disk and audited.
- **Study F is not the same protocol as BNCI.** Do not reuse `p300.decode_char` or `FLASHES_PER_CHAR=120` for it.

## 2. What was found and downloaded

Source: official PhysioNet bigP3BCI v1.0.0 (DOI 10.13026/0byy-ry86, CC BY 4.0), fetched from the documented S3 mirror
`https://physionet-open.s3.amazonaws.com/bigp3bci/1.0.0/` because `physionet.org` itself is blocked in the audit container.
Checksums come from the same mirror's `SHA256SUMS.txt`; all 27 files match. Provenance:
`evaluations/dataset_manifests/provenance_bigp3bci_F_03.json` (source, version, licence, per-file bytes and SHA-256, UTC time).

`data/raw/bigp3bci/StudyF/F_03/SE00{1,2,3}/{Train,Test}/<condition>/*.edf`

| Run type | Condition | Files per session | Notes |
|---|---|---:|---|
| Train | CBCol | 3 | fixed length |
| Test | Static | 2 | fixed length |
| Test | Dyn | 2 | online dynamic stopping |
| Test | DynBigram | 2 | online dynamic stopping with a bigram language prior |

9 files per session, 27 total, 4,120.75 s of recording (Train 1,962.0 s, Test 2,158.75 s).

## 3. Observed structure (F_03 detail; section 3b generalises to all ten)

- **Header:** `bigP3BCI v1.0.0 StudyF`, equipment g.tec gUSBamp x2, female, ALSFRS-R 21 in the patient field (`ALS_21`, identical in all 3 sessions). Dates are shifted by the dataset (start 2020-01-01).
- **Signals (62, all 256 Hz):** 16 `EEG_*`, 36 character flags, 10 BCI2000 state channels: `StimulusType StimulusBegin StimulusCode CurrentTarget SelectedTarget SelectedRow SelectedColumn PhaseInSequence FakeFeedback DisplayResults`.
- **Event codes (values actually seen):** `PhaseInSequence` in {0,1,2,3}; `StimulusType` in {0,1}; character flags in {0,1}. A selection is one `PhaseInSequence==2` segment; a flash is a rising edge of `StimulusBegin`.
- **Timing:** onset-to-onset 64 samples (250 ms, 13,189 of 13,189 within-trial gaps), flash 8 samples (31.25 ms, 13,351 of 13,351), a 960-sample (3.75 s) gap between trials, measured in the first Train file only.
- **`StimulusCode` is not a group id.** In a fixed-length run it is a running flash counter (max 756 = 6 trials x 126). Group membership comes from the 36 character flags.
- **Target label:** `CurrentTarget` (1-36, index into the character channels) is constant inside every selection; `StimulusType==1` coincides exactly with "target character shown" in all 13,351 flashes (checked per flash).
- **Repetitions (derived, 90 of 90 fixed-length selections agree):** a repetition is a maximal run of consecutive flashes with disjoint character sets whose union is all 36 characters. Fixed-length selections contain **14 repetitions in 126 flashes**, alternating 10 and 8 flashes. Every character is shown exactly 14 times.
- **Selections:** 162 in total.

| Run type / condition | Selections | Flashes per selection (min / median / max) | Complete repetitions | Online `SelectedTarget == CurrentTarget` |
|---|---:|---|---|---|
| Train / CBCol | 54 | 126 / 126 / 126 | 14 | n/a (no feedback, 0 values) |
| Test / Static | 36 | 126 / 126 / 126 | 14 | 33 / 36 |
| Test / Dyn | 36 | 15 / 27 / 59 | 1-6 (10, 32, 16, 9, 2, 3 selections for 1..6) | 34 / 36 |
| Test / DynBigram | 36 | 14 / 24 / 58 | 1-6 | 33 / 36 |

Dynamic runs (F_03) stop after 27.9 flashes on average (about 7.0 s of stimulation) and none reaches 126 flashes. Across all ten participants some Dyn/DynBigram selections run to the cap (see 3b).
Their stop is often mid-repetition, so evidence is flash-level, not repetition-level.

## 3b. All ten participants (audit of 270 files, 1,619 selections, 153,202 flashes; `participant_stats.csv`)

Same structure for every participant (nothing was assumed from F_03; every file was parsed and checked): 3 sessions x 9 runs
(CBCol Train x3, Static Test x2, Dyn Test x2, DynBigram Test x2), 16 EEG channels at 256 Hz (same montage, g.tec gUSBamp x2),
36 character-flag channels, no eye tracking, 250 ms SOA, 31.25 ms flash, 3.5 s median pause between selections.
ALSFRS-R is constant across a participant's three sessions.

| Participant | ALSFRS-R | Selections (CBCol / Static / Dyn / DynBigram) | Truncated fixed-length selections | Dyn flashes min / median / max | Online Static accuracy (all sessions) |
|---|---:|---|---:|---|---:|
| F_03 | 21 | 54 / 36 / 36 / 36 | 0 | 14 / 25 / 59 | 0.92 |
| F_05 | 3 | 54 / 36 / 36 / 36 | 0 | 14 / 20 / 60 | 1.00 |
| F_06 | 3 | 54 / 36 / **35** / 36 | 0 | 36 / 127 / 127 | 0.33 |
| F_07 | 21 | 54 / 36 / 36 / 36 | 0 | 37 / 127 / 127 | 0.19 |
| F_08 | 42 | 54 / 36 / 36 / 36 | 0 | 14 / 19 / 28 | 1.00 |
| F_20 | 5 | 54 / 36 / 36 / 36 | 1 | 14 / 30 / 60 | 0.97 |
| F_21 | 1 | 54 / 36 / 36 / 36 | **3** | 16 / 30 / 61 | 1.00 |
| F_23 | 30 | 54 / 36 / 36 / 36 | 0 | 40 / 110 / 127 | 0.67 |
| F_24 | 33 | 54 / 36 / 36 / 36 | 0 | 15 / 35 / 120 | 1.00 |
| F_25 | 33 | 54 / 36 / 36 / 36 | 0 | 14 / 47 / 127 | 0.86 |

Where the ten differ from F_03 (all reported as WARNINGs, none hidden):
- **Four recorded truncations in fixed-length runs**, always the first selection of a run: F_20 SE002 CBCol01 (108 of 126 flashes), F_21 SE001 Static01 (108), F_21 SE002 CBCol01 (108), **F_21 SE003 Static01 (36 flashes, 4 repetitions; this is in the primary evaluation set)**. All end on a repetition boundary. They are kept and flagged (`evidence_truncated`); policies treat them as exhausted evidence.
- **F_06 SE001 Dyn09 has 5 selections** instead of 6.
- **Dynamic runs are not always short.** F_06, F_07, F_23 and F_25 run many Dyn/DynBigram selections to the cap (125 selections have 127 flashes = 14 repetitions plus the first flash of a 15th). Performance, not the protocol, explains this: the same participants have the weakest decoding (below).
- **Decodability varies enormously** (matches the online results): flash-level AUC of the xDAWN/Riemann decoder on sessions 3 is 0.99 (F_08) down to 0.51 (F_07, essentially no usable P300; online Static accuracy 19%). F_06 (0.62) and F_25 (0.66) are weak. Any aggregate over participants must say so.
- Targets are only ever characters with index 1-25 (letters A-Y); all 36 characters are candidates for the decoder.

## 4. Canonical tables (`data/manifests/`, schema in `services/eeg/datasets/schema.py`)

Layers are physically separate files so hidden labels cannot reach an agent-facing observation:

| Layer | Files | Contents |
|---|---|---|
| Raw (never modified) | `data/raw/**`, `data/bnci2014_008/*.mat` | original recordings |
| Agent-safe metadata | `recordings participants sessions runs selections stimulus_events` `.csv` | ids, timing, `chars_shown` per flash, evidence counts. **No target, no target-flash flag** (asserted by a test) |
| Verifier-only | `verifier_targets.csv`, `verifier_flash_labels.csv` | target index/char, online selection, per-flash target flag |
| Derived evidence | `data/evidence/` | empty by design (no probabilities computed) |

Row counts for F_03: recordings 27, participants 1, sessions 3, runs 27, selections 162, stimulus events 13,351. Selection ids are stable,
for example `bigp3bci/F_03/SE003/Test/Static01#t04`. EEG arrays are not copied; an event is `(recording_id, onset_sample)`.
Small tables are committed in `evaluations/dataset_manifests/`; `stimulus_events.csv` and `verifier_flash_labels.csv` (13k rows each) are regenerated.

## 5. Splits (`splits_session_3way.csv`; counts below are per participant, identical for all ten)

| Role | Runs | Selections | Rule |
|---|---:|---:|---|
| calibration | 10 | 60 | sessions 1-2, complete fixed-length runs (CBCol + Static) |
| evaluation (primary) | 2 | 12 | session 3 Static Test |
| evaluation_secondary | 3 | 18 | session 3 CBCol (calibration-style, never trained on) |
| dynamic_replay_only | 12 | 72 | all Dyn / DynBigram runs, all sessions; not for training, not for stopping policies beyond the recorded stop |

Checks pass: no session or run appears in both calibration and evaluation, and no selection has two roles.
All sessions and comparable run types exist for every participant (90 fixed-length + 72 dynamic selections each; the F_06 Dyn09 and truncation exceptions are listed in 3b).

## 6. Quality checks (fail loudly; `audit_eeg_datasets.py` exits 1 on any ERROR)

Implemented in `services/eeg/datasets/qa.py` and `bigp3bci.py`: unreadable/corrupt EDF, non-uniform sampling, non-finite EEG, duplicate ids,
unknown event values, stimulus outside the recording or outside its selection, non-increasing onsets, non-positive flash duration,
missing or out-of-range target, `StimulusType` vs "target shown" mismatch (character mapping), fixed-length run with varying flashes,
session leakage in splits, and `assert_evidence_within()` which raises if a policy asks for a flash that was never recorded.
Result on F_03: **0 errors**. 14 dataset tests pass, plus the 18 existing safety tests (`pytest`: 32 passed).
The validator tests use hand-built invalid tables (to prove failure); the real-data tests skip when files are absent.

## 7. Is Study F suitable for a stopping benchmark?

- **Yes, for frozen-evidence replay on the fixed-length runs.** Static Test and CBCol selections contain all 126 flashes (14 repetitions), so a policy may stop anywhere from 1 to 14 repetitions and never needs unrecorded evidence. Use `assert_evidence_within`.
- **No, for the Dyn / DynBigram runs.** They contain only the evidence the online system collected (2-6 repetitions). They can only be compared against the recorded stop point, never extended.
- **Caveat on size.** One participant gives 12 primary evaluation selections (30 with the secondary set). Ten participants would give about 120 (300). Report per participant; do not pool selections as independent.
- **Time accounting.** One repetition costs 2.0-2.5 s of stimulation (8 or 10 flashes at 250 ms). A full 14-repetition selection costs 31.5 s.

## 8. Is sessions 1-2 / session 3 valid?

Structurally yes for F_03 (section 5). It is a within-participant cross-session test, which is the harder and more honest setting.
ALSFRS-R was identical (21) at all three sessions for F_03, so no severity drift is visible in this participant. Whether other participants have all
three sessions with the same run types is unverified.

## 9. Protocol questions: resolved, partly resolved, open

Sources: the dataset PDF (`bigP3BCI.pdf`, read in full) and the Study F methods paper, Mainsah et al. 2015, *Increasing BCI
communication rates with dynamic stopping towards more practical use: an ALS study*, J. Neural Eng. 12:016013
(<https://oupub.etsu.edu/cas/psychology/bcilab/documents/mainsah_2015.pdf>).

| # | Question | Status | Answer and evidence |
|---|---|---|---|
| 1 | 6x6 vs the "9 x 8" in the overview table | **Resolved (data and paper agree; the overview table is wrong or refers to something else)** | The paper uses "a 6 x 6 grid only" and never mentions 8 x 9. All 270 files have 36 character channels. Row/column indices in the channel labels are not used by the stimulus (see 2). |
| 2 | How flash groups are generated | **Partly resolved** | The paper: groups of 3, 4 or 5 characters flash together (checkerboard paradigm); one *sequence* flashes every defined subset once = **18 flashes, each character flashed twice**; static = 7 sequences = 126 flashes. Measured here: every repetition (10 flashes, then 8) is a random partition of all 36 characters; group sets are re-drawn (15,270 distinct groups over 360 Static selections), so groups carry no row/column identity and the mapping must come from the per-flash character flags. **Open:** the paper does not say how the random partitions are generated; we use the recorded flags and do not model the generator. |
| 3 | Why 10 and 8 flashes alternate | **Resolved** | An 18-flash sequence = two checkerboard halves; 36 chars / 10 groups = 3-4 per flash, 36 / 8 groups = 4-5 per flash, which gives the "3, 4 or 5" of the paper. Our 14 "repetitions" are 7 paper-sequences split in two; every character appears exactly once per repetition (all 12,584 complete fixed-length repetitions verified, `test_protocol_flash_groups_...`). |
| 4 | Derived repetition boundaries vs the protocol | **Resolved for fixed-length runs; open for dynamic runs** | The disjoint-cover rule gives 14 repetitions in 126 flashes in every untruncated fixed-length selection, equal to the paper's 7 x 18. Dynamic runs stop at flash granularity (often mid-repetition); the paper does not state whether its Bayesian update ran per flash or per sequence. |
| 5 | CurrentTarget / SelectedTarget / flags consistency across conditions | **Resolved** | `StimulusType==1` coincides exactly with "target character lit" in all 153,202 flashes of all conditions (test). `CurrentTarget` is constant within every selection. `SelectedTarget` is **not** a decoder output in CBCol training (no feedback: it matches the target in 11% of selections); in Static / Dyn it is the online decision (Static 79%, Dyn 75%, DynBigram 76%), which our offline per-participant accuracy tracks (section 3b). |
| 6 | Meaning of `CBCol` | **Open (inference only)** | Probably checkerboard-paradigm calibration run ("CB" = checkerboard; the ETSU group also studied a *color* paradigm, Ryan et al. 2017). Not confirmed by the files; it is treated as "Train, fixed length, no feedback". |
| 7 | `SelectedTarget` in DynBigram uses a language prior online | **Open** | Online DynBigram accuracy must not be used as a neural-only baseline. Dyn / DynBigram are excluded from stopping benchmarks. |
| 8 | Display-to-EEG latency | **Open** | `StimulusBegin` is the only timing reference; no hardware-latency statement found. Latencies reported are relative to recorded stimulus timestamps. |
| 9 | Other nine participants: same structure? | **Resolved** | Yes, see 3b. |
| 10 | BNCI2014-008 on real files | **Open (blocked)** | `bnci-horizon-2020.eu` is unreachable from this container. Run `python scripts/data/audit_eeg_datasets.py` on a machine that has `data/bnci2014_008/`. License note: CC BY-NC-ND 4.0, do not redistribute derived files. |

## 10. Reproduction

```bash
# Python env (the repo's .venv works; pyedflib was added to requirements.txt)
uv pip install --python .venv -r requirements.txt pytest

# 1. fetch participants (resumable, checksum-verified; writes data/raw + provenance)
for p in F_03 F_05 F_06 F_07 F_08 F_20 F_21 F_23 F_24 F_25; do .venv/bin/python scripts/data/fetch_bigp3bci.py --study F --participant $p; done   # --dry-run lists files
# manual import: copy EDFs into data/raw/bigp3bci/StudyF/F_03/..., then --verify-only

# 2. audit + tables + splits (reads only; exit code 1 on scientific errors)
.venv/bin/python scripts/data/audit_eeg_datasets.py --out data/processed/audit            # ~40 s, all ten participants

# 3. tests
.venv/bin/python -m pytest evaluations/dataset_tests evaluations/safety_tests -q
```

`data/` is gitignored (`data/raw`, `data/manifests`, `data/processed`, `data/evidence`). The existing BNCI folder `data/bnci2014_008/`
is left where `services/eeg/p300.py` expects it; the audit reads it in place.

## 11. Parameter ledger (organizer clarification)

Nemotron Lightning counts as its marketed 30B for eligibility. `evaluations/results/weights.json` and `scripts/verify_weights.py` now carry two labelled figures:
actual checkpoints **32.0394B** (unchanged) and organizer-counted **30.4615B**, headroom **2.5385B** (other models 0.4615B).
`weights.json` was updated arithmetically from the existing counts; `verify_weights.py` was not re-run (needs Hugging Face access).

## 12. Next step

Study F decoding and stopping baselines are in `docs/EEG_STOPPING_BENCHMARK.md`. Still open: the BNCI audit (needs a machine with the files) and questions 2 (generator), 4 (dynamic granularity), 6-8 above.
