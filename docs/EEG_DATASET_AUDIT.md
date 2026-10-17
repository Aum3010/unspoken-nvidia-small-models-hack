# EEG dataset audit: BNCI2014-008 and bigP3BCI Study F

Scope: data foundation only. No model, policy, Gym or Nemotron work was done. Everything below was measured from files
on disk unless it is under "Unresolved". Audit run 2026-10-17.

## 1. Summary

| | BNCI2014-008 | bigP3BCI Study F |
|---|---|---|
| Files audited | **none** (not on disk; source host `bnci-horizon-2020.eu` unreachable from the audit container) | **27 EDF+ files, participant F_03, all 3 sessions, 131,252,050 bytes, SHA-256 verified** |
| Audit status | adapter written and unit-tested on a hand-built recording only; **real-file audit not done** | real files opened and inspected |
| Grid | 6x6 row/column (existing code, unverified here) | **36 per-character channels labelled `<char>_<row>_<col>`, rows/cols 1-6**; the PhysioNet overview table says 9x8: **unresolved** |
| Stimulus unit | row or column of 6 chars | **a subset of 3-5 characters per flash** (no row/column decoding is possible) |
| EEG | 8 ch, 256 Hz | **16 ch, 256 Hz** (F3 Fz F4 T7 C3 Cz C4 T8 CP3 CP4 P3 Pz P4 PO7 PO8 Oz) |
| Eye tracking | no | **no** (no `ET*` channels in any Study F file) |

Two corrections to earlier plans:
- **There is no participant `F_01`.** Study F contains `F_03 F_05 F_06 F_07 F_08 F_20 F_21 F_23 F_24 F_25` (from `SHA256SUMS.txt`, 27 files each). F_03 was downloaded.
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

## 3. Observed structure (F_03)

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

Dynamic runs stop after 27.9 flashes on average (about 7.0 s of stimulation) and **none reaches the 126-flash maximum**.
Their stop is often mid-repetition, so evidence is flash-level, not repetition-level.

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

## 5. Splits (`splits_session_3way.csv`, F_03 only)

| Role | Runs | Selections | Rule |
|---|---:|---:|---|
| calibration | 10 | 60 | sessions 1-2, complete fixed-length runs (CBCol + Static) |
| evaluation (primary) | 2 | 12 | session 3 Static Test |
| evaluation_secondary | 3 | 18 | session 3 CBCol (calibration-style, never trained on) |
| dynamic_replay_only | 12 | 72 | all Dyn / DynBigram runs, all sessions; not for training, not for stopping policies beyond the recorded stop |

Checks pass: no session or run appears in both calibration and evaluation, and no selection has two roles.
All sessions and comparable run types exist for F_03. **This is not established for the other nine participants** (not downloaded).

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

## 9. Unresolved questions

| # | Question | Evidence needed |
|---|---|---|
| 1 | Grid: files show 36 character channels labelled with rows/cols 1-6, but the dataset overview table lists 9x8 for Study F | the PDF's Study F protocol section or BCI2000 parameter files; compare with another Study F participant |
| 2 | Meaning of condition `CBCol` and whether Static Test uses the same flash sequence as Train | dataset PDF / paper; compare `chars_shown` sequences across runs |
| 3 | Why repetitions alternate 10 and 8 flashes, and at what granularity the online dynamic-stopping rule was evaluated (stops are mid-repetition) | Mainsah et al. protocol description |
| 4 | Whether `SelectedTarget` in Test runs is neural-only; DynBigram used a language prior online | paper methods; do not use DynBigram online accuracy as a neural-only baseline |
| 5 | Display-to-EEG latency (is `StimulusBegin` aligned to the screen flash?) | dataset PDF timing note |
| 6 | Other nine Study F participants: same structure? ALSFRS-R per session? | download and re-run the audit (`--participant`) |
| 7 | BNCI2014-008: not audited on real files; known format comes from `services/eeg/p300.py` and its docstring | run the audit on the Mac that has `data/bnci2014_008/` |
| 8 | License note: BNCI2014-008 is CC BY-NC-ND 4.0 (per `services/eeg/app.py`); do not redistribute derived BNCI files | check the BNCI terms |

## 10. Reproduction

```bash
# Python env (the repo's .venv works; pyedflib was added to requirements.txt)
uv pip install --python .venv -r requirements.txt pytest

# 1. fetch one participant (resumable, checksum-verified; writes data/raw + provenance)
.venv/bin/python scripts/data/fetch_bigp3bci.py --study F --participant F_03          # add --dry-run to list files
# manual import: copy EDFs into data/raw/bigp3bci/StudyF/F_03/..., then --verify-only

# 2. audit + tables + splits (reads only; exit code 1 on scientific errors)
.venv/bin/python scripts/data/audit_eeg_datasets.py                                     # writes data/manifests/

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

Run the audit on the other nine Study F participants (about 130 MB each via the same fetcher), resolve questions 1-4 from the dataset PDF, and run the BNCI audit on the Mac. Only then define the replay interface.
