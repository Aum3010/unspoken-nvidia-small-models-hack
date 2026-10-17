# Datasets, consent and media rules

| Dataset | Licence | Used for | Stored where | Redistributed? |
|---|---|---|---|---|
| BNCI2014-008 — P300 speller, 8 participants with ALS (Riccio et al. 2013) | CC BY-NC-ND 4.0 | EEG replay demo + held-out decoder evaluation | `data/bnci2014_008/` (downloaded by `setup_models.sh`) | **No.** Raw recordings are never committed or re-hosted; the UI shows derived scores and a downsampled trace locally |
| GRID audiovisual corpus, speakers s1–s2 (Cooke et al.) | CC BY 4.0 | Unseen-speaker lip-reading benchmark | `data/grid/` | No (not needed); attribution in UI and docs |
| Kokoro-generated utterances | Apache-2.0 model output | Controlled weak/noisy speech benchmark with exact references | generated at evaluation time | n/a |
| Live camera / microphone | Person's own, opt-in per capture | Live communication | **Not stored.** Clips go only to the local gateway and are discarded after inference; only text hypotheses and quality metadata are kept in the in-memory audit log | Never |

## BNCI2014-008 facts used by the decoder (checked against the files)

- 8 channels: Fz, Cz, Pz, Oz, P3, P4, PO7, PO8. 256 Hz.
- 6×6 matrix. Rows and columns flash for 125 ms, with one flash every 250 ms (4 Hz). Each character has 10 repetitions × 12 flashes = 120 flashes.
- 35 characters per participant (7 five-letter words). In the original protocol, runs 1–3 (characters 1–15) were calibration and runs 4–7 (characters 16–35) were online test. **We keep that split.**
- `y_stim` 1–6 are rows and 7–12 are columns. `y` = 2 marks a target flash.
- The glyph layout is the BCI2000 default (A–Z, 1–9, _). We checked this by decoding the ground-truth row/column pairs for A01, which gives `2BACI 5ROSE ZUPPA GATTO MENTE VIOLA REBUS` (the Italian copy-spelling words).
- Ground-truth labels are used only to train on calibration characters and to score test predictions **after** inference. The UI reveals them only after the replay.

## Honest labelling rules (enforced in code and UI)

- Every EEG view and message shows `RECORDED EEG` / "EEG RESEARCH REPLAY — previously recorded signal, newly run inference".
- An EEG-derived candidate's text begins with "Recorded EEG replay result:". The policy gate refuses to speak recorded data without that label, and an **operator** (not "the patient") confirms the read-out.
- No public patient video is downloaded, cropped, transformed or trained on. Clinical papers and videos are linked in the Evidence page for motivation only.

## Local recordings of teammates or judges

Recordings need explicit opt-in before capture, and the person can opt out at any time. Nothing is retained by default. If a recording is ever kept for evaluation, store it outside the repo with written consent and a deletion date.

## Cloud processing

Everything runs locally by default. If the coordinator is moved to vLLM on Brev (`scripts/serve_vllm_brev.sh`), only the structured evidence summary leaves the machine: candidate text, scores and quality flags. Camera frames, audio and EEG never do. Disclose this to the person, and require their opt-in, before turning it on.
