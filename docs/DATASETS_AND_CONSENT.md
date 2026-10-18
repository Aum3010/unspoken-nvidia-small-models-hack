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

## ASR evaluation datasets (`asr_eval/`, run on Brev)

`asr_eval/scripts/fetch_data.py` downloads these. For each one it saves the landing page and the lines mentioning the licence into `data/provenance.json`, which `results/env.md` reproduces. None of them are committed or re-hosted.

| Dataset | Licence | Used for | Notes |
|---|---|---|---|
| TORGO (Rudzicz, Namasivayam & Wolff 2012), official U. Toronto archives F, FC, M, MC | Academic / non-profit research use; cite the paper | Surrogate closed-vocabulary task from frequent single-word prompts (dysarthric vs control reported separately); TORGO sentences not in our phrase list as unknown-speech negatives | Array mic preferred, head mic fallback. Vocabulary chosen automatically (most distinct speakers) and logged to `data/torgo_vocab.json` |
| LibriSpeech test-clean (OpenSLR 12) | CC BY 4.0 | 200 utterances as unknown-speech negatives; WER sanity check | ELDA mirror first, main OpenSLR server as fallback |
| MUSAN noise (OpenSLR 17) | CC BY 4.0 per OpenSLR; files carry their own attributions | Half the noise files: non-speech negatives. The other half: additive noise in the degradation grid | Split by file-name hash, so no recording is both a negative and inside a positive |
| Speech Commands v0.02 `_background_noise_` | CC BY 4.0 | Non-speech negatives | Only that folder is extracted |
| ESC-50 | CC BY-NC 3.0 | Non-speech negatives from 12 categories (breathing, coughing, clock tick, …) | Non-commercial: evaluation only |
| Kokoro-82M synthetic speech | Apache-2.0 model output | 20 phrases + 7 hard negatives × every English voice | **Sanity only**; never in headline tables or tuning |
| Generated silence / mic-like noise | n/a (generated) | Silence negatives | Digital zeros and pink noise + mains hum at −75 to −45 dBFS |

**Gated, referenced but not downloaded:** UASpeech, the Speech Accessibility Project corpus, and CHAINS (LDC2008S09). The dysarthric fine-tunes trained on SAP data (`dys-asr/*`) are skipped for the same reason: they need the SAP data use agreement.

## Local recordings of teammates or judges

- **Opt-in:** recordings need explicit opt-in before capture, and the person can opt out at any time.
- **Default:** nothing is retained.
- **If kept for evaluation:** store it outside the repo, with written consent and a deletion date.

For the ASR evaluation there are two recorders:
- **The live bench's Record tab** (`asr_eval/serve.py`). It has no consent checkbox, at the project owner's request: it shows a notice that clips are stored on the GPU machine, and the person running the session must get a verbal OK from anyone they record. Speakers are logged by pseudonym in `data/own/speakers.csv`.
- **The CLI recorder** `asr_eval/scripts/record.py`, described below.

**Recorded so far (2026-10-17):**
- **Speakers:** 3 pseudonymous speakers, 307 clips.
  - p01: normal, noisy, whisper, muffled (154 clips);
  - p02: normal, whisper (102);
  - p03: normal (51; most clips clipped from recording too loud).
- **Content:** each speaker covers all 20 phrases plus the near-miss sentences and the other negatives.
- **Storage:**
  - on the Brev instance in `/workspace/asr_eval_data/own/`;
  - a laptop backup in `asr_eval/data/own/`, gitignored;
  - never in the repository.
- **Deletion:** both copies are to be deleted after the hackathon. Deleting the instance removes its copy.

`record.py` does the following:
- It shows the consent text and stores the agreement in `data/own/consent.csv`.
- It saves raw 16 kHz mono WAVs as `data/own/{pseudonym}_{condition}_{phraseid}_{take}.wav`.
- Use pseudonyms, never real names.
- The consent text says the clips are also stored on the project's evaluation GPU machine (Brev) and deleted after the hackathon.
- Per-clip outputs that contain transcripts (`results/raw/`) are gitignored.

## Lip enrollment recordings (`data/enroll/`, branch `ui`)

- **What they are:** the lip agent's personal phrases. For 22 phrases there are 90 recordings, each stored as
  Auto-AVSR encoder features (`NNN.npy`, frames × 768, float16) with `phrase.txt`. There is no video, but the
  features are derived from one real person's mouth recordings.
- **Status: committed to git.** They were force-added past the `/data/` gitignore rule. `services/vsr/enroll.py`
  says they "are not uploaded", and the rules above keep personal recordings out of git.
- **For the team:** confirm the person's consent covers a shared repository, or remove them from git history and
  ship them to the instance separately, as `docs/BREV.md` §4b already does for running.

## Cloud processing

**On a Mac, the app runs locally.** If only the coordinator is moved to vLLM on Brev (`scripts/serve_vllm_brev.sh`), only the structured evidence summary leaves the machine: candidate text, scores and quality flags. Camera frames, audio and EEG never do.

**With the whole backend on Brev** (`scripts/brev/start_app.sh`), the browser still processes the camera itself (face, gaze, blink and head never leave it), but the **mouth video clips and voice clips** it cuts are sent through the SSH tunnel to the instance. They are decoded in memory and not stored; the decision, transcript and audit trail stay in the gateway's session memory on the instance. The UI's camera chip says "clips to NVIDIA Brev" whenever the gateway reports a remote host.

In both cases, disclose this to the person, and require their opt-in, before turning it on. **On branch `ui` the UI no longer does this**: the camera chip always says "Camera · on device", and the calibration screen says the camera and microphone stay on this device. Tell the person yourself until it is restored.

**The ASR evaluation is different.** It uploads audio clips, including consented volunteer recordings, to a Brev GPU instance under `/workspace`. That is covered by the recording consent above. Deleting the instance deletes the disk. See [BREV.md](BREV.md) §6.
