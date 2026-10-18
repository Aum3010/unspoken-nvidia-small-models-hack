# Data: what we used, what we made, and consent

The hackathon rules let teams make synthetic data if they publish the data and the method openly. This page lists
what we made, how to rebuild it, and where it is published. It also credits every open dataset we used and sets
out the consent and media rules.

**Published at:** Hugging Face dataset [AumPandya01/unspoken-data](https://huggingface.co/datasets/AumPandya01/unspoken-data). Licence CC BY 4.0.
The dataset card is [`scripts/release/DATASET_CARD.md`](../scripts/release/DATASET_CARD.md).

## What we publish and how it is made

| Set | Rows | Kind | Rebuild |
|---|---|---|---|
| Coordinator Gym tasks (`unspoken_communication`) | 12 / 72 / 44 | Synthetic scenarios | `python evaluations/nemo_gym/unspoken_communication/create_examples.py` (seeds 1, 7, 2026) |
| EEG stopping Gym tasks (`neuro_comm`) | 5 / 600 / 120 | Derived from bigP3BCI Study F | `python evaluations/nemo_gym/neuro_comm/create_examples.py --seed 7` |
| Nano 4B training labels (E6) | 1,717 hindsight, 8,394 value | Model-made labels on bigP3BCI | `scripts/benchmark/neuro_comm_student.py data` and `value_data` ([RESULTS §8](RESULTS.md)) |
| Coordinator router sets (E7b/E7c) | 4 × 700 | Derived from our E1 scores on GRID | `scripts/benchmark/coordinator_student.py prep` ([RESULTS §9](RESULTS.md)) |
| Lip enrollment features | 22 phrases, 90 clips | A team member's own recordings (features, no video) | Recorded in the app's enrollment flow (`services/vsr/enroll.py`) |
| Voice recordings | 307 WAV, p01–p03 | Team members' own recordings | `asr_eval/scripts/record.py` and the bench's Record tab (below) |

The Kokoro TTS sanity clips are not uploaded. `asr_eval/scripts/make_tts.py` rebuilds them exactly, so the script is
the release.

**Not published:** model adapters (team decision, separate from this data rule), anything derived from
BNCI2014-008 (CC BY-NC-ND forbids it), and the MIMIC-IV extract (it stays with PhysioNet; the script rebuilds it).

## How to publish

```bash
bash scripts/release/build_data_release.sh .run/unspoken-data marginal-sapphire-whale   # local sets + voice clips from Brev
hf auth login
hf upload AumPandya01/unspoken-data .run/unspoken-data . --repo-type dataset   # creates the repo if missing
```

## Open data we used

None of it is committed or re-hosted: `data/` is gitignored and the fetch scripts download each set from its source.

| Dataset | Licence | Source (fetched by) | Used for |
|---|---|---|---|
| bigP3BCI v1.0.0, PhysioNet, DOI [10.13026/0byy-ry86](https://doi.org/10.13026/0byy-ry86); Study F (10 ALS participants; methods: Mainsah et al. 2015, *J. Neural Eng.* 12:016013) and Study H (16 participants, not ALS; Kalika, Collins, Caves & Throckmorton 2017, *J. Neural Eng.* 14:056010) | CC BY 4.0 | [physionet.org](https://physionet.org/content/bigp3bci/1.0.0/) via its S3 mirror, SHA-256 checked (`scripts/data/fetch_bigp3bci.py`) | Study F: sessions 1–2 train the stopping policies, the `neuro_comm` episodes and the Nano 4B; session 3 is held out. Study H: EEG + gaze, held-out runs |
| BNCI2014-008, Riccio et al. 2013, 8 participants with ALS | CC BY-NC-ND 4.0 | [bnci-horizon-2020.eu](http://bnci-horizon-2020.eu/database/data-sets/008-2014/) (`scripts/fetch_assets.sh`) | the app's EEG replay; decoder trained on runs 1–3, tested on runs 4–7. Nothing derived is published |
| GRID audiovisual corpus, Cooke, Barker, Cunningham & Shao 2006 | CC BY 4.0 | [Zenodo 3625687](https://zenodo.org/records/3625687) (`scripts/fetch_assets.sh` fetches s1, s2, alignments) | lip reading and lips + voice; s1 and s2 never train the lip model. The lip-decoder fine-tune trains on s3–s8 (Mac) or s3–s34 without s21 (Brev) |
| MIRACL-VC1, Rekik, Ben-Hamadou & Mahdi 2016, 15 speakers | research use only | obtained separately; no download URL is recorded (`scripts/eval_enrollment_miracl.py` reads `data/miracl/`) | testing personal-enrollment lip reading only; not in the product |
| TORGO, Rudzicz, Namasivayam & Wolff 2012, archives F, FC, M, MC | academic / non-profit research use | [U. Toronto](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) | voice tests on dysarthric and control speakers (frequent single-word prompts; vocabulary logged to `data/torgo_vocab.json`); its sentences as unknown-speech negatives |
| LibriSpeech test-clean (OpenSLR 12) | CC BY 4.0 | [openslr.org/12](https://www.openslr.org/12/), ELDA mirror first | 200 unknown-speech negatives; a WER sanity check |
| MUSAN noise (OpenSLR 17) | CC BY 4.0 (files carry their own attributions) | [openslr.org/17](https://www.openslr.org/17/) | half the files as non-speech negatives, half as additive noise in the degradation grid (split by file-name hash) |
| Speech Commands v0.02, `_background_noise_` only | CC BY 4.0 | [tensorflow.org](https://www.tensorflow.org/datasets/catalog/speech_commands) | non-speech negatives |
| ESC-50, 12 categories (breathing, coughing, clock tick, …) | CC BY-NC 3.0 | [GitHub](https://github.com/karolpiczak/ESC-50) | non-speech negatives, evaluation only |
| MIMIC-IV Clinical Database Demo v2.2, Johnson A, Bulgarelli L, Pollard T, Horng S, Celi LA, Mark R; PhysioNet, DOI [10.13026/dp1f-ex47](https://doi.org/10.13026/dp1f-ex47) | ODbL 1.0 | [physionet.org](https://physionet.org/content/mimic-iv-demo/2.2/) (`scripts/vision/fetch_mimic_event.py`) | one ICU event in the vision preview, display only (below) |

`asr_eval/scripts/fetch_data.py` saves each audio dataset's landing page and licence lines into
`data/provenance.json`, which `asr_eval/results/env.md` reproduces.

**Generated, not downloaded:** Kokoro-82M speech (Apache-2.0 output; 20 phrases and 7 hard negatives in every
English voice, sanity checks only, never in headline tables or tuning); silence and mic-like noise (digital zeros,
pink noise and mains hum at −75 to −45 dBFS); the coordinator Gym tasks; the vision preview's synthetic profile.

**Gated, referenced but not used:** UASpeech, the Speech Accessibility Project corpus and CHAINS (LDC2008S09), and
the dysarthric fine-tunes trained on SAP data (`dys-asr/*`), which need the SAP data use agreement.

## Consent and our own recordings

Everyone recorded (voice p01–p03 and the lip enrollment) is on the team. On 18 Oct 2026 all of them agreed to
publish the recordings openly under CC BY 4.0. This replaces the earlier "evaluation only, delete after the
hackathon" terms. The old `consent.csv` text is not uploaded.

- **Voice recordings (2026-10-17):** 3 pseudonymous speakers, 307 clips, covering all 20 phrases plus near-miss
  sentences and other negatives. p01 recorded normal, noisy, whisper and muffled speech (154 clips); p02 normal and
  whisper (102); p03 normal (51, most clipped from recording too loud).
  - Stored on the instance in `/workspace/asr_eval_data/own/` and backed up to `asr_eval/data/own/` (gitignored).
  - Recorded with `asr_eval/scripts/record.py` (shows the consent text, stores the agreement in
    `data/own/consent.csv`, writes 16 kHz mono WAVs named by pseudonym) or the live bench's Record tab
    (`asr_eval/serve.py`; no checkbox, a notice that clips are stored on the GPU machine, and a verbal OK).
- **Lip enrollment (`data/enroll/`):** the person's own phrases, stored as Auto-AVSR encoder features (`NNN.npy`,
  frames × 768, float16) with `phrase.txt`; no video. Published as `lip_enrollment/`. Not tracked in git; ship a
  local copy to the instance with `tar` ([BREV](BREV.md) §4b).
- **Anyone else:** recordings need explicit opt-in before capture, the person can opt out at any time, and nothing
  is kept by default.

## Media rules in the app

- **Live camera and microphone:** clips go only to the gateway and are discarded after inference. Only text
  hypotheses and quality metadata stay in the in-memory audit log.
- **On a Mac** the app runs locally. **With the backend on Brev**, the browser still handles face, gaze, blink and
  head itself, but the mouth and voice clips it cuts go through the SSH tunnel to the instance, where they are
  decoded in memory and not stored. The page says so only on the calibration screen, which is skipped when a
  saved calibration is reused (a final design decision), so tell the person and get their opt-in first.
- **No public patient video** is downloaded, cropped, transformed or trained on.

## EEG replay: BNCI2014-008 facts and labelling

- 8 channels (Fz, Cz, Pz, Oz, P3, P4, PO7, PO8) at 256 Hz. A 6×6 matrix whose rows and columns flash for 125 ms,
  one flash every 250 ms. Each character has 10 repetitions × 12 flashes.
- 35 characters per participant. Runs 1–3 (characters 1–15) are calibration and runs 4–7 (16–35) the online test,
  as in the original protocol. `y_stim` 1–6 are rows, 7–12 columns; `y` = 2 marks a target flash.
- The glyph layout is the BCI2000 default (A–Z, 1–9, _), checked by decoding A01's ground truth into
  `2BACI 5ROSE ZUPPA GATTO MENTE VIOLA REBUS`.
- Ground-truth labels are used only to train on calibration characters and to score test predictions after
  inference. The UI reveals them only after the replay.
- Every EEG view says `RECORDED EEG` / "EEG RESEARCH REPLAY". An EEG-derived candidate begins with "Recorded EEG
  replay result:", the policy gate refuses to speak it without that label, and an operator, not the patient,
  confirms it.

## Vision preview: MIMIC-IV demo (`data/vision/`, flag `UNSPOKEN_CHECKIN`)

- One charted low-glucose event and the 6 h of heart rate and SpO2 before it, numbers only.
  `scripts/vision/fetch_mimic_event.py` downloads only `icu/d_items`, `icu/icustays` and `icu/chartevents`, finds
  the items by name and keeps no free text. The attribution is written to `data/vision/ATTRIBUTION.txt` and shown
  in the panel's label.
- The ids and shifted dates are the dataset's own; never attempt re-identification. Not in git: regenerate it with
  the script and copy it to the instance like `data/enroll/`.
- The profile (`data/vision/profile.yaml`, from `scripts/vision/profile.example.yaml`; "Raj", family member
  "Priya") is SYNTHETIC: not a real person and not linked to the MIMIC record.

Models and their licences are in [MODEL_CARDS_AND_LICENSES.md](MODEL_CARDS_AND_LICENSES.md).
