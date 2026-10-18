# Data release

The hackathon rules let teams make synthetic data if they publish the data and the method openly. This page lists
what we made, how to rebuild it, and where it is published. It also credits every open dataset we used.

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
| Voice recordings | 307 WAV, p01–p03 | Team members' own recordings | `asr_eval/scripts/record.py` and the bench's Record tab ([DATASETS_AND_CONSENT](DATASETS_AND_CONSENT.md)) |

The Kokoro TTS sanity clips are not uploaded. `asr_eval/scripts/make_tts.py` rebuilds them exactly, so the script is
the release.

**Not published:** model adapters (team decision, separate from this data rule), anything derived from
BNCI2014-008 (CC BY-NC-ND forbids it), and the MIMIC-IV extract (it stays with PhysioNet; the script rebuilds it).

## Consent

Everyone recorded (voice p01–p03 and the lip enrollment) is on the team. On 18 Oct 2026 all of them agreed to
publish the recordings openly under CC BY 4.0. This replaces the earlier "evaluation only, delete after the
hackathon" terms in [DATASETS_AND_CONSENT.md](DATASETS_AND_CONSENT.md). The old `consent.csv` text is not uploaded.

## How to publish

```bash
bash scripts/release/build_data_release.sh .run/unspoken-data marginal-sapphire-whale   # local sets + voice clips from Brev
hf auth login
hf upload AumPandya01/unspoken-data .run/unspoken-data . --repo-type dataset   # creates the repo if missing
```

## Open data we used (attribution)

| Dataset | Licence | Used for |
|---|---|---|
| bigP3BCI v1.0.0, PhysioNet, https://doi.org/10.13026/0byy-ry86 | CC BY 4.0 | EEG results (Study F, ALS; Study H, EEG + gaze), `neuro_comm` |
| BNCI2014-008, Riccio et al. 2013 | CC BY-NC-ND 4.0 | EEG replay demo; nothing derived is published |
| GRID audiovisual corpus, Cooke, Barker, Cunningham & Shao 2006 | CC BY 4.0 | Lip reading and lips + voice benchmarks |
| MIMIC-IV Clinical Database Demo v2.2, Johnson et al., PhysioNet, https://doi.org/10.13026/dp1f-ex47 | ODbL 1.0 | Vision preview (one ICU event) |
| TORGO, Rudzicz, Namasivayam & Wolff 2012 | Academic research use | Dysarthric speech evaluation |
| LibriSpeech test-clean (OpenSLR 12) | CC BY 4.0 | Unknown-speech negatives |
| MUSAN (OpenSLR 17) | CC BY 4.0 | Noise and non-speech negatives |
| Speech Commands v0.02, background noise | CC BY 4.0 | Non-speech negatives |
| ESC-50 | CC BY-NC 3.0 | Non-speech negatives (evaluation only) |

Models and their licences are in [MODEL_CARDS_AND_LICENSES.md](MODEL_CARDS_AND_LICENSES.md).
