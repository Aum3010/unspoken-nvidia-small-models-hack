---
license: cc-by-4.0
language: [en]
pretty_name: Unspoken hackathon data
tags: [assistive-communication, nemo-gym, lip-reading, speech, eeg, synthetic]
configs:
  - config_name: coordinator_gym
    data_files: [{split: eval, path: coordinator_gym/eval.jsonl}, {split: heldout, path: coordinator_gym/heldout.jsonl}]
  - config_name: neuro_comm_gym
    data_files: [{split: train, path: neuro_comm_gym/train.jsonl}, {split: eval, path: neuro_comm_gym/eval.jsonl}]
  - config_name: neuro_comm_student
    data_files: [{split: train, path: neuro_comm_student/train_value.jsonl}]
  - config_name: coordinator_student
    data_files: [{split: train, path: "coordinator_student/s?_train.jsonl"}, {split: test, path: "coordinator_student/s?_test.jsonl"}]
---

# Unspoken hackathon data

Data created by team Unspoken at the NVIDIA + Red Hat Small Models Hack (Raleigh, 17–18 Oct 2026): synthetic and
derived training/evaluation sets, plus recordings of the team members themselves. Code and full methodology:
https://github.com/Aum3010/unspoken-nvidia-small-models-hack (`docs/DATA_RELEASE.md`, results in `docs/RESULTS.md`).

## Contents

| Folder | Rows | What | How it was made |
|---|---|---|---|
| `coordinator_gym/` | 12 example, 72 eval, 44 heldout | NeMo Gym `unspoken_communication` tasks: agent candidates + quality flags → one coordinator tool call | Fully synthetic. `evaluations/nemo_gym/unspoken_communication/create_examples.py`, seeds 1 / 7 / 2026; heldout drops every task also in eval |
| `neuro_comm_gym/` | 5 example, 600 train, 120 eval | NeMo Gym `neuro_comm` tasks: when to stop flashing and commit a letter | `evaluations/nemo_gym/neuro_comm/create_examples.py --seed 7`, from bigP3BCI Study F evidence |
| `neuro_comm_student/` | 1,717 hindsight, 8,394 value | LoRA labels for Nemotron Nano 4B on `neuro_comm` (state text → more / commit / abstain) | `scripts/benchmark/neuro_comm_student.py data` (hindsight, attempt 1) and `value_data` (expected Gym reward, attempt 2); bigP3BCI Study F calibration sessions 1–2 only |
| `coordinator_student/` | 700 per file | E7b/E7c coordinator router: voice + lips phrase scores and EEG option → action | `scripts/benchmark/coordinator_student.py prep`; GRID s1/s2 scores from our E1 run, 2-fold by speaker |
| `lip_enrollment/` | 22 phrases, 90 recordings | Auto-AVSR encoder features (`NNN.npy`, frames × 768, float16) + `phrase.txt`; no video | One team member mouthing the patient phrases, recorded in the app's enrollment flow |
| `voice_recordings/` | 307 WAV, 3 speakers | 16 kHz mono; `{speaker}_{condition}_{phraseid}_{take}.wav`; ids and conditions in `phrases_and_conditions.yaml` | Team members p01 (normal, noisy, whisper, muffled), p02 (normal, whisper), p03 (normal; many clips clipped, recorded too loud). 20 phrases, 7 hard negatives, silence / breathing / cough / unrelated |

## Consent

Every person recorded is a member of the team; all agreed on 18 Oct 2026 to publish these recordings openly under
CC BY 4.0. Speakers are pseudonymous (p01–p03). Do not attempt to identify them.

## Sources and attribution

- **bigP3BCI** v1.0.0, PhysioNet, https://doi.org/10.13026/0byy-ry86, CC BY 4.0 (Study F, people with ALS).
  `neuro_comm_gym` and `neuro_comm_student` are derived from it.
- **GRID audiovisual corpus**, Cooke, Barker, Cunningham & Shao (2006), CC BY 4.0. `coordinator_student` is derived
  from model scores on speakers s1 and s2.
- Models used to produce the data: NVIDIA Nemotron Speech Streaming en 0.6B (voice scores), Auto-AVSR (lip
  features and scores).

## Limits

- Small: three speakers, one lip-enrollment speaker. None of the recorded people has a speech impairment.
- The Gym tasks are synthetic and the student labels are model-made; they are training/evaluation material for this
  prototype, not clinical data.
