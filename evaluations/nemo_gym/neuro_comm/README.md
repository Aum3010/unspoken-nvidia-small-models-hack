# neuro_comm (NeMo Gym resources server)

Sequential communication decisions on **real recorded ALS EEG** (bigP3BCI Study F). One episode = one recorded character
selection replayed from the frozen evidence in `data/evidence/study_f/`. The policy sees a posterior over the 36 grid
characters and decides: `request_more_eeg`, `commit_character`, `abstain` (or ask for another modality, which never exists).
Ground truth is read only by the verifier. Built and checked against NeMo Gym 0.7.0.

| Tool | Effect |
|---|---|
| `get_state` | current evidence (repetition, posterior top-5, entropy, recorded acquisition time, exhausted flag) |
| `request_more_eeg` | reveal the next *recorded* repetition; rejected past the recording |
| `commit_character(character)` | final |
| `abstain` | final |
| `request_other_modality(modality)` | always rejected: no audio/gaze/speech was recorded |

Reward (all coefficients in `configs/neuro_comm.yaml`, defaults in `services/eeg/study_f/replay.py::RewardConfig`):
`+1 correct`, `-1 wrong commit`, `-0.2 abstain`, `-0.5 no decision`, `-0.01 per recorded second of EEG`, `-0.005 per tool call`,
`-0.1 per rejected call`, `-0.05 per unavailable-modality request`. Full explanation in `docs/experiments/NEUROCOMM_GYM_EXPERIMENT.md`.

## Use

```bash
python evaluations/nemo_gym/neuro_comm/create_examples.py        # writes data/{train,eval,example}.jsonl (no targets inside)
# in a Gym workspace (cwd contains resources_servers/):
cp -r <unspoken>/evaluations/nemo_gym/neuro_comm resources_servers/neuro_comm
export UNSPOKEN_REPO=<unspoken> UNSPOKEN_EVIDENCE_DIR=<unspoken>/data/evidence/study_f
gym env start --resources-server neuro_comm --model-type openai_model --model <served-model> --model-url http://<host>:8001/v1 --model-api-key x
gym eval run --agent neuro_comm_simple_agent --input resources_servers/neuro_comm/data/eval.jsonl --output results/neuro_comm.jsonl
```

`train.jsonl` holds calibration-role episodes (sessions 1-2) only; `eval.jsonl` is the held-out session-3 Static Test set. Never train on it.
