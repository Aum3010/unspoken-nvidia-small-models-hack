"""Write the neuro_comm task files from the frozen evidence (no EEG is modified; targets are NOT written into any task).

    python evaluations/nemo_gym/neuro_comm/create_examples.py [--evidence data/evidence/study_f]

train.jsonl : calibration-role episodes (sessions 1-2; labels legitimately usable for training; evidence is out-of-fold)
eval.jsonl  : held-out session-3 Static Test episodes (never train on these)
example.jsonl: 5 train-style episodes
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import policies, replay  # noqa: E402

TOOLS = [{"type": "function", "name": f["function"]["name"], "description": f["function"]["description"],
          "parameters": {**f["function"]["parameters"], "additionalProperties": False}, "strict": False} for f in policies.TOOL_SCHEMAS]


def row(ev, cfg):
    return {"responses_create_params": {"input": [{"role": "system", "content": policies.system_prompt(cfg)},
                                                  {"role": "user", "content": "Decide for this selection. Start by calling get_state."}],
                                        "tools": TOOLS},
            "episode_id": ev.episode_id, "participant_id": ev.participant_id}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=str(ROOT / "data/evidence/study_f"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    cfg = replay.RewardConfig()
    eps = replay.load_episodes(Path(a.evidence), roles=("calibration", "evaluation"))
    train = sorted(i for i, e in eps.items() if e.role == "calibration")
    ev = sorted(i for i, e in eps.items() if e.role == "evaluation")
    random.Random(a.seed).shuffle(train)
    for name, ids in (("train", train), ("eval", ev), ("example", train[:5])):
        with open(out / f"{name}.jsonl", "w") as f:
            for i in ids:
                f.write(json.dumps(row(eps[i], cfg)) + "\n")
        print(name, len(ids))


if __name__ == "__main__":
    main()
