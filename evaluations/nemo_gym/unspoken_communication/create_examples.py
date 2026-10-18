"""Regenerate data/*.jsonl in NeMo Gym format (seeded, deterministic)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from scoring import generate  # noqa: E402
from services.coordinator.app import SYSTEM_PROMPT, TOOLS, PlanRequest, _user_payload  # noqa: E402

RESPONSES_TOOLS = [{"type": "function", "strict": False, **t["function"]} for t in TOOLS]


def to_gym(task: dict) -> dict:
    user = _user_payload(PlanRequest(**task["plan_request"]))
    return {
        "responses_create_params": {
            "input": [{"role": "developer", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "tools": RESPONSES_TOOLS,
        },
        "category": task["category"], "plan_request": task["plan_request"], "expected": task["expected"],
    }


if __name__ == "__main__":
    # heldout: same generator, seed 2026, without the tasks that also occur in eval (44 of 72 remain, 10 categories);
    # no prompt or validator change was tuned on it. run_local.py --seed 2026 --distinct-from 7 scores the same set.
    def key(t):
        return json.dumps({k: v for k, v in t["plan_request"].items() if k != "message_id"}, sort_keys=True)

    in_eval = {key(t) for t in generate(6, 7)}
    for name, n, seed in (("example", 1, 1), ("eval", 6, 7), ("heldout", 6, 2026)):
        rows = [to_gym(t) for t in generate(n, seed) if name != "heldout" or key(t) not in in_eval]
        (HERE / "data" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(name, len(rows))
