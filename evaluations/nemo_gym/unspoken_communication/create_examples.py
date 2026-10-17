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
    for name, n, seed in (("example", 1, 1), ("eval", 6, 7)):
        rows = [to_gym(t) for t in generate(n, seed)]
        (HERE / "data" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(name, len(rows))
