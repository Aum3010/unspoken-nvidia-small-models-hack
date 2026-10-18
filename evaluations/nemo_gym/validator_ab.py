"""Paired A/B of the optional validator rules on the NeMo Gym tasks.

One model call per task; the validator then runs on that same call with each flag setting, and a rejected call
falls back to the deterministic policy (as the deployed coordinator does). Differences between settings are
therefore the validator's alone, not the model's run-to-run variation.

  settings: off | negation (UNSPOKEN_VALIDATOR_NEGATION) | ignore_history (UNSPOKEN_VALIDATOR_IGNORE_HISTORY) | both

Usage: python evaluations/nemo_gym/validator_ab.py [--seed 7] [--distinct-from N] --out coordinator_validator_v2_paired_eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent / "unspoken_communication"))

import services.coordinator.app as coord  # noqa: E402
from scoring import generate, score  # noqa: E402

SETTINGS = {"off": (False, False), "negation": (True, False), "ignore_history": (False, True), "both": (True, True)}


def guarded(call: coord.ToolCall | None, req: coord.PlanRequest) -> dict:
    """What the deployed coordinator would do with this model call under the current flags."""
    if call is not None:
        checked = coord.validate(call.model_copy(deep=True), req)
        if checked.valid:
            return {"name": checked.name, "arguments": checked.arguments, "by": "llm"}
    fb = coord.validate(coord.deterministic_plan(req), req)
    return {"name": fb.name, "arguments": fb.arguments, "by": "deterministic_fallback"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--distinct-from", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = generate(args.per_category, seed=args.seed)
    if args.distinct_from is not None:
        def key(t):
            return json.dumps({k: v for k, v in t["plan_request"].items() if k != "message_id"}, sort_keys=True)
        seen = {key(t) for t in generate(args.per_category, seed=args.distinct_from)}
        tasks = [t for t in tasks if key(t) not in seen]

    rows = {name: [] for name in ["raw", *SETTINGS]}
    for t in tasks:
        req = coord.PlanRequest(**t["plan_request"])
        try:
            name, a, _ = await coord._call_llm(req)
            rationale = str(a.pop("rationale", ""))
            call = coord.ToolCall(name=name, arguments=a, rationale=rationale, by="llm")
        except Exception:
            call = None
        rows["raw"].append((t["category"], score(t, {"name": call.name, "arguments": call.arguments} if call else None), "llm"))
        for setting, (neg, ign) in SETTINGS.items():
            coord.VALIDATOR_NEGATION, coord.VALIDATOR_IGNORE_HISTORY = neg, ign
            g = guarded(call, req)
            rows[setting].append((t["category"], score(t, g), g["by"]))
        print(f"  {t['category']:<22} raw={rows['raw'][-1][1]['reward']:.0f} " +
              " ".join(f"{s}={rows[s][-1][1]['reward']:.0f}" for s in SETTINGS), flush=True)

    out = {"env": "unspoken_communication", "n_tasks": len(tasks), "seed": args.seed, "distinct_from": args.distinct_from,
           "model": coord.MODEL, "note": "one model call per task; each setting re-validates that same call", "settings": {}}
    for setting, rs in rows.items():
        by_cat: dict[str, list[float]] = {}
        for cat, s, _ in rs:
            by_cat.setdefault(cat, []).append(s["reward"] / 3)
        out["settings"][setting] = {
            "reward_mean": round(statistics.mean(s["reward"] / 3 for _, s, _ in rs), 4),
            "valid_rate": round(statistics.mean(s["valid"] for _, s, _ in rs), 4),
            "action_rate": round(statistics.mean(s["action"] for _, s, _ in rs), 4),
            "unsafe_rate": round(1 - statistics.mean(s["safety"] for _, s, _ in rs), 4),
            "llm_share": round(statistics.mean(by == "llm" for _, _, by in rs), 3),
            "by_category": {k: round(statistics.mean(v), 3) for k, v in by_cat.items()},
        }
    (ROOT / "evaluations" / "results" / args.out).write_text(json.dumps(out, indent=2))
    for setting, r in out["settings"].items():
        print(f"{setting:<15} reward={r['reward_mean']:.3f} action={r['action_rate']:.2f} unsafe={r['unsafe_rate']:.2f} llm_share={r['llm_share']}")


if __name__ == "__main__":
    asyncio.run(main())
