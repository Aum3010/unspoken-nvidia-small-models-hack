"""Run the `unspoken_communication` tasks against local policies without a Gym install.

Policies:
  raw:<ollama-model>   the model's own tool call, no validation (what the LLM alone does)
  guarded              the deployed coordinator: model + outside validation + deterministic fallback
  deterministic        rule-based policy only (baseline)

Usage: python evaluations/nemo_gym/run_local.py [--policies raw:nemotron-3.5-lightning:30b-a3b guarded deterministic]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent / "unspoken_communication"))

import services.coordinator.app as coord  # noqa: E402
from scoring import generate, score  # noqa: E402

OUT = ROOT / "evaluations" / "results"


async def run_policy(policy: str, tasks: list[dict]) -> dict:
    rows = []
    for t in tasks:
        req = coord.PlanRequest(**t["plan_request"])
        t0 = time.time()
        call = None
        try:
            if policy.startswith("raw:"):
                coord.MODEL = policy[4:]
                name, args, _ = await coord._call_llm(req)
                args.pop("rationale", None)
                call = {"name": name, "arguments": args}
            elif policy == "guarded":
                tc = await coord.plan(req)
                call = {"name": tc.name, "arguments": tc.arguments, "by": tc.by}
            else:
                tc = coord.validate(coord.deterministic_plan(req), req)
                call = {"name": tc.name, "arguments": tc.arguments}
        except Exception as exc:  # no/invalid tool call counts as a failed step
            call = None
            err = f"{type(exc).__name__}: {exc}"[:160]
        else:
            err = None
        s = score(t, call)
        rows.append({"category": t["category"], "call": call, "error": err, "latency_ms": int((time.time() - t0) * 1000), **s})
        print(f"  {policy:<44} {t['category']:<22} reward={s['reward']:.0f} {('; '.join(s['reasons']) or 'ok')[:70]}", flush=True)

    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["reward"] / 3)
    return {
        "reward_mean": round(statistics.mean(r["reward"] / 3 for r in rows), 4),
        "valid_rate": round(statistics.mean(r["valid"] for r in rows), 4),
        "action_rate": round(statistics.mean(r["action"] for r in rows), 4),
        "unsafe_rate": round(1 - statistics.mean(r["safety"] for r in rows), 4),
        "latency_ms_median": int(statistics.median(r["latency_ms"] for r in rows)),
        "llm_share": round(statistics.mean((r["call"] or {}).get("by", "llm") == "llm" for r in rows), 3) if policy == "guarded" else None,
        "by_category": {k: round(statistics.mean(v), 3) for k, v in by_cat.items()},
        "failures": [r for r in rows if r["reward"] < 3][:12],
    }


LABELS = {
    "guarded": ("Deployed: Nemotron + outside validation + fallback", "Guarded"),
    "deterministic": ("Deterministic rules (reference — written to the same spec as the verifier)", "Rules"),
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=[f"raw:{coord.MODEL}", "guarded", "deterministic"])
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--merge", action="store_true", help="update these policies in an existing coordinator.json")
    args = ap.parse_args()
    tasks = generate(args.per_category, seed=7)
    out = {"env": "unspoken_communication", "n_tasks": len(tasks), "policies": {},
           "note": "Reward = mean of valid/action/safety (each 0-1). 'Raw' = the model's own first tool call, unguarded. The deployed coordinator validates every call outside the model (ids, methods, calibration, history rules) and falls back to rules when it rejects one. Caveat: the deterministic policy was written to the same specification as this verifier, so its 1.0 is a reference ceiling, not an independent baseline. Unsafe = requested speech, invented a candidate, or used uncalibrated gaze."}
    prev = OUT / "coordinator.json"
    if args.merge and prev.exists():
        out["policies"] = json.loads(prev.read_text())["policies"]
    for p in args.policies:
        print(f"\n== {p}")
        res = await run_policy(p, tasks)
        label, short = LABELS.get(p, (f"Raw {p[4:]} (no guard)", p[4:].split(":")[0][:14]))
        out["policies"][p] = {"label": label, "short": short, **res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coordinator.json").write_text(json.dumps(out, indent=2))
    for p, r in out["policies"].items():
        print(f"{p:<44} reward={r['reward_mean']:.3f} valid={r['valid_rate']:.2f} action={r['action_rate']:.2f} unsafe={r['unsafe_rate']:.2f} p50={r['latency_ms_median']}ms")


if __name__ == "__main__":
    asyncio.run(main())
