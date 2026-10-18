"""Run the Bayesian policy against a *running* neuro_comm Gym server over HTTP and compare to the in-process replay.

    python scripts/benchmark/run_neuro_comm_live_server.py --url http://127.0.0.1:14545 [--limit N]

Every episode: seed_session -> tool calls -> /verify, with its own session cookie. Writes
evaluations/results/neuro_comm/live_server_bayes.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import policies as pol  # noqa: E402
from services.eeg.study_f import replay  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--evidence", default=str(ROOT / "data/evidence/study_f"))
    a = ap.parse_args()
    eps = replay.load_episodes(Path(a.evidence), roles=("evaluation",))
    truth = replay.load_ground_truth(Path(a.evidence))
    ids = sorted(eps)[: a.limit] if a.limit else sorted(eps)
    rows = []
    with httpx.Client(base_url=a.url, timeout=30) as c:
        c.cookies.jar.clear()
        for sid in ids:
            c2 = httpx.Client(base_url=a.url, timeout=30)           # fresh cookie jar per episode
            env = pol.HttpEnv(c2, sid)
            pol.BayesPolicy(a.tau).run(env)
            body = {"responses_create_params": {"input": "x"},
                    "response": {"id": "r", "created_at": 0, "model": "m", "object": "response", "output": [], "parallel_tool_calls": False,
                                 "tool_choice": "auto", "tools": []}, "episode_id": sid}
            v = c2.post("/verify", json=body, cookies=env.cookies)
            v.raise_for_status()
            v = v.json()
            local = pol.LocalEnv(eps[sid], replay.RewardConfig())
            pol.BayesPolicy(a.tau).run(local)
            o = local.session.outcome(truth[sid])
            rows.append({"episode_id": sid, "server_reward": v["reward"], "local_reward": o["reward"], "server_correct": v["correct"],
                         "local_correct": float(o["correct"]), "server_reps": v["reps"], "local_reps": o["reps"]})
            c2.close()
    df = pd.DataFrame(rows)
    out = ROOT / "evaluations/results/neuro_comm"
    df.to_csv(out / "live_server_bayes.csv", index=False)
    ok = ((df.server_reward - df.local_reward).abs() < 1e-9).all() and (df.server_correct == df.local_correct).all() and (df.server_reps == df.local_reps).all()
    print(f"{len(df)} episodes over HTTP; server mean reward {df.server_reward.mean():.4f}, accuracy {df.server_correct.mean():.3f}; identical to in-process: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
