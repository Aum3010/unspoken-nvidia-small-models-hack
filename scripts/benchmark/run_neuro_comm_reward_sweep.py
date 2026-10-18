"""Effect of the reward coefficients on the Bayesian policy: which threshold tau is reward-optimal, chosen on calibration
episodes (sessions 1-2, out-of-fold evidence) and then scored once on the held-out session-3 episodes.

    python scripts/benchmark/run_neuro_comm_reward_sweep.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import policies as pol  # noqa: E402
from services.eeg.study_f import replay  # noqa: E402

EVD = ROOT / "data/evidence/study_f"
TAUS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999]
BASE = replay.RewardConfig()
CONFIGS = {
    "default": BASE,
    "wrong x0.25 (errors cheap)": replace(BASE, p_wrong=0.25),
    "wrong x3 (errors costly)": replace(BASE, p_wrong=3.0),
    "time x0.1 (patience cheap)": replace(BASE, c_sec=0.001),
    "time x5 (patience costly)": replace(BASE, c_sec=0.05),
    "abstain x0 (abstention free)": replace(BASE, p_abstain=0.0),
}


def mean_reward(eps, truth, cfg, tau, abstain_below=None):
    rs = []
    for sid, ev in eps.items():
        env = pol.LocalEnv(ev, cfg)
        pol.BayesPolicy(tau).run(env)
        rs.append(env.session.outcome(truth[sid])["reward"])
    return sum(rs) / len(rs)


def main() -> int:
    truth = replay.load_ground_truth(EVD)
    cal = replay.load_episodes(EVD, roles=("calibration",))
    ev = replay.load_episodes(EVD, roles=("evaluation",))
    rows = []
    for name, cfg in CONFIGS.items():
        c = {t: mean_reward(cal, truth, cfg, t) for t in TAUS}
        best = max(c, key=c.get)
        e = {t: mean_reward(ev, truth, cfg, t) for t in TAUS}
        rows.append({"config": name, "tau_best_on_calibration": best, "calibration_reward_at_best": round(c[best], 4),
                     "heldout_reward_at_that_tau": round(e[best], 4), "heldout_best_tau_oracle": max(e, key=e.get),
                     "heldout_reward_oracle": round(max(e.values()), 4), **{f"cal_tau{t}": round(v, 4) for t, v in c.items()}})
        print(rows[-1]["config"], rows[-1]["tau_best_on_calibration"], rows[-1]["heldout_reward_at_that_tau"])
    out = ROOT / "evaluations/results/neuro_comm"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "reward_sweep_bayes.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
