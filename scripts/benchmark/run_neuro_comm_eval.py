"""Evaluate policies through the `neuro_comm` replay tools on held-out Study F session-3 Static episodes.

    # Bayesian policy through the Gym replay interface + parity against the standalone benchmark (no LLM needed)
    python scripts/benchmark/run_neuro_comm_eval.py --parity
    # all three policies; Nemotron behind any OpenAI-compatible endpoint (vLLM on Brev, or Ollama /v1)
    python scripts/benchmark/run_neuro_comm_eval.py --policies bayes nemotron hybrid \
        --base-url http://<host>:8001 --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

Everything uses the frozen evidence in data/evidence/study_f/ (same decoder for all policies). Ground truth is read only
by the verifier step after an episode ends. Wall-clock replay speed is never reported as patient communication speed:
end-to-end latency = recorded EEG decision latency + measured model inference time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import policies as pol  # noqa: E402
from services.eeg.study_f import replay  # noqa: E402

BOOT = 2000


def _bench():
    p = ROOT / "scripts" / "benchmark" / "run_study_f_baselines.py"
    s = importlib.util.spec_from_file_location("run_study_f_baselines", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def run_episode(ev, truth: int, policy, cfg: replay.RewardConfig) -> dict:
    env = pol.LocalEnv(ev, cfg)
    info = policy.run(env)
    out = env.session.outcome(truth)             # the only place the target is used
    row = {"episode_id": ev.episode_id, "participant_id": ev.participant_id, **{k: v for k, v in out.items() if k not in ("components", "config")},
           **{f"r_{k}": v for k, v in out["components"].items()}, **{k: v for k, v in info.items() if k != "trace"}}
    row["end_to_end_s"] = out["latency_s"] + info.get("model_latency_s", 0.0)
    row["events"] = env.session.events
    return row


def summarize(df: pd.DataFrame, pause: float) -> pd.DataFrame:
    def f(g):
        t_eeg = g.latency_s.sum()
        t_e2e = g.end_to_end_s.sum()
        return pd.Series({
            "n": len(g), "correct": int(g.correct.sum()), "accuracy": g.correct.mean(), "incorrect_commits": int(g.incorrect_commit.sum()),
            "abstentions": int(g.abstained.sum()), "timeouts": int(g.timed_out.sum()), "mean_reps": g.reps.mean(),
            "mean_acq_s": g.acq_s.mean(), "mean_eeg_latency_s": g.latency_s.mean(), "mean_model_latency_s": g.model_latency_s.mean(),
            "mean_end_to_end_s": g.end_to_end_s.mean(), "ccpm_eeg_only": g.correct.sum() / (t_eeg / 60),
            "ccpm_end_to_end": g.correct.sum() / (t_e2e / 60), "invalid_tool_calls": int(g.n_invalid.sum()),
            "escalation_rate": g.escalated.mean(), "model_calls": int(g.model_calls.sum()), "mean_reward": g.reward.mean(),
            "fallbacks": int(g.get("fallback_used", pd.Series(False, index=g.index)).fillna(False).sum())})
    return df.groupby(["policy", "participant_id"]).apply(f, include_groups=False).reset_index()


def aggregate(per: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = ["accuracy", "mean_reps", "mean_acq_s", "mean_eeg_latency_s", "mean_model_latency_s", "mean_end_to_end_s", "ccpm_eeg_only",
            "ccpm_end_to_end", "escalation_rate", "mean_reward"]
    rows = []
    for p, g in per.groupby("policy"):
        row = {"policy": p, "n_participants": len(g), "n_selections": int(g.n.sum()), "incorrect_commits": int(g.incorrect_commits.sum()),
               "abstentions": int(g.abstentions.sum()), "timeouts": int(g.timeouts.sum()), "invalid_tool_calls": int(g.invalid_tool_calls.sum()),
               "model_calls": int(g.model_calls.sum())}
        for c in cols:
            v = g[c].to_numpy(float)
            bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(BOOT)]
            row[c] = np.nanmean(v); row[c + "_lo"], row[c + "_hi"] = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def parity(evd: Path, eps: dict, truth: dict, taus=(0.9, 0.99, 0.9999), abstain: bool = False) -> pd.DataFrame:
    """Bayes policy through the replay tools vs the standalone benchmark policy on the same frozen evidence."""
    b = _bench()
    items = {i["sid"]: i for i in b.load(evd, "xdawn_riemann_lr") if i["role"] == "evaluation"}
    rows = []
    for tau in taus:
        cfg = replay.RewardConfig()
        for sid, ev in eps.items():
            row = run_episode(ev, truth[sid], pol.BayesAbstainPolicy(tau) if abstain else pol.BayesPolicy(tau), cfg)
            std = b.run_policy(items[sid], {"kind": "thr", "tau": tau, "abstain": abstain}, {})
            rows.append({"tau": tau, "episode_id": sid, "gym_committed": row["committed"], "std_committed": bool(std["committed"]),
                         "gym_correct": row["correct"], "std_correct": bool(std["correct"]), "gym_reps": row["reps"], "std_reps": std["reps"],
                         "gym_flashes": row["flashes"], "std_flashes": std["flashes"], "gym_acq_s": row["acq_s"], "std_acq_s": std["acq_s"],
                         "gym_latency_s": row["latency_s"], "std_latency_s": std["latency_s"]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=str(ROOT / "data/evidence/study_f"))
    ap.add_argument("--out", default=str(ROOT / "evaluations/results/neuro_comm"))
    ap.add_argument("--decoder", default="xdawn_riemann_lr")
    ap.add_argument("--role", default="evaluation")
    ap.add_argument("--policies", nargs="*", default=["bayes"])
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reward-config", help="JSON file or string of RewardConfig overrides")
    ap.add_argument("--tag", default="")
    ap.add_argument("--thinking", action="store_true", help="let Nemotron reason before each tool call (off in the app)")
    a = ap.parse_args()
    evd, out = Path(a.evidence), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = replay.load_episodes(evd, a.decoder, (a.role,))
    truth = replay.load_ground_truth(evd)
    ids = sorted(eps)[: a.limit] if a.limit else sorted(eps)
    eps = {i: eps[i] for i in ids}
    cfg = replay.RewardConfig.from_dict(json.loads(Path(a.reward_config).read_text() if a.reward_config and Path(a.reward_config).exists()
                                                    else a.reward_config or "{}"))
    print(f"{len(eps)} {a.role} episodes, decoder {a.decoder}")
    if a.parity:
        d = pd.concat([parity(evd, eps, truth).assign(policy="bayes"), parity(evd, eps, truth, abstain=True).assign(policy="bayes_abstain")],
                      ignore_index=True)
        d.to_csv(out / "bayes_parity_per_episode.csv", index=False)
        bad = d[(d.gym_committed != d.std_committed) | (d.gym_correct != d.std_correct) | (d.gym_reps != d.std_reps) | (d.gym_flashes != d.std_flashes)
                | ((d.gym_acq_s - d.std_acq_s).abs() > 1e-9) | ((d.gym_latency_s - d.std_latency_s).abs() > 1e-9)]
        summ = d.groupby(["policy", "tau"]).agg(n=("episode_id", "size"), gym_abstentions=("gym_committed", lambda s: int((~s).sum())),
                                                std_abstentions=("std_committed", lambda s: int((~s).sum())), gym_not_correct=("gym_correct", lambda s: int((~s).sum())),
                                    std_not_correct=("std_correct", lambda s: int((~s).sum())), gym_mean_reps=("gym_reps", "mean"),
                                    std_mean_reps=("std_reps", "mean")).reset_index()
        summ["mismatched_episodes"] = [int(((bad.tau == t) & (bad.policy == p)).sum()) for p, t in zip(summ.policy, summ.tau)]
        summ.to_csv(out / "bayes_parity_summary.csv", index=False)
        print(summ.to_string(index=False))
        print("PARITY", "PASS" if bad.empty else f"FAIL ({len(bad)} episodes differ)")
        if not bad.empty:
            return 1
    rows = []
    for name in a.policies:
        if name == "bayes":
            mk = lambda: pol.BayesPolicy(a.tau, name=f"bayes_tau{a.tau}")
        elif name == "bayes_abstain":
            mk = lambda: pol.BayesAbstainPolicy(a.tau, name=f"bayes_abstain_tau{a.tau}")
        else:
            if not a.base_url or not a.model:
                print(f"skipping {name}: needs --base-url and --model (OpenAI-compatible Nemotron endpoint); no results fabricated")
                continue
            # thinking off, as the deployed coordinator serves Nemotron (one short tool call per turn)
            llm = pol.OpenAICompatClient(a.base_url, a.model, a.api_key,
                                         extra_body={} if a.thinking else {"chat_template_kwargs": {"enable_thinking": False}})
            if name == "nemotron":
                mk = lambda: pol.NemotronAgent(llm, cfg, name="nemotron_only")
            elif name == "hybrid":
                mk = lambda: pol.HybridPolicy(pol.NemotronAgent(llm, cfg, name="nemotron_escalation"), tau_commit=a.tau, name=f"bayes{a.tau}+nemotron")
            else:
                raise SystemExit(f"unknown policy {name}")
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda sid: run_episode(eps[sid], truth[sid], mk(), cfg), ids))
        for r in res:
            r["policy"] = mk().name
        rows += res
        print(name, "done")
    if rows:
        df = pd.DataFrame(rows)
        df.drop(columns=["events"]).to_csv(out / f"episodes{a.tag}.csv", index=False)
        with open(out / f"rollouts{a.tag}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({k: (v if not isinstance(v, (np.bool_, np.integer, np.floating)) else v.item()) for k, v in r.items()}, default=str) + "\n")
        per = summarize(df, 0.0)
        per.to_csv(out / f"per_participant{a.tag}.csv", index=False)
        agg = aggregate(per)
        agg.to_csv(out / f"aggregate{a.tag}.csv", index=False)
        print(agg[["policy", "n_selections", "accuracy", "accuracy_lo", "accuracy_hi", "incorrect_commits", "abstentions", "mean_reps",
                   "mean_eeg_latency_s", "mean_model_latency_s", "ccpm_eeg_only", "ccpm_end_to_end", "invalid_tool_calls", "escalation_rate"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
