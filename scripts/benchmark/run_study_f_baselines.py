"""Fixed-repetition / Bayesian / calibrated / abstaining stopping baselines on the frozen Study F evidence.

    python scripts/benchmark/run_study_f_baselines.py [--evidence data/evidence/study_f] [--out evaluations/results/eeg_stopping]

Policies see only rep_evidence_<decoder>.npz (+ selection_roles.csv). ground_truth.csv is read only by the scoring
function, after a policy has produced its decision. alpha (calibration temperature) and the operating thresholds are
fitted on calibration-role (sessions 1-2, out-of-fold) selections; session-3 selections are never used to fit anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import stopping as st  # noqa: E402

SOA_S, EPOCH_END_S = 0.25, 0.7
TAUS = [0.1, 0.2, 0.35, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999]
TARGET_ERR = 0.05            # calibration operating point: pooled out-of-fold commit error <= 5%
FIXED_ABSTAIN_K = (2, 4, 7, 14)
BOOT = 2000


def load(evd: Path, dec: str):
    z = np.load(evd / f"rep_evidence_{dec}.npz", allow_pickle=False)
    roles = pd.read_csv(evd / "selection_roles.csv").set_index("selection_id")
    gt = pd.read_csv(evd / "ground_truth.csv").set_index("selection_id")
    items = []
    for i, sid in enumerate(z["selection_id"]):
        d = z["delta"][i]
        ok = ~np.isnan(d).any(1)
        n_avail = int(ok.argmin()) if not ok.all() else len(ok)
        assert ok[:n_avail].all() and not ok[n_avail:].any(), f"{sid}: non-contiguous repetitions"
        if n_avail == 0:
            continue
        L = np.cumsum(d[:n_avail], 0)
        items.append({"sid": str(sid), "pid": roles.loc[sid, "participant_id"], "role": roles.loc[sid, "role"],
                      "L": L, "n_avail": n_avail, "first": float(z["first_onset_s"][i]), "last": z["last_onset_s"][i],
                      "nfl": z["n_flashes"][i], "target": int(gt.loc[sid, "target_index"]) - 1})
    return items


def pause_s(ad: Path) -> float:
    s = pd.read_csv(ad / "selections.csv").sort_values(["run_id", "trial_index"])
    g = (s.groupby("run_id").start_sample.shift(-1) - s.end_sample).dropna() / 256.0
    return float(g.median())


def cost(it, r: int) -> dict:
    flashes = int(it["nfl"][:r].sum())
    return {"flashes": flashes, "reps": r, "acq_s": float(it["last"][r - 1] - it["first"] + SOA_S),
            "latency_s": float(it["last"][r - 1] - it["first"] + EPOCH_END_S)}


def run_policy(it, spec: dict, alpha: dict) -> dict:
    a = alpha[it["pid"]] if spec.get("calibrated") else 1.0
    k = spec["kind"]
    if k == "fixed":
        o = st.fixed_reps(it["L"], it["n_avail"], spec["k"], a, spec.get("abstain_below"))
    else:
        o = st.threshold_stop(it["L"], it["n_avail"], spec["tau"], a, spec.get("abstain", False))
    c = cost(it, o.n_reps)
    return {**c, "committed": o.committed, "correct": bool(o.committed and o.char == it["target"]),
            "exhausted": o.exhausted, "confidence": o.confidence}


def evaluate(items, specs, alpha):
    rows = []
    for name, spec in specs.items():
        for it in items:
            rows.append({"policy": name, "selection_id": it["sid"], "participant_id": it["pid"], "role": it["role"],
                         **run_policy(it, spec, alpha)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, pause: float) -> pd.DataFrame:
    def f(g):
        n, com = len(g), g.committed.sum()
        t = g.latency_s.sum()
        return pd.Series({
            "n_selections": n, "correct": int(g.correct.sum()), "incorrect_commits": int((g.committed & ~g.correct).sum()),
            "abstentions": int((~g.committed).sum()), "exhausted": int(g.exhausted.sum()),
            "accuracy": g.correct.mean(), "commit_error": ((g.committed & ~g.correct).sum() / com) if com else np.nan,
            "error_rate": (g.committed & ~g.correct).mean(),
            "mean_flashes": g.flashes.mean(), "mean_reps": g.reps.mean(), "mean_acq_s": g.acq_s.mean(),
            "mean_latency_s": g.latency_s.mean(), "ccpm": g.correct.sum() / (t / 60),
            "ccpm_with_pause": g.correct.sum() / ((t + pause * n) / 60)})
    return df.groupby(["decoder", "split", "policy", "participant_id"]).apply(f, include_groups=False).reset_index()


def aggregate(per: pd.DataFrame, rng) -> pd.DataFrame:
    cols = ["accuracy", "error_rate", "commit_error", "mean_flashes", "mean_reps", "mean_acq_s", "mean_latency_s", "ccpm", "ccpm_with_pause"]
    out = []
    for (dec, split, pol), g in per.groupby(["decoder", "split", "policy"]):
        pids = g.participant_id.unique()
        row = {"decoder": dec, "split": split, "policy": pol, "n_participants": len(pids),
               "n_selections": int(g.n_selections.sum()), "incorrect_commits": int(g.incorrect_commits.sum()),
               "abstentions": int(g.abstentions.sum()), "exhausted": int(g.exhausted.sum())}
        gi = g.set_index("participant_id")
        for c in cols:
            v = gi[c].to_numpy(float)
            bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(BOOT)]
            row[c] = np.nanmean(v); row[c + "_lo"], row[c + "_hi"] = np.nanpercentile(bs, [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=str(ROOT / "data/evidence/study_f"))
    ap.add_argument("--audit-dir", default=str(ROOT / "data/processed/audit"))
    ap.add_argument("--out", default=str(ROOT / "evaluations/results/eeg_stopping"))
    ap.add_argument("--decoders", nargs="*", default=["xdawn_riemann_lr", "shrinkage_lda"])
    a = ap.parse_args()
    evd, out = Path(a.evidence), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pause = pause_s(Path(a.audit_dir))
    rng = np.random.default_rng(0)
    per_all, op_rows, cal_rows, sel_rows = [], [], [], []
    for dec in a.decoders:
        items = load(evd, dec)
        cal = [i for i in items if i["role"] == "calibration"]
        alpha = {}
        for pid in sorted({i["pid"] for i in cal}):
            ci = [i for i in cal if i["pid"] == pid]
            alpha[pid] = st.fit_alpha([i["L"] for i in ci], [i["target"] for i in ci])
        # calibration operating thresholds from pooled out-of-fold calibration selections
        taus = {}
        for tag, calibrated in (("bayes", False), ("calibrated", True)):
            best = None
            for tau in TAUS:
                d = evaluate(cal, {"x": {"kind": "thr", "tau": tau, "calibrated": calibrated}}, alpha)
                err = (d.committed & ~d.correct).mean()
                op_rows.append({"decoder": dec, "rule": tag, "tau": tau, "oof_error_rate": err, "oof_accuracy": d.correct.mean(),
                                "oof_mean_reps": d.reps.mean()})
                if err <= TARGET_ERR and best is None:
                    best = tau
            taus[tag] = best if best is not None else TAUS[-1]
        # calibration diagnostics at the final repetition (out-of-fold calibration vs session-3 evaluation)
        for split_name, rr in (("calibration_oof", ("calibration",)), ("evaluation", ("evaluation",)), ("evaluation_secondary", ("evaluation_secondary",))):
            sub = [i for i in items if i["role"] in rr]
            for tag, calibrated in (("bayes_alpha1", False), ("calibrated", True)):
                conf, corr = [], []
                for i in sub:
                    p = st.softmax((alpha[i["pid"]] if calibrated else 1.0) * i["L"][-1])
                    conf.append(p.max()); corr.append(p.argmax() == i["target"])
                conf, corr = np.array(conf), np.array(corr, float)
                cal_rows.append({"decoder": dec, "split": split_name, "posterior": tag, "n": len(sub), "mean_conf": conf.mean(),
                                 "accuracy_final_rep": corr.mean(), "ece": st.ece(conf, corr), "brier": float(((conf - corr) ** 2).mean())})
        specs = {}
        for k in range(1, 15):
            specs[f"fixed_{k:02d}"] = {"kind": "fixed", "k": k}
        for tau in TAUS:
            specs[f"bayes_tau{tau}"] = {"kind": "thr", "tau": tau}
            specs[f"calibrated_tau{tau}"] = {"kind": "thr", "tau": tau, "calibrated": True}
            specs[f"calibrated_abstain_tau{tau}"] = {"kind": "thr", "tau": tau, "calibrated": True, "abstain": True}
            specs[f"bayes_abstain_tau{tau}"] = {"kind": "thr", "tau": tau, "abstain": True}
        for k in FIXED_ABSTAIN_K:
            specs[f"fixed_{k:02d}_abstain_cal"] = {"kind": "fixed", "k": k, "calibrated": True, "abstain_below": taus["calibrated"]}
        specs["SELECTED_bayes"] = {"kind": "thr", "tau": taus["bayes"]}
        specs["SELECTED_calibrated"] = {"kind": "thr", "tau": taus["calibrated"], "calibrated": True}
        specs["SELECTED_calibrated_abstain"] = {"kind": "thr", "tau": taus["calibrated"], "calibrated": True, "abstain": True}
        specs["SELECTED_bayes_abstain"] = {"kind": "thr", "tau": taus["bayes"], "abstain": True}
        for split_name, role in (("evaluation", "evaluation"), ("evaluation_secondary", "evaluation_secondary")):
            sub = [i for i in items if i["role"] == role]
            d = evaluate(sub, specs, alpha)
            d["decoder"], d["split"] = dec, split_name
            sel_rows.append(d[d.policy.str.startswith("SELECTED") | d.policy.isin(["fixed_07", "fixed_14"])])
            per_all.append(summarize(d, pause))
        json.dump({"alpha": alpha, "selected_tau": taus}, open(out / f"calibration_params_{dec}.json", "w"), indent=1)
    per = pd.concat(per_all, ignore_index=True)
    per.to_csv(out / "per_participant.csv", index=False)
    agg = aggregate(per, rng)
    agg.to_csv(out / "aggregate.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out / "calibration_operating_curves.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out / "calibration_diagnostics.csv", index=False)
    pd.concat(sel_rows).to_csv(out / "per_selection_selected_policies.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"recorded_median_inter_selection_pause_s": pause, "taus": TAUS, "target_oof_error": TARGET_ERR,
                                                  "bootstrap": f"{BOOT} resamples of participants, percentile 95% CI"}, indent=1))
    print(agg[agg.policy.str.startswith("SELECTED") | agg.policy.isin(["fixed_07", "fixed_14"])]
          [["decoder", "split", "policy", "accuracy", "accuracy_lo", "accuracy_hi", "mean_reps", "ccpm", "incorrect_commits", "abstentions"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
