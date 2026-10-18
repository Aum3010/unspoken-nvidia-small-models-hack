"""Study F minimum-latency EEG selection: flash-level evidence, calibrated stopping, matched-risk comparison.

    /home/claude/venv-eeg/bin/python scripts/benchmark/run_study_f_fast_selection.py [--decoders ...] [--extra-evidence path]

Inputs (cached, no EEG reprocessing): data/evidence/study_f/{flash_evidence.csv.gz, rep_evidence_<decoder>.npz,
ground_truth.csv}; optional extra flash evidence from scripts/benchmark/study_f_early_decoders.py.
Outputs: evaluations/results/eeg_fast_selection/.

Split discipline: every alpha, threshold and deadline is chosen on calibration-role selections (sessions 1-2, out-of-fold
decoder scores). Session 3 (role evaluation = Static, evaluation_secondary = CBCol) is only scored, after all choices are
fixed and written to chosen_params.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import flash_stopping as fs  # noqa: E402
from services.eeg.study_f import stopping as st  # noqa: E402

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
DEADLINES = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, fs.INF]
RISKS = [0.0, 0.02, 0.05, 0.10]
SWEEP_TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999,
              0.99999]


def boot_ci(per: pd.DataFrame, col: str, rng, n=2000):
    v = per[col].to_numpy(float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    b = rng.choice(v, (n, len(v))).mean(1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def baseline_rep_bayes(evd: Path, dec: str, sids: list[str], targets: dict, tau=0.99) -> pd.DataFrame:
    """The original benchmark's `bayes_tau0.99` (repetition boundaries, alpha 1, commit argmax on exhaustion), recomputed
    from the frozen repetition evidence exactly as scripts/benchmark/run_study_f_baselines.py does."""
    path = next(p for p in (evd / f"rep_evidence_{dec}.npz", ROOT / "data/evidence/study_f_early" / f"rep_evidence_{dec}.npz") if p.exists())
    z = np.load(path)
    idx = {str(s): i for i, s in enumerate(z["selection_id"])}
    rows = []
    for sid in sids:
        i = idx[sid]
        d = z["delta"][i]
        ok = ~np.isnan(d).any(1)
        n = int(ok.argmin()) if not ok.all() else len(ok)
        L = np.cumsum(d[:n], 0)
        o = st.threshold_stop(L, n, tau, 1.0)
        t = float(z["last_onset_s"][i][o.n_reps - 1] - z["first_onset_s"][i] + fs.EPOCH_END_S)
        c = o.char == targets[sid]
        rows.append({"selection_id": sid, "committed": True, "correct": bool(c), "wrong": not c, "abstained": False,
                     "reason": "exhausted" if o.exhausted else "threshold", "flashes": int(z["n_flashes"][i][:o.n_reps].sum()),
                     "t_dec": t, "t_acq": t - fs.EPOCH_END_S + fs.SOA_S, "conf": o.confidence})
    return pd.DataFrame(rows)


def eval_policy(trs, sels, targets, pol) -> pd.DataFrame:
    rows = []
    for tr, s in zip(trs, sels):
        r = fs.outcome(tr, pol, targets[s.selection_id])
        r["selection_id"], r["participant_id"] = s.selection_id, s.participant_id
        rows.append(r)
    return pd.DataFrame(rows)


def ece_by_band(sels, trs, targets) -> list[dict]:
    rows = []
    bands = [(0, 2.95, "<=1 rep"), (2.95, 9.95, "1-4 reps"), (9.95, 1e9, ">4 reps")]
    for lo, hi, name in bands:
        conf, cor = [], []
        for s, tr in zip(sels, trs):
            m = (tr.t_dec > lo) & (tr.t_dec <= hi)
            conf.append(tr.pmax[m]); cor.append(tr.arg[m] == targets[s.selection_id])
        conf, cor = np.concatenate(conf), np.concatenate(cor)
        rows.append({"band": name, "n_prefixes": len(conf), "mean_conf": conf.mean(), "accuracy": cor.mean(),
                     "ece": st.ece(conf, cor)})
    return rows


def run_decoder(dec, flash, evd, targets, out, rng, log):
    t0 = time.time()
    cal = fs.load_selections(flash, dec, ["calibration"])
    ev = {"evaluation": fs.load_selections(flash, dec, ["evaluation"]),
          "evaluation_secondary": fs.load_selections(flash, dec, ["evaluation_secondary"])}
    pids = sorted({s.participant_id for s in cal})
    Lcal = {s.selection_id: s.evidence() for s in cal}
    tcal = {s.selection_id: targets[s.selection_id] for s in cal}

    # ---- alpha (calibration only) -----------------------------------------------------------------------------------
    a_glob = fs.fit_alpha([Lcal[s.selection_id] for s in cal], [tcal[s.selection_id] for s in cal])
    a_part = {p: fs.fit_alpha([Lcal[s.selection_id] for s in cal if s.participant_id == p],
                              [tcal[s.selection_id] for s in cal if s.participant_id == p]) for p in pids}
    log(f"[{dec}] alpha global {a_glob:.3f}; per participant " + ", ".join(f"{p}:{a:.2f}" for p, a in a_part.items()))

    def trs(sels, mode):
        return [fs.trajectory(s, 1.0 if mode == "one" else a_glob if mode == "glob" else a_part[s.participant_id]) for s in sels]

    cal_tr = {m: trs(cal, m) for m in ("one", "glob", "part")}
    cal_t = [tcal[s.selection_id] for s in cal]

    # ---- choose operating points at matched calibration risk ---------------------------------------------------------
    chosen = {"decoder": dec, "alpha_global": a_glob, "alpha_participant": a_part, "risks": {}}
    for r in RISKS:
        g = fs.choose_params(cal_tr["glob"], cal_t, r, TAUS, DEADLINES, "flash")
        h = fs.choose_params(cal_tr["part"], cal_t, r, TAUS, DEADLINES, "flash")
        pf, pr = {}, {}
        for p in pids:
            ix = [i for i, s in enumerate(cal) if s.participant_id == p]
            trp, tp = [cal_tr["part"][i] for i in ix], [cal_t[i] for i in ix]
            pf[p] = fs.choose_params(trp, tp, r, TAUS, DEADLINES, "flash")
            pr[p] = fs.choose_params(trp, tp, r, TAUS, DEADLINES, "repetition")
        chosen["risks"][str(r)] = {"flash_global": g, "flash_hybrid": h, "flash_participant": pf, "rep_participant": pr}
    (out / f"chosen_params_{dec}.json").write_text(json.dumps(chosen, indent=1))
    log(f"[{dec}] calibration choices fixed ({time.time() - t0:.0f}s)")

    # ---- session 3 (held out) ---------------------------------------------------------------------------------------
    all_rows, oracle_rows, sweep_rows, prefix_rows, ece_rows = [], [], [], [], []
    for role, sels in ev.items():
        tg = {s.selection_id: targets[s.selection_id] for s in sels}
        tr = {m: trs(sels, m) for m in ("one", "glob", "part")}

        def add(name, df, family, risk=None):
            df = df.copy()
            df["policy"], df["family"], df["risk_target"], df["role"], df["decoder"] = name, family, risk, role, dec
            all_rows.append(df)

        base = baseline_rep_bayes(evd, dec, [s.selection_id for s in sels], tg)
        base["participant_id"] = [s.participant_id for s in sels]
        add("baseline_rep_bayes_tau0.99_commit", base, "baseline")
        add("flash_bayes_tau0.99_commit", eval_policy(tr["one"], sels, tg, fs.Policy(0.99, fs.INF, "commit")), "flash_commit")
        for r in RISKS:
            c = chosen["risks"][str(r)]
            g = c["flash_global"]
            add(f"flash_global_risk{r:g}", eval_policy(tr["glob"], sels, tg, fs.Policy(g["tau"], g["deadline"])), "flash_global", r)
            h = c["flash_hybrid"]
            add(f"flash_hybrid_risk{r:g}", eval_policy(tr["part"], sels, tg, fs.Policy(h["tau"], h["deadline"])), "flash_hybrid", r)
            for fam, key, gran in (("flash_participant", "flash_participant", "flash"), ("rep_participant", "rep_participant", "repetition")):
                rows = []
                for p in pids:
                    ix = [i for i, s in enumerate(sels) if s.participant_id == p]
                    q = c[key][p]
                    rows.append(eval_policy([tr["part"][i] for i in ix], [sels[i] for i in ix], tg,
                                            fs.Policy(q["tau"], q["deadline"], "abstain", gran)))
                add(f"{fam}_risk{r:g}", pd.concat(rows, ignore_index=True), fam, r)

        # descriptive frontier: one global tau swept on session 3 (NOT a deployable choice; for the shape of the frontier)
        for fam, mode, gran, onf in (("sweep_flash_participant_abstain", "part", "flash", "abstain"),
                                     ("sweep_flash_alpha1_commit", "one", "flash", "commit"),
                                     ("sweep_rep_alpha1_commit", "one", "repetition", "commit"),
                                     ("sweep_rep_participant_abstain", "part", "repetition", "abstain")):
            for tau in SWEEP_TAUS:
                d = eval_policy(tr[mode], sels, tg, fs.Policy(tau, fs.INF, onf, gran))
                per = d.groupby("participant_id").apply(fs.summarize, include_groups=False).apply(pd.Series)
                sweep_rows.append({"decoder": dec, "role": role, "family": fam, "tau": tau, **fs.summarize(d),
                                   "mean_participant_accuracy": per.accuracy.mean()})

        # information reference (not a policy) and fixed-prefix argmax accuracy
        for s in sels:
            o = fs.oracle_times(s, a_part[s.participant_id], tg[s.selection_id])
            oracle_rows.append({"decoder": dec, "role": role, "selection_id": s.selection_id, "participant_id": s.participant_id, **o})
        for s, t in zip(sels, tr["part"]):
            hit = t.arg == tg[s.selection_id]
            for k in range(126):
                kk = min(k, s.n - 1)
                prefix_rows.append({"decoder": dec, "role": role, "participant_id": s.participant_id, "k": k + 1,
                                    "t_dec": round(0.25 * k + fs.EPOCH_END_S, 3), "correct": bool(hit[kk]),
                                    "conf": float(t.pmax[kk]), "exhausted": k >= s.n})
        for m in ("one", "part"):
            for row in ece_by_band(sels, tr[m], tg):
                ece_rows.append({"decoder": dec, "split": role, "alpha": "1" if m == "one" else "participant", **row})
    for m in ("one", "part"):
        for row in ece_by_band(cal, cal_tr[m], tcal):
            ece_rows.append({"decoder": dec, "split": "calibration (out-of-fold)", "alpha": "1" if m == "one" else "participant", **row})
    # calibration-side fixed-prefix accuracy too (for the decoder comparison on training sessions)
    for s, t in zip(cal, cal_tr["part"]):
        hit = t.arg == tcal[s.selection_id]
        for k in range(126):
            kk = min(k, s.n - 1)
            prefix_rows.append({"decoder": dec, "role": "calibration", "participant_id": s.participant_id, "k": k + 1,
                                "t_dec": round(0.25 * k + fs.EPOCH_END_S, 3), "correct": bool(hit[kk]),
                                "conf": float(t.pmax[kk]), "exhausted": k >= s.n})

    # ---- CPU inference cost of the stopping update (decoder scoring is timed in study_f_early_decoders.py) ----------
    s = ev["evaluation"][0]
    L = np.zeros(36)
    reps = 20000
    t1 = time.perf_counter()
    for i in range(reps):
        j = i % s.n
        L += s.llr[j] * s.shown[j]
        p = st.softmax(a_glob * L)
        _ = p.max() >= 0.99
    cpu_us = (time.perf_counter() - t1) / reps * 1e6
    return (pd.concat(all_rows, ignore_index=True), pd.DataFrame(oracle_rows), pd.DataFrame(sweep_rows),
            pd.DataFrame(prefix_rows), pd.DataFrame(ece_rows), chosen, cpu_us)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=str(ROOT / "data/evidence/study_f"))
    ap.add_argument("--extra-evidence", nargs="*", default=[], help="extra flash_evidence files (new decoders)")
    ap.add_argument("--decoders", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "evaluations/results/eeg_fast_selection"))
    a = ap.parse_args()
    evd, out = Path(a.evidence), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flash = pd.read_csv(evd / "flash_evidence.csv.gz")
    for x in a.extra_evidence:
        flash = pd.concat([flash, pd.read_csv(x)], ignore_index=True)
    gt = pd.read_csv(evd / "ground_truth.csv")
    targets = {str(k): int(v) - 1 for k, v in zip(gt.selection_id, gt.target_index)}
    decs = a.decoders or sorted(flash.decoder.unique())
    rng = np.random.default_rng(0)
    log = lambda m: print(m, flush=True)  # noqa: E731

    res = [run_decoder(d, flash, evd, targets, out, rng, log) for d in decs]
    sel = pd.concat([r[0] for r in res], ignore_index=True)
    oracle = pd.concat([r[1] for r in res], ignore_index=True)
    sweep = pd.concat([r[2] for r in res], ignore_index=True)
    prefix = pd.concat([r[3] for r in res], ignore_index=True)
    ece = pd.concat([r[4] for r in res], ignore_index=True)
    sel.to_csv(out / "per_selection.csv.gz", index=False)
    oracle.to_csv(out / "oracle_information_times.csv", index=False)
    sweep.to_csv(out / "descriptive_frontier_sweep.csv", index=False)
    ece.to_csv(out / "calibration_ece_by_band.csv", index=False)
    pc = prefix.groupby(["decoder", "role", "participant_id", "k", "t_dec"]).agg(accuracy=("correct", "mean"),
                                                                              mean_conf=("conf", "mean"),
                                                                              frac_exhausted=("exhausted", "mean")).reset_index()
    pc.to_csv(out / "prefix_accuracy_per_participant.csv", index=False)

    per = sel.groupby(["decoder", "role", "policy", "family", "risk_target", "participant_id"], dropna=False).apply(
        fs.summarize, include_groups=False).apply(pd.Series).reset_index()
    per.to_csv(out / "per_participant.csv", index=False)
    agg_rows = []
    for key, g in sel.groupby(["decoder", "role", "policy", "family", "risk_target"], dropna=False):
        pooled = fs.summarize(g)
        pp = per[(per.decoder == key[0]) & (per.role == key[1]) & (per.policy == key[2])]
        lo, hi = boot_ci(pp, "accuracy", rng)
        clo, chi = boot_ci(pp, "ccpm", rng)
        agg_rows.append({"decoder": key[0], "role": key[1], "policy": key[2], "family": key[3], "risk_target": key[4],
                         **pooled, "mean_participant_accuracy": pp.accuracy.mean(), "acc_ci_lo": lo, "acc_ci_hi": hi,
                         "mean_participant_ccpm": pp.ccpm.mean(), "ccpm_ci_lo": clo, "ccpm_ci_hi": chi,
                         "participants_with_wrong": int((pp.wrong > 0).sum())})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out / "aggregate.csv", index=False)
    info = {"decoders": decs, "cpu_us_per_flash_update": {d: r[6] for d, r in zip(decs, res)},
            "taus": TAUS, "deadlines": DEADLINES, "risks": RISKS, "n_selections": sel.groupby(["decoder", "role", "policy"]).size().iloc[0:1].tolist(),
            "timing": "t_dec = onset_k - onset_1 + 0.7 s (epoch end); t_acq = onset_k - onset_1 + 0.25 s; recorded timestamps",
            "risk_rule": "calibration incorrect commits <= floor(risk * n_calibration_selections), maximise calibration correct/min"}
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    cols = ["decoder", "role", "policy", "accuracy", "wrong", "abstained", "selective_accuracy", "lat_median", "lat_p90",
            "lat_p95", "correct_within_3s", "correct_within_5s", "ccpm"]
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.precision", 3):
        print(agg[agg.role == "evaluation"][cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
