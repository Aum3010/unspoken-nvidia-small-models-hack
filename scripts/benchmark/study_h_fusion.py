"""Study H (bigP3BCI): EEG alone vs gaze alone vs EEG x gaze, on co-recorded real data.

    /home/claude/venv-eeg/bin/python scripts/benchmark/study_h_fusion.py build      # evidence per participant/fold (cached)
    /home/claude/venv-eeg/bin/python scripts/benchmark/study_h_fusion.py evaluate   # policies, stress test, online reference

Study H: 16 non-ALS participants, 9x8 grid (72 keys), 6 keys per flash, 125 ms SOA, 16 EEG channels + Tobii Pro X2-30 gaze
(normalised screen coordinates) in the same EDF at 256 Hz. Per participant: 3 calibration runs (Train, copy spelling,
14 repetitions = 168 flashes, 6 selections each) and online Test runs whose selections were stopped online.

Evaluation design (no label or gaze-model leakage):
  * outer leave-one-block-out over the full-length Train runs (3 blocks; one run each, H_08's 6 runs two each): the held-out block is scored only by models fitted on the
    other two runs;
  * inside the two training runs: decoder out-of-fold scores (fit one run, score the other) -> LLR + alpha; gaze affine
    map fitted on one run and applied to the other -> gaze sigma; these inner trajectories also choose tau/deadline;
  * policy thresholds are pooled across participants for a fold (all from training runs only);
  * Test runs (online-stopped, truncated evidence) are not used for offline policies; their recorded online outcomes are
    reported as the original study's measured reference.
Decision time t: EEG flash k counts once onset_k + 0.7 s <= t (relative to the first flash); gaze uses valid samples in
[first flash, first flash + t]. Grid of decision times every 125 ms.
Gaze stress test: a per-selection offset (SD in key widths) is added to the real mapped gaze in both calibration and
evaluation selections. This is a synthetic perturbation of real gaze, reported as a stress test, not as ALS data.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.study_f_early_decoders import EnsembleLdaRiemann  # noqa: E402
from services.eeg.datasets.bigp3bci import read_recording  # noqa: E402
from services.eeg.datasets.schema import AuditResult  # noqa: E402
from services.eeg.study_f import evidence as ev  # noqa: E402
from services.eeg.study_f import flash_stopping as fs  # noqa: E402
from services.eeg.study_f.stopping import softmax  # noqa: E402

RAW = ROOT / "data/raw"
HDIR = RAW / "bigp3bci/StudyH"
CACHE = ROOT / "data/processed/study_h"
RES = ROOT / "evaluations/results/eeg_gaze_fusion"
STEP = 0.125
EPOCH_END = 0.7
TAUS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999]
DEADLINES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, fs.INF]
RISKS = [0.0, 0.02, 0.05, 0.10]
BUDGETS = [1.0, 2.0, 3.0, 5.0, 10.0]                  # fixed time budget: yield at the budget if not confident
BUDGET_RISKS = [0.02, 0.05]
STRESS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
SIGMAS = np.exp(np.linspace(np.log(0.05), np.log(5.0), 40))
LAPSES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]   # 0.0 = pure Gaussian (the first run); chosen by calibration NLL
GRID_AREA = 72.0                                       # 9 x 8 keys, in key units


# ---- reading ------------------------------------------------------------------------------------------------------------

def read_run(path: Path) -> dict:
    import pyedflib

    d = read_recording(path, RAW, AuditResult())
    r = pyedflib.EdfReader(str(path))
    lab = r.getSignalLabels()
    sig = lambda n: r.readSignal(lab.index(n))  # noqa: E731
    eeg = np.stack([r.readSignal(i) for i, l in enumerate(lab) if l.startswith("EEG_")])
    vl, vr = sig("ETLeftEyeValid") > 0, sig("ETRightEyeValid") > 0
    gx = np.stack([sig("ETLeftEyeGazeX"), sig("ETRightEyeGazeX")])
    gy = np.stack([sig("ETLeftEyeGazeY"), sig("ETRightEyeGazeY")])
    r.close()
    w = np.stack([vl, vr]).astype(float)
    n = w.sum(0)
    valid = n > 0
    g = np.stack([(gx * w).sum(0), (gy * w).sum(0)]) / np.maximum(n, 1)
    chars = d["recording"]["_chars"]
    pos = np.array([[int(c.rsplit("_", 2)[2]), int(c.rsplit("_", 2)[1])] for c in chars], float)   # (col, row)
    ev_df = pd.DataFrame(d["events"])
    lab_df = pd.DataFrame(d["flash_labels"])
    sels, skipped = [], []
    for s, t in zip(d["selections"], d["targets"]):
        if t["target_index"] is None:            # trial with no CurrentTarget (audit ERROR); not a usable selection
            skipped.append(s["selection_id"])
            continue
        e = ev_df[ev_df.selection_id == s["selection_id"]].sort_values("flash_index")
        y = lab_df[lab_df.selection_id == s["selection_id"]].sort_values("flash_index").is_target_flash.to_numpy()
        shown = np.zeros((len(e), len(chars)), bool)
        for i, c in enumerate(e.chars_shown):
            shown[i, [int(x) - 1 for x in c.split(",")]] = True
        sels.append({"selection_id": s["selection_id"], "target": int(t["target_index"]) - 1,
                     "online_correct": t["online_selected_index"] == t["target_index"],
                     "onsets": e.onset_sample.to_numpy(), "shown": shown, "y": y})
    return {"meta": d["meta"], "fs": d["fs"], "eeg": eeg, "gaze": g, "valid": valid, "chars": chars, "pos": pos, "sels": sels,
            "skipped": skipped}


# ---- gaze model ---------------------------------------------------------------------------------------------------------

def sel_mean_gaze(run, s, t_end=None):
    a = s["onsets"][0]
    b = s["onsets"][-1] + int(EPOCH_END * run["fs"]) if t_end is None else a + int(t_end * run["fs"])
    v = run["valid"][a:b]
    return run["gaze"][:, a:b][:, v].mean(1) if v.any() else None


def fit_affine(runs) -> np.ndarray:
    """Least-squares affine map from mean screen gaze during a selection to the target's (col, row)."""
    X, Y = [], []
    for run in runs:
        for s in run["sels"]:
            g = sel_mean_gaze(run, s)
            if g is not None:
                X.append([g[0], g[1], 1.0]); Y.append(run["pos"][s["target"]])
    A, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
    return A                                  # (3, 2)


def gaze_track(run, s, A, times) -> np.ndarray:
    """(n_t, 2) mapped mean gaze (key units) over [first flash, first flash + t]; NaN where no valid sample yet."""
    a, f = s["onsets"][0], run["fs"]
    b = a + int(np.ceil(times[-1] * f)) + 1
    v = run["valid"][a:b].astype(float)
    cx = np.cumsum(run["gaze"][0, a:b] * v); cy = np.cumsum(run["gaze"][1, a:b] * v); cn = np.cumsum(v)
    out = np.full((len(times), 2), np.nan)
    for i, t in enumerate(times):
        k = min(int(t * f), len(cn) - 1)
        if cn[k] > 0:
            g = np.array([cx[k] / cn[k], cy[k] / cn[k], 1.0])
            out[i] = g @ A
    return out


def gaze_ll(track: np.ndarray, pos: np.ndarray, sigma: float, lapse: float = 0.0) -> np.ndarray:
    """log p(mean gaze | key c): isotropic Gaussian around the key, mixed with a uniform lapse over the grid so a gaze
    that lands on the wrong key cannot make the posterior near-certain. Columns are 0 where there is no valid gaze yet."""
    d2 = ((track[:, None, :] - pos[None]) ** 2).sum(-1)
    lg = -d2 / (2 * sigma ** 2) - np.log(2 * np.pi * sigma ** 2)
    with np.errstate(invalid="ignore"):
        ll = lg if lapse == 0 else np.logaddexp(np.log1p(-lapse) + lg, np.log(lapse / GRID_AREA))
    ll[np.isnan(ll)] = 0.0
    return ll


# ---- EEG evidence -------------------------------------------------------------------------------------------------------

def epochs(run):
    x = ev.bandpass(run["eeg"])
    return [ev.epoch(x, s["onsets"]) for s in run["sels"]]


def eeg_track(llr: np.ndarray, shown: np.ndarray, onsets: np.ndarray, f: float, times: np.ndarray) -> np.ndarray:
    rel = (onsets - onsets[0]) / f + EPOCH_END
    cum = np.cumsum(llr[:, None] * shown, 0)
    k = np.searchsorted(rel, times + 1e-9, side="right")         # flashes whose response window has ended
    out = np.zeros((len(times), shown.shape[1]))
    m = k > 0
    out[m] = cum[k[m] - 1]
    return out


def times_for(s, f):
    end = (s["onsets"][-1] - s["onsets"][0]) / f + EPOCH_END
    return np.arange(STEP, end + 1e-9, STEP)


def build_participant(pid: str) -> dict:
    tr_paths = sorted((HDIR / pid).rglob("*_Train*.edf"))
    runs = [read_run(p) for p in tr_paths]
    # three contiguous blocks of Train runs (3 runs -> one run per block; H_08 recorded 6 -> two per block)
    blocks = [list(b) for b in np.array_split(np.arange(len(runs)), 3)]
    Xr = [np.concatenate(epochs(r)) for r in runs]
    Yr = [np.concatenate([s["y"] for s in r["sels"]]) for r in runs]
    X = [np.concatenate([Xr[i] for i in b]) for b in blocks]
    Y = [np.concatenate([Yr[i] for i in b]) for b in blocks]
    R = [[runs[i] for i in b] for b in blocks]
    out = {"pid": pid, "folds": [], "blocks": [[tr_paths[i].name for i in b] for b in blocks]}
    for k in range(3):
        a, b = [i for i in range(3) if i != k]
        inner = {a: EnsembleLdaRiemann().fit(X[b], Y[b]).decision_function(X[a]),
                 b: EnsembleLdaRiemann().fit(X[a], Y[a]).decision_function(X[b])}
        g = ev.GaussianLLR().fit(np.concatenate([inner[a], inner[b]]), np.concatenate([Y[a], Y[b]]))
        final = EnsembleLdaRiemann().fit(np.concatenate([X[a], X[b]]), np.concatenate([Y[a], Y[b]]))
        s_out = final.decision_function(X[k])
        A_in = {a: fit_affine(R[b]), b: fit_affine(R[a])}
        A_out = fit_affine(R[a] + R[b])
        fold = {"held_out_runs": out["blocks"][k], "cal": [], "eval": []}
        for role, idx, scores, Amap in (("cal", a, inner[a], A_in[a]), ("cal", b, inner[b], A_in[b]), ("eval", k, s_out, A_out)):
            off = 0
            for run in R[idx]:
                for s in run["sels"]:
                    n = len(s["onsets"])
                    llr = g(scores[off:off + n]); off += n
                    t = times_for(s, run["fs"])
                    fold[role].append({"selection_id": s["selection_id"], "participant_id": pid, "target": s["target"], "times": t,
                                       "L": eeg_track(llr, s["shown"], s["onsets"], run["fs"], t),
                                       "gaze": gaze_track(run, s, Amap, t), "pos": run["pos"]})
            assert off == len(scores)
        out["folds"].append(fold)
    out["skipped"] = {p.name: r["skipped"] for p, r in zip(tr_paths, runs) if r["skipped"]}
    # online reference from Test runs (recorded online decisions; evidence was stopped online)
    online = []
    for p in sorted((HDIR / pid).rglob("*_Test*.edf")):
        run = read_run(p)
        for s in run["sels"]:
            on = s["onsets"]
            online.append({"participant_id": pid, "condition": run["meta"]["condition"], "selection_id": s["selection_id"],
                           "n_flashes": len(on), "online_correct": bool(s["online_correct"]),
                           "stim_time_s": (on[-1] - on[0]) / run["fs"] + STEP})
    out["online"] = online
    return out


def build():
    CACHE.mkdir(parents=True, exist_ok=True)
    for pdir in sorted(HDIR.glob("H_*")):
        f = CACHE / f"{pdir.name}.pkl"
        if f.exists():
            continue
        t0 = time.time()
        res = build_participant(pdir.name)
        f.write_bytes(pickle.dumps(res))
        print(pdir.name, f"{time.time() - t0:.0f}s", flush=True)


# ---- policies -----------------------------------------------------------------------------------------------------------

def fit_alpha(items):
    return fs.fit_alpha([it["L"] for it in items], [it["target"] for it in items])


def fit_gaze(items, tracks):
    """(sigma, lapse) minimising the calibration negative log posterior of the target over all decision times."""
    best, arg = np.inf, (1.0, 0.0)
    for lp in LAPSES:
        for sg in SIGMAS:
            nll = 0.0
            for it, tr in zip(items, tracks):
                z = gaze_ll(tr, it["pos"], sg, lp)
                m = z.max(1, keepdims=True)
                nll += (np.log(np.exp(z - m).sum(1)) + m[:, 0] - z[:, it["target"]]).sum()
            if nll < best:
                best, arg = nll, (float(sg), lp)
    return arg


def traj(score: np.ndarray, times: np.ndarray) -> fs.Trajectory:
    P = softmax(score, axis=-1)
    re = np.zeros(len(times), bool); re[-1] = True
    return fs.Trajectory(P.max(1), P.argmax(1), times, times, re)


def perturb(items, level, rng):
    if level == 0:
        return [it["gaze"] for it in items]
    return [it["gaze"] + rng.normal(0, level, 2) for it in items]


def run_fold(cal, evl, level, rng, risks=RISKS):
    gc, ge = perturb(cal, level, rng), perturb(evl, level, rng)
    a = fit_alpha(cal)
    sg, lp = fit_gaze(cal, gc)
    scores = {
        "eeg": (lambda it, g: a * it["L"]),
        "gaze": (lambda it, g: gaze_ll(g, it["pos"], sg, lp)),
        "fused": (lambda it, g: a * it["L"] + gaze_ll(g, it["pos"], sg, lp)),
    }
    rows, chosen = [], {"alpha": a, "sigma": sg, "lapse": lp}
    for mod, fn in scores.items():
        trc = [traj(fn(it, g), it["times"]) for it, g in zip(cal, gc)]
        tre = [traj(fn(it, g), it["times"]) for it, g in zip(evl, ge)]
        for r in risks:
            c = fs.choose_params(trc, [it["target"] for it in cal], r, TAUS, DEADLINES)
            chosen[f"{mod}_{r:g}"] = c
            pol = fs.Policy(c["tau"], c["deadline"])
            for it, tr in zip(evl, tre):
                o = fs.outcome(tr, pol, it["target"])
                rows.append({**o, "modality": mod, "risk_target": r, "participant_id": it["participant_id"],
                             "selection_id": it["selection_id"], "gaze_offset_sd": level})
        for r in BUDGET_RISKS:      # same risk control, deadline fixed to the budget (tau still chosen on calibration)
            for D in BUDGETS:
                c = fs.choose_params(trc, [it["target"] for it in cal], r, TAUS, [D])
                chosen[f"{mod}_{r:g}_budget{D:g}"] = c
                pol = fs.Policy(c["tau"], c["deadline"])
                for it, tr in zip(evl, tre):
                    o = fs.outcome(tr, pol, it["target"])
                    rows.append({**o, "modality": mod, "risk_target": r, "budget_s": D, "participant_id": it["participant_id"],
                                 "selection_id": it["selection_id"], "gaze_offset_sd": level})
        if level == 0:     # fixed-time reference: argmax accuracy at a few decision times (information, not policy)
            for it, tr in zip(evl, tre):
                for t in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
                    j = min(np.searchsorted(tr.t_dec, t - 1e-9), len(tr.t_dec) - 1)
                    rows.append({"modality": mod, "risk_target": None, "fixed_time": t, "correct": bool(tr.arg[j] == it["target"]),
                                 "participant_id": it["participant_id"], "selection_id": it["selection_id"], "gaze_offset_sd": 0.0})
    return rows, chosen


def evaluate():
    RES.mkdir(parents=True, exist_ok=True)
    parts = [pickle.loads(f.read_bytes()) for f in sorted(CACHE.glob("H_*.pkl"))]
    rng = np.random.default_rng(0)
    rows, chosen = [], []
    for level in STRESS:
        for k in range(3):
            cal = [it for p in parts for it in p["folds"][k]["cal"]]
            evl = [it for p in parts for it in p["folds"][k]["eval"]]
            r, c = run_fold(cal, evl, level, rng)
            for x in r:
                x["fold"] = k
            rows += r
            chosen.append({"fold": k, "gaze_offset_sd": level, **c})
        print("level", level, flush=True)
    df = pd.DataFrame(rows)
    budget = df[df.budget_s.notna()].copy()
    df = df[df.budget_s.isna()]
    pol = df[df.risk_target.notna() & df.get("fixed_time", pd.Series(np.nan, index=df.index)).isna()].copy()
    pol.to_csv(RES / "per_selection.csv.gz", index=False)
    fixed = df[df.get("fixed_time", pd.Series(np.nan, index=df.index)).notna()]
    fixed.groupby(["modality", "fixed_time"]).correct.mean().unstack(0).to_csv(RES / "argmax_accuracy_at_fixed_times.csv")
    fixed.groupby(["modality", "fixed_time", "participant_id"]).correct.mean().unstack(0).to_csv(RES / "argmax_accuracy_at_fixed_times_per_participant.csv")
    (RES / "chosen_params.json").write_text(json.dumps(chosen, indent=1, default=float))
    agg = pol.groupby(["gaze_offset_sd", "modality", "risk_target"]).apply(fs.summarize, include_groups=False).apply(pd.Series).reset_index()
    per = pol.groupby(["gaze_offset_sd", "modality", "risk_target", "participant_id"]).apply(fs.summarize, include_groups=False).apply(pd.Series).reset_index()
    agg.to_csv(RES / "aggregate.csv", index=False)
    bud = budget.groupby(["gaze_offset_sd", "modality", "risk_target", "budget_s"]).apply(fs.summarize, include_groups=False).apply(pd.Series).reset_index()
    bud.to_csv(RES / "budget_frontier.csv", index=False)
    budget.to_csv(RES / "per_selection_budget.csv.gz", index=False)
    per.to_csv(RES / "per_participant.csv", index=False)
    online = pd.DataFrame([o for p in parts for o in p["online"]])
    on = online.groupby(["condition", "participant_id"]).agg(acc=("online_correct", "mean"), flashes=("n_flashes", "mean"),
                                                              stim_s=("stim_time_s", "mean"), n=("n_flashes", "size")).reset_index()
    on.to_csv(RES / "online_reference_per_participant.csv", index=False)
    on.groupby("condition")[["acc", "flashes", "stim_s", "n"]].mean().to_csv(RES / "online_reference.csv")
    cols = ["gaze_offset_sd", "modality", "risk_target", "n", "accuracy", "wrong", "abstained", "selective_accuracy", "lat_median",
            "lat_p90", "correct_within_1s", "correct_within_2s", "correct_within_3s", "ccpm"]
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.precision", 3):
        print(agg[cols].to_string(index=False))
        print(bud[bud.gaze_offset_sd.isin([0.0, 1.0, 3.0])][["gaze_offset_sd", "modality", "risk_target", "budget_s", "accuracy", "wrong",
                                                              "abstained", "lat_median", "ccpm"]].to_string(index=False))
        print(on.groupby("condition")[["acc", "flashes", "stim_s", "n"]].mean())


if __name__ == "__main__":
    {"build": build, "evaluate": evaluate}[sys.argv[1]]()
