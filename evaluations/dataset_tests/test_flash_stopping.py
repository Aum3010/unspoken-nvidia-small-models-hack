"""Flash-level stopping: evidence for incomplete flash groups, causality, timing, deadline/yield, calibration-only choice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.eeg.study_f import flash_stopping as fs

ROOT = Path(__file__).resolve().parents[2]
EVD = ROOT / "data/evidence/study_f"
RES = ROOT / "evaluations/results/eeg_fast_selection"


def _sel(llr, groups, onsets=None, reps=None):
    n = len(llr)
    onsets = np.arange(n) * 0.25 if onsets is None else np.asarray(onsets, float)
    shown = np.zeros((n, 36), bool)
    for i, g in enumerate(groups):
        shown[i, g] = True
    reps = np.zeros(n, int) if reps is None else np.asarray(reps)
    rep_end = np.r_[reps[1:] != reps[:-1], True]
    return fs.FlashSelection("s", "P", "calibration", np.asarray(llr, float), shown, onsets - onsets[0] + fs.EPOCH_END_S,
                             onsets - onsets[0] + fs.SOA_S, rep_end)


def test_incomplete_group_evidence_leaves_unlit_characters_at_prior():
    s = _sel([2.0, -1.0], [[0, 1, 2], [3, 4]])
    L = s.evidence()
    assert np.allclose(L[0, :3], 2.0) and np.allclose(L[0, 3:], 0.0)
    assert np.allclose(L[1, 3:5], -1.0) and np.allclose(L[1, 5:], 0.0)
    p = fs.trajectory(s, 1.0)
    assert p.arg[0] in (0, 1, 2)


def test_decision_time_uses_recorded_onsets_and_response_window():
    s = _sel([0.1, 0.1, 0.1], [[0], [1], [2]], onsets=[10.0, 10.25, 10.6])
    assert np.allclose(s.t_dec, [0.7, 0.95, 1.3])
    assert np.allclose(s.t_acq, [0.25, 0.5, 0.85])


def test_policy_is_causal_future_flashes_do_not_change_an_earlier_decision():
    rng = np.random.default_rng(0)
    groups = [list(rng.choice(36, 4, replace=False)) for _ in range(40)]
    llr = rng.normal(0, 1, 40)
    llr[:6] += 4 * np.array([0 in g for g in groups[:6]])
    a = fs.trajectory(_sel(llr, groups), 1.0)
    pol = fs.Policy(0.6)
    d = fs.run(a, pol)
    assert d["commit"]
    k = d["k"]
    llr2 = llr.copy()
    llr2[k + 1:] = rng.normal(0, 5, 40 - k - 1)       # rewrite the future
    b = fs.trajectory(_sel(llr2, groups), 1.0)
    assert fs.run(b, pol) == d


def test_deadline_yields_at_the_deadline_and_exhaustion_at_the_last_flash():
    s = _sel([0.0] * 20, [[i % 36] for i in range(20)])
    tr = fs.trajectory(s, 1.0)
    d = fs.run(tr, fs.Policy(0.99, deadline=2.0))
    assert d["reason"] == "deadline" and not d["commit"] and d["t_dec"] == 2.0
    e = fs.run(tr, fs.Policy(0.99))
    assert e["reason"] == "exhausted" and not e["commit"] and e["t_dec"] == pytest.approx(s.t_dec[-1])
    c = fs.run(tr, fs.Policy(0.99, on_fail="commit"))
    assert c["commit"]


def test_repetition_granularity_only_decides_at_repetition_ends():
    reps = [0] * 10 + [1] * 8
    llr = [5.0] + [0.0] * 17
    s = _sel(llr, [[0]] + [[i] for i in range(1, 18)], reps=reps)
    tr = fs.trajectory(s, 1.0)
    assert fs.run(tr, fs.Policy(0.5))["k"] == 0
    assert fs.run(tr, fs.Policy(0.5, granularity="repetition"))["k"] == 9


def test_choose_params_meets_calibration_risk_and_yields_immediately_when_useless():
    rng = np.random.default_rng(1)
    trs, tg = [], []
    for i in range(60):
        groups = [list(rng.choice(36, 4, replace=False)) for _ in range(60)]
        llr = rng.normal(0, 1, 60) + 1.5 * np.array([0 in g for g in groups])
        trs.append(fs.trajectory(_sel(llr, groups), 1.0)); tg.append(0)
    c = fs.choose_params(trs, tg, 0.05, [0.5, 0.9, 0.99], [2.0, 5.0, fs.INF])
    assert c["cal_wrong"] <= 3
    pol = fs.Policy(c["tau"], c["deadline"])
    assert sum(fs.outcome(t, pol, 0)["wrong"] for t in trs) == c["cal_wrong"]
    # pure noise: nothing can be committed safely -> never commit, yield at time 0
    noise = [fs.trajectory(_sel(rng.normal(0, 1, 30), [list(rng.choice(36, 4, replace=False)) for _ in range(30)]), 1.0) for _ in range(60)]
    c0 = fs.choose_params(noise, [0] * 60, 0.0, [0.9, 0.99], [5.0, fs.INF])
    if c0.get("never_commit"):
        o = fs.outcome(noise[0], fs.Policy(c0["tau"], c0["deadline"]), 0)
        assert o["abstained"] and o["t_dec"] == 0.0


needs_data = pytest.mark.skipif(not (EVD / "flash_evidence.csv.gz").exists() or not (RES / "aggregate.csv").exists(),
                                reason="needs data/evidence/study_f and a run of run_study_f_fast_selection.py")


@needs_data
def test_baseline_reproduces_original_benchmark():
    agg = pd.read_csv(RES / "aggregate.csv")
    b = agg[(agg.decoder == "xdawn_riemann_lr") & (agg.role == "evaluation") & (agg.policy == "baseline_rep_bayes_tau0.99_commit")].iloc[0]
    assert b.n == 120 and b.accuracy == pytest.approx(0.758, abs=1e-3) and b.wrong == 29
    assert b.lat_mean == pytest.approx(21.6, abs=0.05)


@needs_data
def test_chosen_params_are_a_function_of_calibration_only():
    flash = pd.read_csv(EVD / "flash_evidence.csv.gz")
    gt = pd.read_csv(EVD / "ground_truth.csv")
    tg = {str(a): int(b) - 1 for a, b in zip(gt.selection_id, gt.target_index)}
    ch = json.loads((RES / "chosen_params_xdawn_riemann_lr.json").read_text())
    cal = [s for s in fs.load_selections(flash, "xdawn_riemann_lr", ["calibration"]) if s.participant_id == "F_08"]
    assert {s.role for s in cal} == {"calibration"} and len(cal) == 60
    a = fs.fit_alpha([s.evidence() for s in cal], [tg[s.selection_id] for s in cal])
    assert a == pytest.approx(ch["alpha_participant"]["F_08"])
    from scripts.benchmark.run_study_f_fast_selection import DEADLINES, TAUS
    c = fs.choose_params([fs.trajectory(s, a) for s in cal], [tg[s.selection_id] for s in cal], 0.05, TAUS, DEADLINES)
    saved = ch["risks"]["0.05"]["flash_participant"]["F_08"]
    assert (c["tau"], c["deadline"]) == (saved["tau"], saved["deadline"])
