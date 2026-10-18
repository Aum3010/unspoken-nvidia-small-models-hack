"""Study F benchmark tests. Unit tests use hand-built evidence; real-data tests are skipped (never faked) when the
recordings / frozen artifacts are not on disk."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import evidence as ev  # noqa: E402
from services.eeg.study_f import stopping as st  # noqa: E402

AUDIT = ROOT / "data" / "processed" / "audit"
EVD = ROOT / "data" / "evidence" / "study_f"
HAVE_AUDIT = (AUDIT / "stimulus_events.csv").exists()
HAVE_EVD = (EVD / "manifest.json").exists()
needs_audit = pytest.mark.skipif(not HAVE_AUDIT, reason="audit tables not on disk")
needs_evd = pytest.mark.skipif(not HAVE_EVD, reason="frozen evidence not built")


def _script(name):
    p = ROOT / "scripts" / "benchmark" / f"{name}.py"
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def shown(*groups):
    return np.stack([ev.parse_chars(",".join(str(c) for c in g)) for g in groups])


# ---- stimulus -> character mapping and protocol groups ------------------------------------------------------------

def test_flash_increment_credits_only_lit_characters():
    llr = np.array([2.0, -1.0, 0.5])
    inc = st.flash_increments(llr, shown([1, 2, 3], [3, 4], [1, 36]))
    assert inc[0] == pytest.approx(2.5) and inc[1] == pytest.approx(2.0) and inc[2] == pytest.approx(1.0)
    assert inc[3] == pytest.approx(-1.0) and inc[35] == pytest.approx(0.5) and inc[10] == 0


@needs_audit
def test_protocol_flash_groups_every_repetition_is_a_partition_alternating_10_8():
    e = pd.read_csv(AUDIT / "stimulus_events.csv")
    s = pd.read_csv(AUDIT / "selections.csv")
    r = pd.read_csv(AUDIT / "runs.csv")
    fixed = set(r[r.condition.isin(["Static", "CBCol"])].run_id)
    e = e[e.selection_id.isin(s[s.run_id.isin(fixed)].selection_id)]
    for (sid, rep), g in e.groupby(["selection_id", "repetition_index"]):
        cs = [c for cell in g.chars_shown for c in cell.split(",")]
        assert sorted(map(int, cs)) == list(range(1, 37)), (sid, rep)      # each of the 36 characters exactly once
        assert len(g) == (10 if rep % 2 == 0 else 8), (sid, rep)
        assert all(3 <= len(c.split(",")) <= 5 for c in g.chars_shown)


@needs_audit
def test_target_flag_matches_character_flags_in_every_selection_and_condition():
    e = pd.read_csv(AUDIT / "stimulus_events.csv").merge(pd.read_csv(AUDIT / "verifier_flash_labels.csv"), on=["selection_id", "flash_index"])
    t = pd.read_csv(AUDIT / "verifier_targets.csv").set_index("selection_id").target_index
    lit = [str(int(t[sid])) in cell.split(",") for sid, cell in zip(e.selection_id, e.chars_shown)]
    assert (np.array(lit, int) == e.is_target_flash.values).all()


# ---- splits and leakage ------------------------------------------------------------------------------------------------

@needs_audit
def test_splits_never_mix_sessions():
    sp = pd.read_csv(AUDIT / "splits_session_3way.csv")
    assert set(sp[sp.role == "calibration"].session) == {1, 2}
    assert set(sp[sp.role.isin(["evaluation", "evaluation_secondary"])].session) == {3}
    assert not (set(sp[sp.role == "calibration"].run_id) & set(sp[sp.role.str.startswith("evaluation")].run_id))
    assert set(sp[sp.role == "evaluation"].condition) == {"Static"}
    assert set(sp[sp.role == "evaluation_secondary"].condition) == {"CBCol"}
    assert not set(sp[sp.role == "dynamic_replay_only"].condition) - {"Dyn", "DynBigram"}


@needs_evd
def test_frozen_evidence_roles_match_sessions_and_hide_labels():
    roles = pd.read_csv(EVD / "selection_roles.csv")
    assert set(roles[roles.role == "calibration"].session) == {1, 2}
    assert set(roles[roles.role != "calibration"].session) == {3}
    flash_cols = pd.read_csv(EVD / "flash_evidence.csv.gz", nrows=2).columns
    z = np.load(EVD / "rep_evidence_xdawn_riemann_lr.npz")
    forbidden = {"target", "target_index", "is_target_flash", "target_char", "y"}
    assert not forbidden & set(flash_cols) and not forbidden & set(z.files)
    assert "target_index" in pd.read_csv(EVD / "ground_truth.csv").columns


def test_calibration_fit_uses_only_calibration_arguments():
    """alpha depends only on the evidence/targets passed in: perturbing unrelated (evaluation) data cannot change it."""
    rng = np.random.default_rng(1)
    cal = [np.cumsum(rng.normal(0, 1, (5, 36)) + np.eye(36)[3] * 1.5, 0) for _ in range(20)]
    a1 = st.fit_alpha(cal, [3] * 20)
    _eval = [rng.normal(0, 50, (5, 36)) for _ in range(5)]                  # never passed in
    assert st.fit_alpha(cal, [3] * 20) == a1


def test_gaussian_llr_fitted_only_on_given_scores():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(1600)].astype(int)
    s = np.where(y == 1, rng.normal(1, 1, len(y)), rng.normal(0, 1, len(y)))
    g = ev.GaussianLLR().fit(s, y)
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    explicit = -0.5 * ((x - g.m1) ** 2 - (x - g.m0) ** 2) / g.var
    assert np.allclose(g(x), explicit) and g(np.array([3.0]))[0] > g(np.array([-3.0]))[0]


# ---- frozen-evidence reproducibility ------------------------------------------------------------------------------------

@needs_evd
def test_manifest_hashes_match_artifacts():
    m = json.loads((EVD / "manifest.json").read_text())
    for name, digest in m["artifacts"].items():
        assert ev.sha(EVD / name) == digest, name


@pytest.mark.skipif(not (ROOT / "data/evidence/smoke/flash_evidence.csv.gz").exists() or not HAVE_EVD, reason="smoke rebuild absent")
def test_rebuild_reproduces_frozen_scores():
    a = pd.read_csv(EVD / "flash_evidence.csv.gz").query("participant_id == 'F_03'").reset_index(drop=True)
    b = pd.read_csv(ROOT / "data/evidence/smoke/flash_evidence.csv.gz").reset_index(drop=True)
    assert len(a) == len(b) and np.allclose(a.score, b.score, atol=1e-4) and np.allclose(a.llr, b.llr, atol=1e-4)


# ---- Bayesian update, exhaustion, abstention ------------------------------------------------------------------------------

def test_bayesian_posterior_equals_explicit_naive_bayes():
    llr = np.array([1.2, -0.4, 0.9, 0.3])
    groups = [[1, 2], [3], [1, 3], [2, 3]]
    L = st.flash_increments(llr, shown(*groups))[:3]
    # explicit: for hypothesis c, product over flashes of p(s|T)/p(s|N) when c lit, else 1 (shared N likelihood cancels)
    lik = np.array([np.exp(sum(l for l, g in zip(llr, groups) if c in g)) for c in (1, 2, 3)])
    expect = lik / lik.sum()
    assert np.allclose(st.softmax(L), expect)


def _L(n, winner=2, gain=1.5, chars=36):
    inc = np.zeros(chars); inc[winner] = gain
    return np.cumsum(np.tile(inc, (n, 1)), 0)


def test_threshold_stop_commits_at_first_confident_repetition():
    o = st.threshold_stop(_L(14), 14, tau=0.9)
    assert o.committed and o.char == 2 and o.n_reps < 14 and not o.exhausted
    assert o.confidence >= 0.9


def test_evidence_exhaustion_fixed_and_threshold():
    L = _L(3, gain=0.05)
    f = st.fixed_reps(L, 3, 7)
    assert f.n_reps == 3 and f.exhausted
    t = st.threshold_stop(L, 3, tau=0.99)
    assert t.n_reps == 3 and t.exhausted and t.committed          # commit-on-exhaustion without abstention
    a = st.threshold_stop(L, 3, tau=0.99, abstain=True)
    assert not a.committed and a.char is None and a.exhausted


def test_policy_never_reads_beyond_recorded_repetitions():
    L = _L(4)
    assert st.fixed_reps(L, 4, 14).n_reps == 4
    assert st.threshold_stop(L, 4, tau=0.9999999, max_reps=14).n_reps <= 4


def test_abstention_when_confidence_low_even_with_full_evidence():
    L = _L(14, gain=0.001)
    assert not st.threshold_stop(L, 14, 0.9, abstain=True).committed
    assert not st.fixed_reps(L, 14, 14, abstain_below=0.9).committed
    assert st.fixed_reps(_L(14), 14, 14, abstain_below=0.9).committed


# ---- timestamp-based latency accounting ----------------------------------------------------------------------------------

def test_latency_uses_recorded_timestamps_not_assumed_soa():
    m = _script("run_study_f_baselines")
    it = {"first": 10.0, "last": np.array([12.25, 14.0, 17.5]), "nfl": np.array([10, 8, 10])}
    c = m.cost(it, 2)
    assert c["flashes"] == 18 and c["reps"] == 2
    assert c["acq_s"] == pytest.approx(14.0 - 10.0 + m.SOA_S)
    assert c["latency_s"] == pytest.approx(14.0 - 10.0 + m.EPOCH_END_S)


@needs_evd
def test_truncated_selection_is_exhausted_after_four_repetitions():
    m = _script("run_study_f_baselines")
    items = {i["sid"]: i for i in m.load(EVD, "xdawn_riemann_lr")}
    sid = "bigp3bci/F_21/SE003/Test/Static01#t00"
    if sid not in items:
        pytest.skip("F_21 not in evidence")
    assert items[sid]["n_avail"] == 4
    assert m.run_policy(items[sid], {"kind": "fixed", "k": 14}, {"F_21": 1.0})["exhausted"]
