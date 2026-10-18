"""Flash-granularity sequential evidence accumulation and stopping for Study F (pure numpy; no EEG, no labels in policies).

Evidence after the first k flashes of a selection (k = 1 .. n_recorded):
    L_c(k) = sum_{j <= k, c in G_j} l_j          (l_j = per-flash log-likelihood ratio, G_j = characters lit by flash j)
    p(c | k) = softmax_c(alpha * L_c(k))           uniform prior over all 36 characters
A character that has not been lit yet has L_c = 0, i.e. exactly the prior odds against the lit ones; this is the correct
naive-Bayes posterior for incomplete flash groups, so no repetition has to be complete before a decision.

Timing (recorded stimulus timestamps, never wall clock):
    decision time t_dec(k) = onset_k - onset_1 + EPOCH_END_S   (flash k's 0.7 s response window must have been recorded)
    acquisition   t_acq(k) = onset_k - onset_1 + SOA_S
CPU inference time is measured separately and is not part of t_dec.

A policy sees only (pmax, argmax, t_dec) up to the current flash, never later flashes or the target. Its parameters
(alpha, tau, deadline) are chosen on calibration sessions only (`choose_params`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from services.eeg.study_f.stopping import softmax

N_CHARS = 36
SOA_S, EPOCH_END_S = 0.25, 0.7
INF = float("inf")


@dataclass
class FlashSelection:
    selection_id: str
    participant_id: str
    role: str
    llr: np.ndarray        # (n,)
    shown: np.ndarray      # (n, 36) bool
    t_dec: np.ndarray      # (n,) decision time after flash k
    t_acq: np.ndarray      # (n,)
    rep_end: np.ndarray    # (n,) bool: flash k is the last recorded flash of a repetition (repetition-boundary decisions)

    @property
    def n(self) -> int:
        return len(self.llr)

    def evidence(self) -> np.ndarray:
        """(n, 36) cumulative character log-evidence after each flash prefix."""
        return np.cumsum(self.llr[:, None] * self.shown, axis=0)


def parse_shown(cells) -> np.ndarray:
    out = np.zeros((len(cells), N_CHARS), bool)
    for i, c in enumerate(cells):
        out[i, [int(x) - 1 for x in str(c).split(",")]] = True
    return out


def load_selections(flash: pd.DataFrame, decoder: str, roles=None) -> list[FlashSelection]:
    f = flash[flash.decoder == decoder]
    if roles is not None:
        f = f[f.role.isin(roles)]
    out = []
    for sid, g in f.groupby("selection_id", sort=True):
        g = g.sort_values("flash_index")
        on = g.onset_s.to_numpy(float)
        rep = g.repetition_index.to_numpy()
        rep_end = np.r_[rep[1:] != rep[:-1], True]
        out.append(FlashSelection(sid, g.participant_id.iloc[0], g.role.iloc[0], g.llr.to_numpy(float),
                                  parse_shown(g.chars_shown.to_numpy()), on - on[0] + EPOCH_END_S, on - on[0] + SOA_S, rep_end))
    return out


# ---- trajectories ---------------------------------------------------------------------------------------------------

@dataclass
class Trajectory:
    pmax: np.ndarray
    arg: np.ndarray
    t_dec: np.ndarray
    t_acq: np.ndarray
    rep_end: np.ndarray


def trajectory(sel: FlashSelection, alpha: float, L: np.ndarray | None = None) -> Trajectory:
    P = softmax(alpha * (sel.evidence() if L is None else L), axis=-1)
    return Trajectory(P.max(1), P.argmax(1), sel.t_dec, sel.t_acq, sel.rep_end)


def fit_alpha(Ls: list[np.ndarray], targets: list[int], grid: np.ndarray | None = None) -> float:
    """Inverse temperature minimising mean character NLL over every flash prefix of calibration selections."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(3.0), 50)) if grid is None else grid
    nll = np.zeros(len(grid))
    n = 0
    for L, t in zip(Ls, targets):
        for i, a in enumerate(grid):
            z = a * L
            m = z.max(1, keepdims=True)
            nll[i] += (np.log(np.exp(z - m).sum(1)) + m[:, 0] - z[:, t]).sum()
        n += len(L)
    return float(grid[int(np.argmin(nll))])


# ---- policy ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    """Stop at the first allowed flash k with pmax >= tau and t_dec <= deadline -> commit argmax.
    Otherwise stop at the last allowed flash before the deadline (or the end of the recording) and
    `on_fail` = "abstain" (yield to another modality) or "commit" (commit the current argmax)."""

    tau: float
    deadline: float = INF
    on_fail: str = "abstain"
    granularity: str = "flash"   # "flash" | "repetition"


def run(tr: Trajectory, pol: Policy) -> dict:
    """Decision for one selection. Returns the decision flash index `k` (-1 = none), decision time and reason."""
    allowed = np.ones(len(tr.pmax), bool) if pol.granularity == "flash" else tr.rep_end.copy()
    allowed[-1] = True                                   # the end of the recording is always a decision point
    in_time = tr.t_dec <= pol.deadline
    ok = allowed & in_time & (tr.pmax >= pol.tau)
    if ok.any():
        k = int(np.argmax(ok))
        return {"k": k, "commit": True, "char": int(tr.arg[k]), "reason": "threshold", "t_dec": float(tr.t_dec[k])}
    last = int(np.flatnonzero(allowed)[-1])
    if tr.t_dec[last] <= pol.deadline:                   # recording ran out first
        k, reason, t = last, "exhausted", float(tr.t_dec[last])
    else:                                                # deadline came first: decide at the deadline
        cand = np.flatnonzero(allowed & in_time)
        k, reason, t = (int(cand[-1]) if len(cand) else -1), "deadline", float(pol.deadline)
    commit = pol.on_fail == "commit" and k >= 0
    return {"k": k, "commit": commit, "char": int(tr.arg[k]) if commit else None, "reason": reason, "t_dec": t}


def outcome(tr: Trajectory, pol: Policy, target: int) -> dict:
    d = run(tr, pol)
    k, t_dec = d["k"], d["t_dec"]
    correct = bool(d["commit"] and d["char"] == target)
    return {"committed": d["commit"], "correct": correct, "wrong": bool(d["commit"] and not correct),
            "abstained": not d["commit"], "reason": d["reason"], "flashes": int((tr.t_dec <= t_dec + 1e-9).sum()),
            "t_dec": t_dec, "t_acq": max(t_dec - EPOCH_END_S + SOA_S, 0.0),
            "conf": float(tr.pmax[k]) if k >= 0 else float("nan")}


# ---- metrics --------------------------------------------------------------------------------------------------------

WITHIN = (1.0, 2.0, 3.0, 5.0)


def summarize(df: pd.DataFrame) -> dict:
    """Metrics over a set of attempted selections (one row per attempt)."""
    n = len(df)
    com = df.committed.sum()
    t = df.t_dec.sum()
    out = {"n": n, "accuracy": df.correct.mean(), "wrong": int(df.wrong.sum()), "wrong_per_120": df.wrong.mean() * 120,
           "abstained": int(df.abstained.sum()), "abstain_rate": df.abstained.mean(), "coverage": com / n,
           "selective_accuracy": df.correct.sum() / com if com else float("nan"),
           "lat_mean": df.t_dec.mean(), "lat_median": df.t_dec.median(), "lat_p90": df.t_dec.quantile(0.9), "lat_p95": df.t_dec.quantile(0.95),
           "lat_median_correct": df.t_dec[df.correct].median() if df.correct.any() else float("nan"),
           "mean_flashes": df.flashes.mean(), "ccpm": df.correct.sum() / (t / 60) if t > 0 else 0.0}
    for s in WITHIN:
        out[f"correct_within_{s:g}s"] = float((df.correct & (df.t_dec <= s)).mean())
    return out


# ---- calibration-only parameter choice at matched risk ----------------------------------------------------------------

def choose_params(trs: list[Trajectory], targets: list[int], risk: float, taus, deadlines, granularity: str = "flash") -> dict:
    """Pick (tau, deadline) on calibration selections: among settings whose calibration incorrect commits are
    <= floor(risk * n) (incorrect commits per attempted selection), maximise calibration correct characters per minute.
    Abstain on failure. Ties go to the setting that spends less total time. If no feasible setting yields any correct
    calibration selection, the policy never commits and yields immediately (decision time 0)."""
    n = len(trs)
    allowed_wrong = int(np.floor(risk * n + 1e-9))
    best = None
    for tau in taus:
        for dl in deadlines:
            pol = Policy(float(tau), float(dl), "abstain", granularity)
            rows = [outcome(tr, pol, t) for tr, t in zip(trs, targets)]
            w = sum(r["wrong"] for r in rows)
            if w > allowed_wrong:
                continue
            c = sum(r["correct"] for r in rows)
            tt = sum(r["t_dec"] for r in rows)
            score = (c / (tt / 60) if tt > 0 else 0.0, -tt, tau)
            if best is None or score > best[0]:
                best = (score, {"tau": float(tau), "deadline": float(dl), "cal_wrong": int(w), "cal_correct": int(c),
                                "cal_n": n, "cal_ccpm": score[0], "allowed_wrong": allowed_wrong})
    if best is None or best[1]["cal_correct"] == 0:   # EEG cannot help this user at this risk: yield immediately
        return {"tau": 1.01, "deadline": 0.0, "cal_wrong": 0, "cal_correct": 0, "cal_n": n, "cal_ccpm": 0.0,
                "allowed_wrong": allowed_wrong, "never_commit": True}
    return best[1]


def oracle_times(sel: FlashSelection, alpha: float, target: int) -> dict:
    """Information reference, NOT a policy: first flash whose argmax is the target, and first flash from which the argmax
    stays the target until the end of the recording. A policy cannot know either time."""
    tr = trajectory(sel, alpha)
    hit = tr.arg == target
    first = int(np.argmax(hit)) if hit.any() else None
    stable = None
    if hit[-1]:
        miss = np.flatnonzero(~hit)
        stable = int(miss[-1] + 1) if len(miss) else 0
    return {"first_correct_t": float(sel.t_dec[first]) if first is not None else float("nan"),
            "stable_correct_t": float(sel.t_dec[stable]) if stable is not None else float("nan")}
