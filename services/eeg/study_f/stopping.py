"""Stopping policies over frozen per-repetition evidence (pure numpy, no EEG, no labels except at scoring time).

Evidence model (Bayesian update rule, documented in docs/experiments/EEG_STOPPING_BENCHMARK.md):
    Each flash j has a classifier score s_j and a log-likelihood ratio  l_j = log p(s_j | target flash) - log p(s_j | non-target).
    Under the hypothesis "character c is the target", flash j is a target flash iff c was lit in flash j, so
        log p(data | c) = const + sum_{j : c in G_j} l_j.
    Flashes are treated as conditionally independent given the target (the usual naive-Bayes assumption; it makes raw
    posteriors over-confident, which is exactly what the calibrated policy corrects).
    posterior(c | first r repetitions) = softmax_c( alpha * L_c(r) ),   L_c(r) = sum of per-repetition increments.
    alpha = 1 is the plain Bayesian rule; alpha fitted on calibration-only out-of-fold selections is the calibrated rule.
A prior over characters is uniform (the 36 characters of the grid).

Replay constraint: a policy may stop at any repetition r <= n_available. It can never ask for repetitions that were not
recorded. If it has not stopped by n_available, the evidence is *exhausted*: it either commits its current argmax or
abstains, depending on the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CHARS = 36


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cumulative_evidence(delta: np.ndarray) -> np.ndarray:
    """delta: (n_reps, 36) per-repetition log-evidence increments (NaN rows = not recorded) -> cumulative L, NaN kept."""
    return np.cumsum(np.nan_to_num(delta, nan=0.0), axis=0) + 0.0 * delta


def flash_increments(llr: np.ndarray, shown: np.ndarray) -> np.ndarray:
    """Per-character log-evidence increment of a block of flashes: llr (n,), shown (n, 36) bool -> (36,)."""
    return (llr[:, None] * shown).sum(0)


@dataclass
class Outcome:
    committed: bool
    char: int | None          # 0-based character index, None if abstained
    n_reps: int               # repetitions consumed
    exhausted: bool           # evidence ran out before the policy's own stop rule fired
    confidence: float         # posterior of the argmax at the stopping repetition under the policy's alpha


def _post(L: np.ndarray, alpha: float) -> np.ndarray:
    return softmax(alpha * L, axis=-1)


def fixed_reps(L: np.ndarray, n_avail: int, k: int, alpha: float = 1.0, abstain_below: float | None = None) -> Outcome:
    """Always consume k repetitions (fewer if the recording has fewer: exhausted). Optionally abstain if confidence is low."""
    r = min(k, n_avail)
    p = _post(L[r - 1], alpha)
    conf = float(p.max())
    ex = n_avail < k
    if abstain_below is not None and conf < abstain_below:
        return Outcome(False, None, r, ex, conf)
    return Outcome(True, int(p.argmax()), r, ex, conf)


def threshold_stop(L: np.ndarray, n_avail: int, tau: float, alpha: float = 1.0, abstain: bool = False,
                   min_reps: int = 1, max_reps: int | None = None) -> Outcome:
    """Stop at the first repetition (>= min_reps) whose posterior max reaches tau.

    If none does by min(n_avail, max_reps): without abstention commit the argmax (the standard dynamic-stopping rule);
    with abstention return no character. `exhausted` is True when the recording (not a cap) ended the selection.
    """
    last = n_avail if max_reps is None else min(n_avail, max_reps)
    for r in range(min_reps, last + 1):
        p = _post(L[r - 1], alpha)
        if p.max() >= tau:
            return Outcome(True, int(p.argmax()), r, False, float(p.max()))
    p = _post(L[last - 1], alpha)
    ex = max_reps is None or n_avail < max_reps  # recording ended before any cap that would have ended it
    if abstain:
        return Outcome(False, None, last, ex, float(p.max()))
    return Outcome(True, int(p.argmax()), last, ex, float(p.max()))


def fit_alpha(Ls: list[np.ndarray], targets: list[int], grid: np.ndarray | None = None) -> float:
    """Temperature (inverse) minimising character NLL over every repetition count of calibration-only selections."""
    grid = np.exp(np.linspace(np.log(0.02), np.log(3.0), 60)) if grid is None else grid
    best, best_a = np.inf, 1.0
    for a in grid:
        nll, n = 0.0, 0
        for L, t in zip(Ls, targets):
            P = softmax(a * L, axis=-1)
            nll += -np.log(np.clip(P[:, t], 1e-12, None)).sum()
            n += len(L)
        if nll / n < best:
            best, best_a = nll / n, float(a)
    return best_a


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)
