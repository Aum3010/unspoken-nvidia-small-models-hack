"""Phase 5: metrics with exact counts, Wilson intervals and the rule of three."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from harness.decide import NEG_TYPES
from harness.text import normalize


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    out = {"k": int(k), "n": int(n), "rate": (k / n) if n else math.nan, "ci_lo": lo, "ci_hi": hi}
    if n and k == 0:
        out["rule_of_three"] = 3 / n
    return out


def fmt(r: dict, ci: bool = True) -> str:
    if not r["n"]:
        return "—"
    s = f"{r['k']}/{r['n']} ({100 * r['rate']:.1f}%)"
    if ci:
        s += f" [{100 * r['ci_lo']:.1f}–{100 * r['ci_hi']:.1f}]"
        if "rule_of_three" in r:
            s += f" ≤{100 * r['rule_of_three']:.1f}%"
    return s


def positive_metrics(d: pd.DataFrame) -> dict:
    p = d[d["label_type"] == "positive"]
    n = len(p)
    top1 = int(sum(r[0] == t if r else False for r, t in zip(p["ranked"], p["phrase_id"])))
    top3 = int(sum(t in r[:3] if r else False for r, t in zip(p["ranked"], p["phrase_id"])))
    accepted = ~p["decision"].isin(["unknown", "no_speech"])
    correct = int((accepted & (p["decision"] == p["phrase_id"])).sum())
    wrong = int((accepted & (p["decision"] != p["phrase_id"])).sum())
    return {"n": n, "top1": rate(top1, n), "top3": rate(top3, n), "accepted_correct": rate(correct, n),
            "wrong_accept": rate(wrong, n), "abstain": rate(n - correct - wrong, n),
            "no_speech": rate(int((p["decision"] == "no_speech").sum()), n),
            "precision_when_accepting": rate(correct, correct + wrong)}


def negative_metrics(d: pd.DataFrame) -> dict:
    out = {}
    for t in NEG_TYPES:
        g = d[d["label_type"] == t]
        out[t] = rate(int((~g["decision"].isin(["unknown", "no_speech"])).sum()), len(g))
    return out


def wer_cer(d: pd.DataFrame) -> dict | None:
    import jiwer

    g = d[d["text"].fillna("").str.len() > 0]
    if g.empty:
        return None
    refs = [normalize(t) for t in g["text"]]
    hyps = [normalize(t) or "<empty>" for t in g["transcript"].fillna("")]
    keep = [i for i, r in enumerate(refs) if r]
    refs, hyps = [refs[i] for i in keep], [hyps[i] for i in keep]
    return {"n": len(refs), "wer": float(jiwer.wer(refs, hyps)), "cer": float(jiwer.cer(refs, hyps))}


def latency(d: pd.DataFrame) -> dict:
    out = {}
    for col in ("vad_ms", "asr_ms", "score_ms", "total_ms"):
        v = d[col].dropna().to_numpy(dtype=float)
        if len(v):
            out[col] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(v.max()), "n": len(v)}
    if "total_ms" in d and d["total_ms"].notna().any():
        rtfx = (d["duration_s"] * 1000 / d["total_ms"]).replace([np.inf, -np.inf], np.nan).dropna()
        out["rtfx_median"] = float(rtfx.median()) if len(rtfx) else None
    return out
