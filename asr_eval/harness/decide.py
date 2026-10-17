"""Phase 4: phrase ranking, the candidate / unknown / no_speech decision, and calib-only tuning.

Rankers (per clip, per task vocabulary):
  text      Method A, every model: normalised transcript vs each phrase, CMU phonetic similarity
            (Levenshtein on phoneme sequences), char ratio when a word is out of the CMU dictionary
  forced_sum   Method B: forced log P(phrase | audio) from the model's own loss (CTC, or RNNT/TDT for transducers)
  forced_norm  Method B: the same, divided by the phrase's token count
  forced_llr   (extra) per-token log-likelihood ratio of each phrase against the model's own free
            transcript, scored the same way: ~0 when the phrase is what the model heard anyway

decide(): no_speech if the VAD gate fails; unknown if best < abs, margin < margin, the transcript's
negation disagrees with the best phrase's, or the model's own transcript is empty (it heard no words, so
forced scoring must not create one: plan.md "never infer words absent from all usable input");
otherwise candidate = best phrase.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from harness.text import has_negation, normalize

NEG_TYPES = ["hard_negative", "unknown_speech", "non_speech", "silence"]


@lru_cache(maxsize=None)
def phones(norm_text: str) -> tuple[str, ...] | None:
    import pronouncing

    out = []
    for w in norm_text.split():
        p = pronouncing.phones_for_word(w)
        if not p:
            return None
        out.extend(re.sub(r"\d", "", ph) for ph in p[0].split())
    return tuple(out)


def similarity(a_norm: str, b_norm: str) -> float:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    if not a_norm or not b_norm:
        return 0.0
    pa, pb = phones(a_norm), phones(b_norm)
    if pa is not None and pb is not None:
        return float(Levenshtein.normalized_similarity(pa, pb))
    return fuzz.ratio(a_norm, b_norm) / 100.0


def surface_forms(text: str) -> list[str]:
    """Surface forms scored for one phrase (forced scoring); the phrase keeps its best form."""
    written = " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).split())
    cased = " ".join(re.sub(r"[^A-Za-z0-9' ]+", " ", text.replace("’", "'")).split())
    cased = cased[:1].upper() + cased[1:]
    # lowercase forms for lowercase vocabularies (Parakeet-CTC); cased and full-stop forms for models trained with
    # punctuation and capitals (Parakeet-TDT, Nemotron). Each phrase keeps the best-scoring form.
    return list(dict.fromkeys([written, normalize(text), cased, cased + "."]))


def rank_scores(row: dict, vocab: dict[str, str], method: str) -> list[tuple[str, float]]:
    """Ranked (phrase_id, score) for one clip; row holds the model outputs for its task."""
    if method == "text":
        t = normalize(row["transcript"])
        scores = {pid: similarity(t, normalize(txt)) for pid, txt in vocab.items()}
    else:
        ph = row["phrase_logp"]  # {pid: [logp, n_tokens]}
        if not ph:
            return []
        if method == "forced_sum":
            scores = {pid: v[0] for pid, v in ph.items()}
        elif method == "forced_norm":
            scores = {pid: v[0] / max(1, v[1]) for pid, v in ph.items()}
        elif method == "forced_llr":
            free_lp, free_n = row["free_logp"]
            scores = {pid: (v[0] - free_lp) / max(1, v[1], free_n) for pid, v in ph.items()}
        else:
            raise ValueError(method)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def features(rows: pd.DataFrame, vocabs: dict[str, dict[str, str]], method: str, negation_words: set[str]) -> pd.DataFrame:
    """One row per clip with the ranked list and the decision features for `method`."""
    out = []
    for r in rows.to_dict("records"):
        vocab = vocabs[r["eval_task"]]
        ranked = rank_scores(r, vocab, method)
        if not ranked:
            out.append({**r, "ranked": [], "best": None, "best_score": -np.inf, "margin": 0.0, "neg_mismatch": False,
                        "empty_transcript": not normalize(r["transcript"])})
            continue
        best, best_s = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -np.inf
        t_neg = has_negation(normalize(r["transcript"]), negation_words)
        p_neg = has_negation(normalize(vocab[best]), negation_words)
        out.append({**r, "ranked": [p for p, _ in ranked[:5]], "best": best, "best_score": float(best_s),
                    "margin": float(best_s - second) if np.isfinite(second) else 1e9, "neg_mismatch": t_neg != p_neg,
                    "empty_transcript": not normalize(r["transcript"])})
    return pd.DataFrame(out)


def decide_arrays(f: pd.DataFrame, vad_t: float, abs_t: float, margin_t: float) -> np.ndarray:
    return _decide(_arrays(f), vad_t, abs_t, margin_t)


def _arrays(f: pd.DataFrame) -> dict:
    a = {"best": f["best"].fillna("").to_numpy(dtype=object), "score": f["best_score"].to_numpy(dtype=float),
         "margin": f["margin"].to_numpy(dtype=float), "neg": f["neg_mismatch"].to_numpy(dtype=bool),
         "has": f["best"].notna().to_numpy() & ~f["empty_transcript"].to_numpy(dtype=bool)}
    for c in f.columns:
        if c.startswith("vad_speech_s@"):
            a[c] = f[c].to_numpy() > 0
    return a


def _decide(a: dict, vad_t: float, abs_t: float, margin_t: float) -> np.ndarray:
    ok = (a["score"] >= abs_t) & (a["margin"] >= margin_t) & ~a["neg"] & a["has"]
    return np.where(~a[f"vad_speech_s@{vad_t}"], "no_speech", np.where(ok, a["best"], "unknown"))


def explain(row: dict, th: dict) -> str:
    """Which gate decided one clip: no_speech, low_score, small_margin, negation, no_hypothesis, or accepted."""
    if not row.get(f"vad_speech_s@{th['vad_t']}", 0) > 0:
        return "no_speech"
    if row.get("best") is None:
        return "no_hypothesis"
    if row.get("empty_transcript"):
        return "empty_transcript"
    if row["best_score"] < th["abs_t"]:
        return "low_score"
    if row["margin"] < th["margin_t"]:
        return "small_margin"
    if row["neg_mismatch"]:
        return "negation"
    return "accepted"


def grid(values: np.ndarray, n: int, floor: float) -> list[float]:
    v = values[np.isfinite(values)]
    qs = np.unique(np.quantile(v, np.linspace(0, 1, n))) if len(v) else np.array([])
    return [floor] + [float(x) for x in qs]


def tune(f: pd.DataFrame, vad_thresholds: list[float], far_target: float, constraint: str, nq_abs: int, nq_margin: int) -> dict:
    """Maximise accepted top-1 accuracy (correct accepts / positives) with false-accept rate <= target.

    constraint='max_per_type' applies the target to every negative type separately.
    Ties: lower worst-case FAR, then the most permissive thresholds (accept more while staying within the limit).
    """
    pos = (f["label_type"] == "positive").to_numpy()
    truth = f["phrase_id"].fillna("").to_numpy()
    neg_masks = {t: (f["label_type"] == t).to_numpy() for t in NEG_TYPES if (f["label_type"] == t).any()}
    if not pos.any():
        return {"feasible": False, "reason": "no calib positives", "vad_t": vad_thresholds[len(vad_thresholds) // 2],
                "abs_t": np.inf, "margin_t": np.inf}
    abs_grid = grid(f["best_score"].to_numpy(), nq_abs, -np.inf)
    margin_grid = grid(f["margin"].clip(upper=1e6).to_numpy(), nq_margin, 0.0)
    arrays = _arrays(f)
    best = None
    for vt in vad_thresholds:
        for at in abs_grid:
            for mt in margin_grid:
                d = _decide(arrays, vt, at, mt)
                acc = d != "unknown"
                acc &= d != "no_speech"
                obj = float((acc & pos & (d == truth)).sum() / pos.sum())
                fars = {t: float((acc & m).sum() / m.sum()) for t, m in neg_masks.items()}
                pooled = float(sum((acc & m).sum() for m in neg_masks.values()) / max(1, sum(m.sum() for m in neg_masks.values())))
                worst = max(fars.values()) if fars else 0.0
                far = worst if constraint == "max_per_type" else pooled
                feasible = far <= far_target
                # ties (common on easy calib data, e.g. clean TTS) go to the MOST permissive thresholds that still
                # meet the false-accept limit: the limit is the safety constraint, so inside it we accept as much as possible
                key = (feasible, obj if feasible else -far, -worst, -at, -mt, -vt)
                if best is None or key > best[0]:
                    best = (key, {"feasible": feasible, "vad_t": vt, "abs_t": at, "margin_t": mt, "calib_objective": obj,
                                  "calib_far": fars, "calib_far_pooled": pooled, "n_calib_pos": int(pos.sum()),
                                  "n_calib_neg": {t: int(m.sum()) for t, m in neg_masks.items()}})
    return best[1]
