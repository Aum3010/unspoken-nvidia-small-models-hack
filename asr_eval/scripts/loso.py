"""Leave-one-speaker-out over your own recordings: every person is tested by thresholds tuned without them.

    python scripts/loso.py --config configs/eval.yaml          # after run_eval.py has produced model outputs

For each model/method and each own speaker S:
  tune on  : all other own speakers' clips + the calib split of the real negative sources (LibriSpeech, TORGO
             sentences, noise, silence), exactly like run_eval's tuning
  test on  : every clip of S (phrases, hard negatives, other negatives)
Counts are summed over the folds, so each clip is tested exactly once. Writes results/loso.md and loso.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.config import load_config  # noqa: E402
from harness.decide import NEG_TYPES, decide_arrays, features, tune  # noqa: E402
from harness.metrics import fmt, negative_metrics, positive_metrics, rate  # noqa: E402
from run_eval import joined, vocabs  # noqa: E402


def fold_rows(f: pd.DataFrame, dc: dict, cfg, far_target: float) -> pd.DataFrame:
    own = f[f["source"] == "own"]
    out = []
    for s in sorted(own["speaker_id"].unique()):
        cal = pd.concat([f[(f["source"] != "own") & (f["split"] == "calib") & f["origin"].isin(dc["tune_origins"])],
                         own[own["speaker_id"] != s]])
        test = own[own["speaker_id"] == s]
        th = tune(cal, cfg["vad"]["thresholds"], far_target, dc["far_constraint"], dc["abs_grid_quantiles"], dc["margin_grid_quantiles"])
        out.append(test.assign(decision=decide_arrays(test, th["vad_t"], th["abs_t"], th["margin_t"]), fold=s))
    return pd.concat(out, ignore_index=True)


def system_view(t: pd.DataFrame) -> dict:
    """How the clip plays out in the full app (gateway + coordinator + gaze confirmation).

    candidate -> one candidate to confirm; unknown (speech present) -> the coordinator offers the top 3 for a gaze
    choice; no_speech -> retake / switch modality. A positive is resolved within one gaze question if the candidate
    was right, or it was unknown and the truth was in the top 3. On negatives, an unknown costs one extra question
    (answered "none of these"); a candidate there is a false accept.
    """
    pos = t[t["label_type"] == "positive"]
    neg = t[t["label_type"] != "positive"]
    right = (pos["decision"] == pos["phrase_id"])
    offered = (pos["decision"] == "unknown") & [p in (r or [])[:3] for p, r in zip(pos["phrase_id"], pos["ranked"])]
    return {"one_question": rate(int((right | offered).sum()), len(pos)),
            "extra_question_on_negatives": rate(int((neg["decision"] == "unknown").sum()), len(neg))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    a = ap.parse_args()
    a.limit = None
    cfg = load_config(a.config)
    dc = cfg["decision"]
    points = {"strict": dc["far_target"], **(dc.get("operating_points") or {})}
    voc = vocabs(cfg)
    res, per = {}, {}
    for spec in cfg.enabled_models():
        d = joined(cfg, a, spec["name"])
        if d is None or not (d["source"] == "own").any():
            continue
        d = d[d["eval_task"] == "phrases20"]
        scoring = d["phrase_logp"].notna().any()
        for method in dc["methods"]:
            if method != "text" and not scoring:
                continue
            f = features(d, voc, method, set(dc["negation_words"]))
            for pname, far in points.items():
                t = fold_rows(f, dc, cfg, far)
                key = f"{spec['name']}/{method}"
                real, deg = t[~t["degraded"]], t[t["degraded"]]  # degraded copies are synthetic: never mixed into the headline
                res.setdefault(key, {})[pname] = {"pos": positive_metrics(real), "neg": negative_metrics(real),
                                                  "degraded_pos": positive_metrics(deg), "system": system_view(real)}
                if pname == "balanced" or len(points) == 1:
                    per[key] = t
            print(f"loso {spec['name']}/{method}", flush=True)

    def score(k, p="strict"):
        return res[k][p]["pos"]["accepted_correct"]["rate"]

    order = sorted(res, key=lambda k: -score(k))
    md = ["# Leave-one-speaker-out on own recordings", "",
          "Each person is tested by thresholds tuned on the other people (plus real negatives from the calib split). "
          "Counts are k/n over all folds; each clip is tested once. Main tables: real recordings only (normal, whisper, "
          "noisy, muffled as spoken). Synthetic degraded copies of the same recordings are reported separately below.", "",
          "**Resolved ≤1 gaze question**: the candidate was right, or the agent said unknown and the coordinator's top-3 "
          "question contained the right phrase (README: the coordinator asks the smallest useful question). "
          "**Extra question on negatives**: negatives that trigger a top-3 question the person answers with \"none of these\".", ""]
    for pname, far in points.items():
        md += [f"## {pname} (false accepts ≤ {100 * far:.0f}% per negative type on calib): real recordings", "",
               "| Model / method | Top-1 rank | Top-3 rank | Accepted correct | Wrong accept | Abstain | **Resolved ≤1 gaze question** | FA hard negatives | FA silence / noise / cough | FA unrelated speech | Extra question on negatives |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(res, key=lambda k: -res[k][pname]["pos"]["accepted_correct"]["rate"]):
            p, n = res[k][pname]["pos"], res[k][pname]["neg"]
            sil = rate(n["silence"]["k"] + n["non_speech"]["k"], n["silence"]["n"] + n["non_speech"]["n"])
            sv = res[k][pname]["system"]
            md.append(f"| {k} | {fmt(p['top1'], False)} | {fmt(p['top3'], False)} | **{fmt(p['accepted_correct'], False)}** | "
                      f"{fmt(p['wrong_accept'], False)} | {fmt(p['abstain'], False)} | **{fmt(sv['one_question'], False)}** | "
                      f"{fmt(n['hard_negative'])} | {fmt(sil)} | {fmt(n['unknown_speech'])} | {fmt(sv['extra_question_on_negatives'], False)} |")
        md.append("")
    md += ["## Synthetic degraded copies of the same recordings (not headline)", "",
           "| Model / method | strict: accepted correct | strict: wrong accept | balanced: accepted correct | balanced: wrong accept |",
           "|---|---|---|---|---|"]
    for k in order:
        r = res[k]
        b = r.get("balanced", r["strict"])
        md.append(f"| {k} | {fmt(r['strict']['degraded_pos']['accepted_correct'], False)} | {fmt(r['strict']['degraded_pos']['wrong_accept'], False)} | "
                  f"{fmt(b['degraded_pos']['accepted_correct'], False)} | {fmt(b['degraded_pos']['wrong_accept'], False)} |")
    md += ["", "## By condition and speaker (balanced point, top 4 configurations; degraded rows are synthetic)", ""]
    for k in [k for k in sorted(res, key=lambda k: -score(k, "balanced" if "balanced" in points else "strict"))][:4]:
        t = per[k]
        pos = t[t["label_type"] == "positive"]
        md += [f"### {k}", "", "| Speaker | Condition | n | Top-1 rank | Accepted correct | Wrong accept |", "|---|---|---:|---|---|---|"]
        for (spk, cond), g in pos.groupby(["speaker_id", "condition"]):
            p = positive_metrics(g)
            md.append(f"| {spk.split(':')[1]} | {cond} | {p['n']} | {fmt(p['top1'], False)} | {fmt(p['accepted_correct'], False)} | {fmt(p['wrong_accept'], False)} |")
        wrong = t[~t["decision"].isin(["unknown", "no_speech"]) & ((t["label_type"] != "positive") | (t["decision"] != t["phrase_id"]))]
        if len(wrong):
            md += ["", "Wrong or false accepts:", "", "| Clip | Truth | Decision | Transcript |", "|---|---|---|---|"]
            for r in wrong.head(15).to_dict("records"):
                truth = r["phrase_id"] or f"({r['label_type']}) {r.get('text') or ''}"
                md.append(f"| `{Path(r['path']).name}` | {truth} | {r['decision']} | {r['transcript']} |")
        md.append("")
    out = cfg.results
    (out / "loso.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "loso.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out / 'loso.md'}")


if __name__ == "__main__":
    main()
