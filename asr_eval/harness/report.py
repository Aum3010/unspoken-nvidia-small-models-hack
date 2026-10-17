"""Phase 6: results/summary.md, results/plots/, results/failures.md (test split only).

The best configuration is selected on the calib split (its tuning objective), then
reported on test, so choosing a winner never looks at test labels.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from harness.decide import NEG_TYPES
from harness.metrics import fmt, latency, negative_metrics, positive_metrics, rate, wer_cer

LITERATURE = [
    ("Open ASR Leaderboard", "https://huggingface.co/spaces/hf-audio/open_asr_leaderboard",
     "English WER on public test sets (incl. LibriSpeech test-clean) under the leaderboard's own text normaliser; open transcription, no abstention."),
    ("SAP 2025 challenge baselines", "https://arxiv.org/html/2507.22047v1",
     "Whisper sizes on Speech Accessibility Project dysarthric speech; open-vocabulary WER on the challenge splits."),
    ("Takahashi et al., Interspeech 2025", "https://www.isca-archive.org/interspeech_2025/takahashi25_interspeech.pdf",
     "Parakeet-TDT vs Whisper fine-tuned on SAP dysarthric speech; WER on SAP evaluation data."),
    ("Whisper non-speech hallucination + VAD", "https://arxiv.org/pdf/2501.11378",
     "Hallucination rates of Whisper on non-speech audio and the effect of VAD gating."),
    ("Calm-Whisper, Interspeech 2025", "https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.pdf",
     "Non-speech hallucination rates and a mitigation by modifying attention heads."),
    ("Cross-model hallucination, Findings ACL 2025", "https://aclanthology.org/2025.findings-acl.1190.pdf",
     "Hallucination behaviour across Whisper and transducer (RNN-T/TDT) models."),
    ("de Groot et al., Interspeech 2025", "https://www.isca-archive.org/interspeech_2025/degroot25_interspeech.pdf",
     "Speech enhancement degrading dysarthric ASR on TORGO/UASpeech."),
    ("Mawalim et al., Interspeech 2024", "https://www.isca-archive.org/interspeech_2024/mawalim24_interspeech.pdf",
     "DeepFilterNet3 enhancement evaluated on downstream ASR."),
    ("Latent Phrase Matching (CMU 2023)", "https://www.cs.cmu.edu/~jbigham/pubs/pdfs/2023/lpm-dysarthric.pdf",
     "Few-shot, per-user template matching of dysarthric phrases (closed set, enrollment)."),
    ("Personalized vs normative dysarthric ASR, EMNLP 2025", "https://aclanthology.org/2025.emnlp-main.1701.pdf",
     "Speaker-dependent vs speaker-independent models on TORGO."),
    ("Cross-etiology transfer SAP to TORGO", "https://arxiv.org/pdf/2501.14994v1",
     "Training on SAP and testing on TORGO speakers (cross-etiology generalisation)."),
]


def _pct(r: dict) -> str:
    return fmt(r, ci=False)


def select_best(thresholds: dict, task: str = "phrases20") -> str | None:
    cands = [(k, v) for k, v in thresholds.items() if k.endswith(f"|{task}") and v.get("feasible")]
    if not cands:
        return None
    k, _ = max(cands, key=lambda kv: (kv[1]["calib_objective"], -max(kv[1]["calib_far"].values() or [0])))
    return k.rsplit("|", 1)[0]


def write(cfg, decisions: dict[str, pd.DataFrame], infos: dict[str, dict], thresholds: dict) -> None:
    res = cfg.results
    (res / "plots").mkdir(parents=True, exist_ok=True)
    tests = {k: d[d["split"] == "test"] for k, d in decisions.items()}
    best = select_best(thresholds)
    md = ["# ASR evaluation summary", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} · config `{cfg.path.name}` · **test split only** (thresholds tuned on calib).", ""]
    tts_tuned = "tts" in cfg["decision"]["tune_origins"]
    warn = []
    if tts_tuned:
        warn.append("Thresholds were tuned with TTS clips in the calib set (local smoke config). These numbers are plumbing checks, not results.")
    any_real_pos = any(((d["origin"] == "real") & (d["label_type"] == "positive") & (d["task"] == "phrases20")).any() for d in tests.values())
    if not any_real_pos:
        warn.append("No real human recordings of the 20 phrases in the test split yet: the headline positive columns are empty. "
                    "TTS rows below are sanity checks only.")
    fallback = [k for k, v in thresholds.items() if v.get("fallback") and k.count("|") == 1]
    if fallback:
        warn.append("Thresholds for " + ", ".join(fallback) + " were tuned with **TTS positives** (no human calib speakers for that task) "
                    "against real negatives. Record at least two speakers so tuning uses human speech.")
    infeasible = [k for k, v in thresholds.items() if not v.get("feasible") and k.count("|") == 1]
    if infeasible:
        warn.append(f"No threshold met the calib false-accept target for: {', '.join(infeasible)} (strictest point used; see Thresholds).")
    md += [f"> ⚠️ {w}" for w in warn] + ([""] if warn else [])

    # ---- headline -------------------------------------------------------------------------------
    md += ["## Headline: 20 phrases, real recordings, not degraded", "",
           "| Model / method | Top-1 rank | Top-3 rank | Accepted correct | Wrong accept | Abstain | FA hard-neg | FA unknown speech | FA non-speech | FA silence |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k, d in tests.items():
        g = d[(d["origin"].isin(["real", "generated"])) & (~d["degraded"]) & (d["eval_task"] == "phrases20")]
        p, n = positive_metrics(g), negative_metrics(g)
        md.append(f"| {k} | {_pct(p['top1'])} | {_pct(p['top3'])} | {_pct(p['accepted_correct'])} | {_pct(p['wrong_accept'])} | "
                  f"{_pct(p['abstain'])} | " + " | ".join(fmt(n[t]) for t in NEG_TYPES) + " |")
    md += ["", "FA = false accept (a phrase proposed on a negative clip): k/n (rate) [95% Wilson CI]; ≤x% = rule-of-three upper bound when k = 0. "
           "Ranking columns are before abstention; 'Accepted correct' is correct candidates over all positives.", ""]

    # ---- per condition ----------------------------------------------------------------------------
    focus = best or next(iter(tests), None)
    if focus:
        d = tests[focus]
        md += [f"## Per condition ({focus})", ""]
        for title, mask in [("Real recordings", (d["origin"] == "real") & ~d["degraded"]),
                            ("Degraded copies of real recordings (synthetic degradation)", (d["origin"] == "real") & d["degraded"]),
                            ("TTS — sanity only", (d["origin"] == "tts") & ~d["degraded"]),
                            ("Degraded TTS — sanity only", (d["origin"] == "tts") & d["degraded"])]:
            g = d[mask & (d["label_type"] == "positive") & (d["eval_task"] == "phrases20")]
            if g.empty:
                continue
            md += [f"### {title}", "", "| Condition | n | Top-1 rank | Accepted correct | Wrong accept | Abstain |", "|---|---:|---|---|---|---|"]
            for cond, gg in g.groupby("condition"):
                p = positive_metrics(gg)
                md.append(f"| {cond} | {p['n']} | {_pct(p['top1'])} | {_pct(p['accepted_correct'])} | {_pct(p['wrong_accept'])} | {_pct(p['abstain'])} |")
            md.append("")

    # ---- TORGO ----------------------------------------------------------------------------------------
    torgo_rows = []
    for k, d in tests.items():
        g = d[(d["source"] == "torgo") & ~d["degraded"]]
        if g.empty:
            continue
        for grp, gg in g.groupby("group"):
            words = gg[gg["eval_task"] == "torgo_words"]
            p, n = positive_metrics(words), negative_metrics(words)
            w = wer_cer(gg[gg["eval_task"] == gg["task"]])
            wtxt = "{:.1f}% / {:.1f}% (n={})".format(100 * w["wer"], 100 * w["cer"], w["n"]) if w else "—"
            torgo_rows.append(f"| {k} | {grp} | {_pct(p['top1'])} | {_pct(p['accepted_correct'])} | {_pct(p['wrong_accept'])} | "
                              f"{fmt(n['unknown_speech'])} | {wtxt} |")
    if torgo_rows:
        vocab = json.loads((cfg.data / "torgo_vocab.json").read_text()) if (cfg.data / "torgo_vocab.json").exists() else {}
        md += ["## TORGO surrogate closed-vocabulary task: dysarthric vs control", "",
               f"Vocabulary ({vocab.get('chosen_by', '')}): {', '.join(vocab.get('vocab', []))}", "",
               "| Model / method | Group | Top-1 rank | Accepted correct | Wrong accept | FA other words | WER / CER (all TORGO clips) |",
               "|---|---|---|---|---|---|---|"] + torgo_rows + [""]
        tbest = select_best(thresholds, task="torgo_words")
        if tbest and tbest in tests:
            d = tests[tbest]
            g = d[(d["source"] == "torgo") & (d["eval_task"] == "torgo_words") & (d["label_type"] == "positive")]
            md += [f"### TORGO words by condition ({tbest}, best on calib for this task; degraded rows are synthetic)", "",
                   "| Condition | Group | n | Top-1 rank | Accepted correct | Wrong accept |", "|---|---|---:|---|---|---|"]
            for (cond, grp), gg in g.groupby(["condition", "group"]):
                p = positive_metrics(gg)
                md.append(f"| {cond} | {grp} | {p['n']} | {_pct(p['top1'])} | {_pct(p['accepted_correct'])} | {_pct(p['wrong_accept'])} |")
            md.append("")

    # ---- negatives by source ------------------------------------------------------------------------------
    md += ["## Negatives by source (test)", "", "| Model / method | Source | Label | False accepts |", "|---|---|---|---|"]
    for k, d in tests.items():
        g = d[(d["label_type"] != "positive") & (d["eval_task"] == "phrases20")]
        for (src, lab), gg in g.groupby(["source", "label_type"]):
            r = rate(int((~gg["decision"].isin(["unknown", "no_speech"])).sum()), len(gg))
            md.append(f"| {k} | {src}{' (sanity)' if src == 'tts' else ''} | {lab} | {fmt(r)} |")
    md.append("")

    # ---- WER ------------------------------------------------------------------------------------------------
    md += ["## WER / CER of the free transcript", "", "| Model | Source | WER | CER | n |", "|---|---|---|---|---|"]
    for k, d in tests.items():
        if not k.endswith("/text"):
            continue
        g = d[(d["eval_task"] == d["task"]) & ~d["degraded"]]
        for src, gg in g.groupby("source"):
            w = wer_cer(gg)
            if w:
                md.append(f"| {k.split('/')[0]} | {src} | {100 * w['wer']:.1f}% | {100 * w['cer']:.1f}% | {w['n']} |")
    md += ["", "Text normalised (lowercase, contractions expanded, punctuation removed) on both sides; empty hypotheses count as deletions.", ""]

    # ---- latency + resources ------------------------------------------------------------------------------------
    md += ["## Latency (batch 1, warm) and resources", "",
           "| Model | Params | Load | Peak torch alloc | GPU used (device) | VAD p50 | ASR p50 / p95 / max | Scoring p50 / p95 | Total p50 / p95 / max | RTFx (median) |",
           "|---|---:|---:|---:|---:|---:|---|---|---|---:|"]
    lat_rows = {}
    for k, d in decisions.items():
        model = k.split("/")[0]
        if model in lat_rows:
            continue
        lat_rows[model] = latency(d.drop_duplicates(["path", "eval_task"]))
        info, L = infos.get(model, {}), lat_rows[model]

        def trio(c, keys=("p50", "p95", "max")):
            return " / ".join(f"{L[c][q]:.0f}" for q in keys) + " ms" if c in L else "—"
        params = f"{info['params'] / 1e6:.1f}M" if info.get("params") else "—"
        md.append(f"| {model} | {params} | {info.get('load_s', math.nan):.1f} s | "
                  f"{info.get('peak_alloc_gb', 'n/a')} GB | {info.get('device_used_gb_at_end', 'n/a')} GB | "
                  f"{L.get('vad_ms', {}).get('p50', math.nan):.1f} ms | {trio('asr_ms')} | "
                  f"{trio('score_ms', ('p50', 'p95'))} | {trio('total_ms')} | {L.get('rtfx_median') or math.nan:.1f} |")
    md += ["", "'Peak torch alloc' is PyTorch's allocator (0 for CTranslate2/faster-whisper); 'GPU used (device)' is the whole device "
           "at the end of the model's run, including its CUDA context.", "", "Total = VAD + ASR + phrase scoring (scoring is the forced pass over all phrases; the text method skips it). "
           "Network, capture and endpointing are not included.", ""]
    md += ["| Model | Checkpoint | Revision | Device |", "|---|---|---|---|"]
    for model, info in infos.items():
        md.append(f"| {model} | {info.get('checkpoint')} | `{str(info.get('revision', ''))[:12]}` | {info.get('device_name', info.get('device'))} |")
    md.append("")

    # ---- thresholds ------------------------------------------------------------------------------------------------
    md += ["## Thresholds (frozen from calib)", "",
           "Rows ending in a name (e.g. · balanced) are extra operating points for the live UI; every number above uses the headline point "
           f"(false accepts ≤ {100 * cfg['decision']['far_target']:.0f}% per negative type on calib).", "",
           "| Model / method · task | VAD thr | abs | margin | Calib accepted correct | Calib FA by type | Feasible |",
           "|---|---|---|---|---|---|---|"]
    for k, v in thresholds.items():
        fars = ", ".join(f"{t} {100 * f:.1f}% (n={v['n_calib_neg'][t]})" for t, f in (v.get("calib_far") or {}).items())
        md.append(f"| {k.replace('|', ' · ')} | {v['vad_t']} | {v['abs_t']:.3f} | {v['margin_t']:.3f} | "
                  f"{100 * v.get('calib_objective', 0):.1f}% (n={v.get('n_calib_pos', 0)}{', TTS' if v.get('fallback') else ''}) | {fars or v.get('reason', '')} | {'yes' if v.get('feasible') else 'NO'} |")
    md.append("")

    # ---- verdict --------------------------------------------------------------------------------------------------------
    md += ["## Which model wins and why (measured numbers only)", ""]
    if not best:
        md.append("No configuration met the calib false-accept target, so none is recommended.")
    elif not any_real_pos:
        md.append(f"Best on calib: **{best}**. Without real human recordings of the 20 phrases there is no headline result yet; "
                  "the comparison below this line is about plumbing, not accuracy.")
    else:
        d = tests[best]
        g = d[(d["origin"] == "real") & ~d["degraded"] & (d["eval_task"] == "phrases20")]
        p, n = positive_metrics(g), negative_metrics(g)
        md.append(f"**{best}** had the highest calib objective among configurations meeting the calib false-accept target. On test real recordings it "
                  f"accepted the correct phrase for {_pct(p['accepted_correct'])} of positives, wrongly accepted {_pct(p['wrong_accept'])}, and "
                  f"false-accepted " + ", ".join(f"{t.replace('_', ' ')} {fmt(n[t], ci=False)}" for t in NEG_TYPES) + ".")
    md.append("")
    md += ["## Context from literature (not our measurements)", "",
           "Protocols differ from ours (open transcription vs a 20-phrase closed set with abstention), so their numbers are not comparable "
           "to the tables above and are not copied here.", ""]
    md += [f"- [{t}]({u}) — {p}" for t, u, p in LITERATURE]
    (res / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    _failures(res, tests, best)
    _plots(res, decisions, tests, thresholds, best)


def _failures(res: Path, tests: dict, best: str | None) -> None:
    k = best or next(iter(tests), None)
    if not k:
        return
    d = tests[k]
    acc = ~d["decision"].isin(["unknown", "no_speech"])
    wrong = d[acc & (d["label_type"] == "positive") & (d["decision"] != d["phrase_id"])].assign(kind="wrong accept")
    fa = d[acc & (d["label_type"] != "positive")].assign(kind="false accept")
    errs = pd.concat([fa, wrong]).sort_values("best_score", ascending=False)
    if len(errs) < 20:
        miss = d[(d["label_type"] == "positive") & ~acc].assign(kind="missed (abstained)")
        errs = pd.concat([errs, miss.sort_values("best_score", ascending=False)])
    lines = [f"# 20 worst errors ({k}, test split)", "", "Most confident first: false accepts, then wrong accepts, then missed positives.", "",
             "| Kind | Path | Source / condition | Truth | Decision | Score | Transcript |", "|---|---|---|---|---|---|---|"]
    for r in errs.head(20).to_dict("records"):
        truth = r["phrase_id"] if r["label_type"] == "positive" else f"({r['label_type']}) {r.get('text') or ''}"
        lines.append(f"| {r['kind']} | `{r['path']}` | {r['source']} / {r['condition']} | {truth} | {r['decision']} | "
                     f"{r['best_score']:.3f} | {str(r['transcript'])[:80]} |")
    (res / "failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(res: Path, decisions: dict, tests: dict, thresholds: dict, best: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from harness.decide import _arrays, _decide

    # latency boxplot (per model)
    models, data = [], []
    for k, d in decisions.items():
        m = k.split("/")[0]
        if m not in models and d["total_ms"].notna().any():
            models.append(m)
            data.append(d.drop_duplicates(["path", "eval_task"])["total_ms"].dropna().to_numpy())
    if data:
        fig, ax = plt.subplots(figsize=(1.6 * len(data) + 3, 4))
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(models) + 1), models, rotation=20)
        ax.set_ylabel("VAD + ASR + scoring (ms)")
        ax.set_title("Per-clip latency, batch 1, warm")
        fig.tight_layout()
        fig.savefig(res / "plots" / "latency_boxplot.png", dpi=130)
        plt.close(fig)
    if not best:
        return
    d = tests[best]

    # accuracy vs condition
    g = d[(d["label_type"] == "positive") & (d["eval_task"] == "phrases20")]
    if not g.empty:
        g = g.assign(bucket=np.where(g["origin"] == "tts", "TTS: ", "real: ") + g["condition"].astype(str))
        stats = []
        for b, gg in g.groupby("bucket"):
            p = positive_metrics(gg)
            stats.append((b, p["accepted_correct"]["rate"], p["top1"]["rate"], p["wrong_accept"]["rate"], p["n"]))
        stats.sort()
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(stats) + 2), 4.5))
        x = np.arange(len(stats))
        ax.bar(x - 0.27, [s[2] for s in stats], 0.27, label="top-1 rank")
        ax.bar(x, [s[1] for s in stats], 0.27, label="accepted correct")
        ax.bar(x + 0.27, [s[3] for s in stats], 0.27, label="wrong accept")
        ax.set_xticks(x, [f"{s[0]} (n={s[4]})" for s in stats], rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"Accuracy by condition — {best} (test)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(res / "plots" / "accuracy_vs_condition.png", dpi=130)
        plt.close(fig)

    # false accept vs abs threshold on calib, at the tuned VAD/margin
    key = f"{best}|phrases20"
    th = thresholds.get(key)
    cal = decisions[best]
    cal = cal[(cal["split"] == "calib") & (cal["eval_task"] == "phrases20")]
    if th and not cal.empty:
        a = _arrays(cal)
        xs = np.unique(np.quantile(cal["best_score"].replace(-np.inf, np.nan).dropna(), np.linspace(0, 1, 60)))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        pos = (cal["label_type"] == "positive").to_numpy()
        truth = cal["phrase_id"].fillna("").to_numpy()
        curves = {t: [] for t in NEG_TYPES if (cal["label_type"] == t).any()}
        accc = []
        for x in xs:
            dec = _decide(a, th["vad_t"], x, th["margin_t"])
            acc = ~np.isin(dec, ["unknown", "no_speech"])
            for t in curves:
                m = (cal["label_type"] == t).to_numpy()
                curves[t].append((acc & m).sum() / m.sum())
            accc.append((acc & pos & (dec == truth)).sum() / max(1, pos.sum()))
        for t, ys in curves.items():
            ax.plot(xs, ys, label=f"FA {t}")
        ax.plot(xs, accc, "k--", label="accepted correct (positives)")
        ax.axvline(th["abs_t"], color="grey", lw=0.8)
        ax.set_xlabel("absolute score threshold")
        ax.set_ylabel("rate")
        ax.set_title(f"Calib: false accept vs threshold — {best}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(res / "plots" / "far_vs_threshold.png", dpi=130)
        plt.close(fig)

    # confusion matrix: positives truth x decision, plus one row per negative type
    g = d[d["eval_task"] == "phrases20"]
    if not g.empty:
        labels = sorted(g.loc[g["label_type"] == "positive", "phrase_id"].dropna().unique())
        rows = labels + [f"[{t}]" for t in NEG_TYPES if (g["label_type"] == t).any()]
        cols = labels + ["unknown", "no_speech"]
        extra = sorted(set(g["decision"]) - set(cols))
        cols = cols[:-2] + extra + ["unknown", "no_speech"]
        mat = np.zeros((len(rows), len(cols)))
        for r in g.to_dict("records"):
            ri = rows.index(r["phrase_id"]) if r["label_type"] == "positive" and r["phrase_id"] in labels else rows.index(f"[{r['label_type']}]")
            mat[ri, cols.index(r["decision"])] += 1
        norm = mat / np.maximum(1, mat.sum(1, keepdims=True))
        fig, ax = plt.subplots(figsize=(0.32 * len(cols) + 3, 0.32 * len(rows) + 2))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(rows)), rows, fontsize=6)
        ax.set_xlabel("decision")
        ax.set_ylabel("truth")
        ax.set_title(f"Confusion (row-normalised) — {best} (test)", fontsize=9)
        fig.tight_layout()
        fig.savefig(res / "plots" / "confusion_best.png", dpi=150)
        plt.close(fig)
