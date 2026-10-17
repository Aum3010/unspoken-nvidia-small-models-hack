"""Reproduce every number: python run_eval.py --config configs/eval.yaml

Phases (all by default, or --phases a,b):
  manifest  scripts/build_manifest.py            -> data/manifest.csv
  vad       Silero frame probabilities            -> results/raw/vad.csv
  models    each enabled model in its own process -> results/raw/{model}_outputs.jsonl, {model}_info.json
  decide    rank, tune on calib, apply            -> results/raw/{model}_{method}.csv, results/thresholds.json
  report    metrics, plots, failures              -> results/summary.md, results/plots/, results/failures.md

--limit N keeps N clips per (source, label_type, split, degraded) group, for quick end-to-end checks.
A model that fails is logged (results/raw/{model}_info.json "error") and the rest continue.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness.config import load_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest(cfg, limit: int | None) -> pd.DataFrame:
    m = pd.read_csv(cfg.data / "manifest.csv", keep_default_na=False)
    m["degraded"] = m["degraded"].astype(str).str.lower() == "true"
    if limit:
        m = (m.sample(frac=1, random_state=cfg["seed"]).groupby(["source", "label_type", "split", "degraded"], sort=False)
             .head(limit).sort_index())
    return m.reset_index(drop=True)


def vocabs(cfg) -> dict[str, dict[str, str]]:
    v = {"phrases20": cfg.phrases}
    tv = cfg.data / "torgo_vocab.json"
    if tv.exists():
        v["torgo_words"] = {w: w for w in json.loads(tv.read_text())["vocab"]}
    return v


def expand_tasks(m: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    """Silence / non-speech clips are negatives for every task; everything else for its own task."""
    agnostic = m["label_type"].isin(["silence", "non_speech"])
    parts = [m[~agnostic].assign(eval_task=m.loc[~agnostic, "task"])]
    parts += [m[agnostic].assign(eval_task=t) for t in tasks]
    out = pd.concat(parts, ignore_index=True)
    return out[out["eval_task"].isin(tasks)]


# ---- phases ------------------------------------------------------------------------------------------

def phase_manifest(cfg, args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_manifest.py"), "--config", str(cfg.path)], check=True)


def phase_vad(cfg, args):
    from harness import audio
    from harness.vad import Vad, stats

    m = load_manifest(cfg, args.limit)
    v = Vad()
    rows = []
    for i, p in enumerate(m["path"].unique()):
        x = audio.load(cfg.data / p)
        t = time.perf_counter()
        pr = v.probs(x)
        ms = (time.perf_counter() - t) * 1000
        rows.append({"path": p, **stats(pr, cfg["vad"]["thresholds"], cfg["vad"]["min_speech_ms"]), "vad_ms": round(ms, 2)})
        if i % 500 == 0:
            log(f"vad {i}/{m['path'].nunique()}")
    out = cfg.results / "raw" / "vad.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"wrote {out}")


def model_worker(cfg, args):
    """Runs inside a subprocess: one model, every clip, then exit (frees the GPU completely)."""
    import torch

    from harness import audio
    from harness.decide import surface_forms
    from harness.models import make

    spec = cfg.model(args.model)
    raw = cfg.results / "raw"
    info_path, out_path = raw / f"{spec['name']}_info.json", raw / f"{spec['name']}_outputs.jsonl"
    info = {"name": spec["name"], "kind": spec["kind"], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        m = load_manifest(cfg, args.limit)
        voc = vocabs(cfg)
        clips = expand_tasks(m, list(voc))
        forms = {task: [(pid, f) for pid, txt in v.items() for f in surface_forms(txt)] for task, v in voc.items()}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            info["device_name"] = torch.cuda.get_device_name()
        model = make(spec, cfg)
        t = time.perf_counter()
        model.load()
        info["load_s"] = round(time.perf_counter() - t, 2)
        info.update(model.info())

        def run(x, tasks):
            sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
            sync(); t0 = time.perf_counter()
            tr = model.transcribe(x)
            sync(); asr_ms = (time.perf_counter() - t0) * 1000
            outs = []
            for task in tasks:
                sync(); t1 = time.perf_counter()
                ph = free = None
                if model.supports_scoring:
                    texts = [f for _, f in forms[task]] + [tr["text"]]
                    sc = model.score_phrases(x, texts)
                    ph = {}
                    for (pid, _), s in zip(forms[task], sc[:-1]):
                        if pid not in ph or s[0] > ph[pid][0]:
                            ph[pid] = [round(s[0], 4), s[1]]
                    free = [round(sc[-1][0], 4), sc[-1][1]]
                sync()
                outs.append((task, ph, free, (time.perf_counter() - t1) * 1000 if model.supports_scoring else None))
            return tr, asr_ms, outs

        by_path = clips.groupby("path", sort=False)["eval_task"].apply(list)
        warm = list(by_path.items())[: cfg["latency"]["warmup_runs"]]
        for p, tasks in warm:  # discarded warm-up runs
            run(audio.load(cfg.data / p), tasks)
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            errors = []
            for i, (p, tasks) in enumerate(by_path.items()):
                x = audio.load(cfg.data / p)
                try:
                    tr, asr_ms, outs = run(x, tasks)
                except Exception as exc:  # one bad clip must not stop the model; it is counted and listed
                    errors.append({"path": p, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                    tr, asr_ms, outs = {"text": "", "error": errors[-1]["error"]}, None, [(t, None, None, None) for t in tasks]
                for task, ph, free, score_ms in outs:
                    f.write(json.dumps({"path": p, "eval_task": task, "transcript": tr["text"], "nbest": tr.get("nbest"),
                                        "signals": {k: v for k, v in tr.items() if k not in ("text", "nbest")},
                                        "phrase_logp": ph, "free_logp": free, "asr_ms": round(asr_ms, 2) if asr_ms is not None else None,
                                        "score_ms": round(score_ms, 2) if score_ms is not None else None}) + "\n")
                    n += 1
                if i % 200 == 0:
                    log(f"{spec['name']} {i}/{len(by_path)}")
        info["n_rows"] = n
        info["clip_errors"] = len(errors)
        info["clip_error_examples"] = errors[:10]
        if torch.cuda.is_available():
            info["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            free_b, total_b = torch.cuda.mem_get_info()
            info["device_used_gb_at_end"] = round((total_b - free_b) / 1e9, 2)
        else:
            import psutil

            info["peak_rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
        model.unload()
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["traceback"] = traceback.format_exc()[-4000:]
        log(f"{spec['name']} FAILED: {info['error']}")
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info_path.write_text(json.dumps(info, indent=2))
    return 1 if "error" in info else 0


def phase_models(cfg, args):
    names = [s["name"] for s in cfg.enabled_models()] if not args.models else args.models.split(",")
    for name in names:
        log(f"model {name}")
        cmd = [sys.executable, str(ROOT / "run_eval.py"), "--config", str(cfg.path), "--worker", name]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(ROOT)}).returncode
        log(f"model {name}: {'ok' if rc == 0 else 'FAILED (see its _info.json)'}")


def joined(cfg, args, name: str) -> pd.DataFrame | None:
    raw = cfg.results / "raw"
    out_path = raw / f"{name}_outputs.jsonl"
    if not out_path.exists():
        return None
    outs = pd.read_json(out_path, lines=True)
    m = expand_tasks(load_manifest(cfg, args.limit), list(vocabs(cfg)))
    vad = pd.read_csv(raw / "vad.csv")
    d = m.merge(vad, on="path", how="left").merge(outs, on=["path", "eval_task"], how="inner")
    d["transcript"] = d["transcript"].fillna("")
    d["total_ms"] = d["vad_ms"] + d["asr_ms"] + d["score_ms"].fillna(0)
    return d


def phase_decide(cfg, args):
    from harness.decide import decide_arrays, features, tune

    dc = cfg["decision"]
    voc = vocabs(cfg)
    thresholds = {}
    raw = cfg.results / "raw"
    for spec in cfg["models"]:
        d = joined(cfg, args, spec["name"])
        if d is None:
            continue
        scoring = d["phrase_logp"].notna().any()
        for method in dc["methods"]:
            if method != "text" and not scoring:
                continue
            f = features(d, voc, method, set(dc["negation_words"]))
            parts = []
            for task, ft in f.groupby("eval_task"):
                cal = ft[(ft["split"] == "calib") & ft["origin"].isin(dc["tune_origins"])]
                if not dc["tune_include_degraded"]:
                    cal = cal[~cal["degraded"]]
                th = tune(cal, cfg["vad"]["thresholds"], dc["far_target"], dc["far_constraint"],
                          dc["abs_grid_quantiles"], dc["margin_grid_quantiles"])
                if th.get("reason") == "no calib positives" and "tts" not in dc["tune_origins"]:
                    # no human calib speakers for this task: fall back to TTS positives, real negatives, and say so
                    cal2 = ft[(ft["split"] == "calib") & (ft["origin"].isin(dc["tune_origins"]) |
                                                          ((ft["origin"] == "tts") & (ft["label_type"] == "positive")))]
                    th = tune(cal2, cfg["vad"]["thresholds"], dc["far_target"], dc["far_constraint"],
                              dc["abs_grid_quantiles"], dc["margin_grid_quantiles"])
                    th["fallback"] = "calib positives were TTS (no human calib speakers)"
                thresholds[f"{spec['name']}/{method}|{task}"] = th
                ft = ft.assign(decision=decide_arrays(ft, th["vad_t"], th["abs_t"], th["margin_t"]))
                parts.append(ft)
            out = pd.concat(parts, ignore_index=True)
            cols = [c for c in out.columns if c not in ("nbest", "signals", "phrase_logp")]
            out[cols].to_csv(raw / f"{spec['name']}_{method}.csv", index=False)
            log(f"decided {spec['name']}/{method}")
    (cfg.results / "thresholds.json").write_text(json.dumps(thresholds, indent=2, default=float))


def phase_report(cfg, args):
    from harness import report

    raw = cfg.results / "raw"
    decisions, infos = {}, {}
    for spec in cfg["models"]:
        ip = raw / f"{spec['name']}_info.json"
        if ip.exists():
            infos[spec["name"]] = json.loads(ip.read_text())
        for method in cfg["decision"]["methods"]:
            p = raw / f"{spec['name']}_{method}.csv"
            if p.exists():
                d = pd.read_csv(p, keep_default_na=False, na_values=[""])
                d["degraded"] = d["degraded"].astype(str).str.lower() == "true"
                d["ranked"] = d["ranked"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
                d["transcript"] = d["transcript"].fillna("")
                d["neg_mismatch"] = d["neg_mismatch"].astype(str).str.lower() == "true"
                decisions[f"{spec['name']}/{method}"] = d
    th = json.loads((cfg.results / "thresholds.json").read_text())  # +-Infinity round-trips as floats
    report.write(cfg, decisions, infos, th)
    log(f"wrote {cfg.results / 'summary.md'}")


PHASES = {"manifest": phase_manifest, "vad": phase_vad, "models": phase_models, "decide": phase_decide, "report": phase_report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--models", help="comma list; default = enabled in config")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.worker:
        args.model = args.worker
        sys.exit(model_worker(cfg, args))
    cfg.results.mkdir(parents=True, exist_ok=True)
    import yaml

    (cfg.results / "config_used.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False), encoding="utf-8")  # merged, as run
    for ph in args.phases.split(","):
        log(f"== phase {ph}")
        PHASES[ph](cfg, args)


if __name__ == "__main__":
    main()
