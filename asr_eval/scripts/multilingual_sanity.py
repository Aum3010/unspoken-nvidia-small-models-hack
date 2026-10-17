"""Sanity check for the multilingual voice agent: does it recognise the translated phrases at all?

    python scripts/multilingual_sanity.py --config configs/eval.yaml --model nemotron-3.5-asr-0.6b

Kokoro speaks each language's phrases (first alternative) with every voice it has for that language;
each clip goes through the same pipeline as the live bench (language prompt, forced scoring, decide())
with the English-tuned thresholds. Synthetic speech: a plumbing/feasibility check, never a headline number.
Kokoro has no German voice, so German is reported as untested. Writes results/multilingual_sanity.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import audio  # noqa: E402
from harness.config import load_config  # noqa: E402
from harness.decide import alternatives, decide_arrays, features  # noqa: E402
from harness.models import make  # noqa: E402
from harness.pipeline import analyze, task_forms  # noqa: E402
from harness.vad import Vad, stats  # noqa: E402

KOKORO_LANG = {"es": "e", "fr": "f", "hi": "h", "zh": "z", "en": "a"}  # no German voice in Kokoro


def kokoro_voices(code: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(m.group(1) for f in list_repo_files("hexgrad/Kokoro-82M") if (m := re.fullmatch(rf"voices/({code}[fm]_\w+)\.pt", f)))


def synth(cfg, lang: str, out: Path) -> list[tuple[str, str, Path]]:
    from kokoro import KPipeline
    from scipy.signal import resample_poly

    code = KOKORO_LANG[lang]
    phrases = cfg.phrases if lang == "en" else cfg["languages"][lang]["phrases"]
    clips, pipe = [], None
    for voice in kokoro_voices(code):
        for pid, txt in phrases.items():
            dst = out / lang / voice / f"{pid}.wav"
            if not dst.exists():
                pipe = pipe or KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M")
                chunks = [np.asarray(r.audio).reshape(-1) for r in pipe(alternatives(txt)[0], voice=voice, speed=1.0)]
                y = resample_poly(np.concatenate(chunks), 2, 3).astype(np.float32)
                audio.save(dst, np.concatenate([np.zeros(4000, np.float32), y, np.zeros(4000, np.float32)]))
            clips.append((voice, pid, dst))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--model", default="nemotron-3.5-asr-0.6b")
    ap.add_argument("--method", default="forced_llr")
    a = ap.parse_args()
    cfg = load_config(a.config)
    th_all = json.loads((cfg.results / "thresholds.json").read_text())
    langs = {k: v for k, v in cfg["languages"].items() if k != "default"}
    m = make(cfg.model(a.model), cfg)
    m.load()
    vad = Vad()
    rows = []
    for lang, prof in langs.items():
        if lang not in KOKORO_LANG:
            rows.append({"lang": lang, "note": "no Kokoro voice: untested"})
            continue
        try:
            clips = synth(cfg, lang, cfg.data / "tts_ml")
        except Exception as exc:
            rows.append({"lang": lang, "note": f"TTS failed: {type(exc).__name__}: {str(exc)[:120]}"})
            continue
        m.set_language(prof["locale"])
        vocab = {"phrases20": cfg.phrases if lang == "en" else prof["phrases"]}
        forms = task_forms(vocab, lang)
        neg = set(prof.get("negation_words") or cfg["decision"]["negation_words"])
        recs = []
        for voice, pid, path in clips:
            x = audio.load(path)
            tr, asr_ms, outs = analyze(m, x, ["phrases20"], forms)
            _, ph, free, sc_ms = outs[0]
            recs.append({"path": str(path), "eval_task": "phrases20", "transcript": tr["text"], "phrase_logp": ph, "free_logp": free,
                         "phrase_id": pid, "label_type": "positive", "voice": voice, "ms": asr_ms + (sc_ms or 0),
                         **stats(vad.probs(x), cfg["vad"]["thresholds"], cfg["vad"]["min_speech_ms"])})
        f = features(pd.DataFrame(recs), vocab, a.method, neg, lang)
        for point in ("", "|balanced"):
            th = th_all.get(f"{a.model}/{a.method}|phrases20{point}")
            if not th:
                continue
            d = decide_arrays(f, th["vad_t"], th["abs_t"], th["margin_t"])
            top1 = int(sum(r[0] == p if r else False for r, p in zip(f["ranked"], f["phrase_id"])))
            ok = int((d == f["phrase_id"].to_numpy()).sum())
            wrong = int(((d != "unknown") & (d != "no_speech") & (d != f["phrase_id"].to_numpy())).sum())
            rows.append({"lang": lang, "point": point.strip("|") or "strict", "n": len(f), "voices": f["voice"].nunique(),
                         "top1": top1, "accepted_correct": ok, "wrong": wrong, "latency_ms_p50": round(float(np.median(f["ms"])), 1),
                         "examples": [(p, t) for p, t in zip(f["phrase_id"], f["transcript"])][:3]})
        print(f"{lang}: done", flush=True)
    md = ["# Multilingual sanity check (synthetic Kokoro speech, not a headline number)", "",
          f"Model `{a.model}`, method `{a.method}`, thresholds tuned on English recordings. Translations are drafts.", "",
          "| Language | Sensitivity | Clips (voices) | Correct ranked 1st | Accepted correct | Wrong accepts | Model+scoring p50 | Example transcripts |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "note" in r:
            md.append(f"| {r['lang']} | — | — | — | — | — | — | {r['note']} |")
            continue
        ex = "; ".join(f"{p}: {t}" for p, t in r["examples"])
        md.append(f"| {r['lang']} | {r['point']} | {r['n']} ({r['voices']}) | {r['top1']}/{r['n']} | {r['accepted_correct']}/{r['n']} | "
                  f"{r['wrong']}/{r['n']} | {r['latency_ms_p50']} ms | {ex} |")
    (cfg.results / "multilingual_sanity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {cfg.results / 'multilingual_sanity.md'}")


if __name__ == "__main__":
    main()
