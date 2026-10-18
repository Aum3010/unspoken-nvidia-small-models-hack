"""Does personal enrollment help lip reading? Measured on MIRACL-VC1 (research use only).

MIRACL-VC1: 15 speakers each say 10 words and 10 phrases, 10 times each. For every
speaker, k repetitions of each class are the "enrollment" and the rest are tests.
Compared on the same test clips:
  phrase-list   score every class text against the video (no enrollment, services/vsr/phrases.py)
  enroll-dtw    nearest enrolled clip of the same person, DTW over Auto-AVSR encoder features
  combined      both scores, each z-normalised over the class list, summed

  PYTHONPATH=$PWD .venv/bin/python scripts/eval_enrollment_miracl.py extract
  PYTHONPATH=$PWD .venv/bin/python scripts/eval_enrollment_miracl.py eval
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from services.vsr.app import _load
from services.vsr.phrases import dtw, score_phrases, tokenize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "miracl"
CACHE = ROOT / "data" / "cache" / "miracl"
OUT = ROOT / "evaluations" / "results" / "vsr_enrollment_miracl.json"
SRC_FPS, MODEL_FPS = 15.0, 25.0  # Kinect colour stream; Auto-AVSR was trained on 25 fps

WORDS = ["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "web"]
PHRASES = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye",
           "i love this game", "nice to meet you", "you are welcome", "how are you", "have a good time"]


def classes(kind: str) -> list[tuple[str, str, str]]:
    """(folder kind, folder id, text) for each class."""
    ws = [("words", f"{i + 1:02d}", t) for i, t in enumerate(WORDS)]
    ps = [("phrases", f"{i + 1:02d}", t) for i, t in enumerate(PHRASES)]
    return ps if kind == "phrases" else ws + ps


def speakers() -> list[str]:
    return sorted(p.name for p in DATA.iterdir() if p.is_dir())


# ---------------------------------------------------------------- extract
def extract():
    m = _load()
    avsr, det = m["avsr"], m["detector"]
    t0, done, failed = time.time(), 0, 0
    for spk in speakers():
        for kind, cid, _ in classes("all"):
            for rep in sorted((DATA / spk / kind / cid).iterdir()):
                dst = CACHE / spk / kind / cid / f"{rep.name}.npy"
                if dst.exists():
                    continue
                files = sorted(rep.glob("color_*.jpg"))
                frames = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files])
                idx = np.round(np.linspace(0, len(frames) - 1, max(2, round(len(frames) * MODEL_FPS / SRC_FPS)))).astype(int)
                frames = frames[idx]
                try:
                    lms = det.detect(frames, det.full_range_detector)
                    if all(l is None for l in lms):
                        lms = det.detect(frames, det.short_range_detector)
                    crops = m["process"](frames, lms)
                    x = m["transform"](torch.tensor(crops))
                    with torch.no_grad():
                        enc = avsr.model.encode(x.to(avsr.device))
                except Exception as exc:  # no face found in the clip
                    failed += 1
                    print(f"  skip {spk}/{kind}/{cid}/{rep.name}: {type(exc).__name__}", flush=True)
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.save(dst, enc.cpu().numpy().astype(np.float16))
                done += 1
        print(f"{spk}: total {done} extracted, {failed} failed, {(time.time() - t0) / max(1, done):.2f}s/clip", flush=True)


# ---------------------------------------------------------------- eval
def load_feats(spk: str, kind: str, cid: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(raw encoder features, L2-normalised frames) for every repetition."""
    out = []
    for f in sorted((CACHE / spk / kind / cid).glob("*.npy")):
        x = np.load(f).astype(np.float32)
        out.append((x, x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)))
    return out


def zscore(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)


def evaluate(ks: list[int], seeds: int):
    m = _load()
    avsr = m["avsr"]
    piece_to_id = {p: i for i, p in enumerate(avsr.token_list)}
    results = {}
    for kind in ("phrases", "all"):
        cls = classes(kind)
        toks = [tokenize(t, piece_to_id) for _, _, t in cls]
        n_cls = len(cls)
        for k in ks:
            hits = {name: [0, 0] for name in ("phrase-list", "enroll-dtw", "combined")}
            n_test = 0
            per_speaker = {}
            for spk in speakers():
                feats = [load_feats(spk, kd, cid) for kd, cid, _ in cls]
                sp_hits = {name: 0 for name in hits}
                sp_n = 0
                for seed in range(seeds):
                    rng = random.Random(seed)
                    split = [rng.sample(range(len(f)), len(f)) for f in feats]
                    enroll = [[feats[c][i][1] for i in split[c][:k]] for c in range(n_cls)]
                    for c in range(n_cls):
                        for i in split[c][k:]:
                            raw, x = feats[c][i]
                            with torch.no_grad():
                                pl = score_phrases(avsr.model, torch.from_numpy(raw).to(avsr.device), toks).numpy()
                            en = np.array([-min(dtw(x, e) for e in enroll[j]) for j in range(n_cls)])
                            scores = {"phrase-list": pl, "enroll-dtw": en, "combined": zscore(pl) + zscore(en)}
                            for name, s in scores.items():
                                rank = int((s > s[c]).sum())
                                hits[name][0] += rank == 0
                                hits[name][1] += rank < 3
                                sp_hits[name] += rank == 0
                            n_test += 1
                            sp_n += 1
                per_speaker[spk] = {name: round(v / sp_n, 3) for name, v in sp_hits.items()}
                print(f"  {kind:<7} k={k} {spk}: " + "  ".join(f"{n} {v:.0%}" for n, v in per_speaker[spk].items()), flush=True)
            key = f"{kind} ({n_cls} classes), enroll k={k}"
            results[key] = {"n_test": n_test, "chance": round(1 / n_cls, 3),
                            **{name: {"top1": round(h[0] / n_test, 4), "top3": round(h[1] / n_test, 4)} for name, h in hits.items()},
                            "per_speaker_top1": per_speaker}
            print(f"{key:<40} " + "  ".join(f"{n}: top-1 {h[0] / n_test:.1%} top-3 {h[1] / n_test:.1%}" for n, h in hits.items()), flush=True)
    OUT.write_text(json.dumps({"dataset": "MIRACL-VC1 (research use only)", "speakers": speakers(), "seeds": seeds,
                               "protocol": "per speaker: k repetitions per class enrolled, remaining repetitions tested",
                               "results": results}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("extract")
    e = sub.add_parser("eval"); e.add_argument("--ks", nargs="+", type=int, default=[1, 3]); e.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    extract() if a.stage == "extract" else evaluate(a.ks, a.seeds)
