"""Lips + voice on the same utterances: does fusing the two agents beat either alone, and at what latency?

GRID test speakers s1 and s2 (never used to train either model) have real, synchronised video and audio. Each clip
gets one closed candidate list: the true sentence plus N-1 GRID sentences that differ from it in exactly one word
(the hard case: "bin blue at f two now" vs "bin blue at f TWO soon"). Both agents score the same list:

  voice  Nemotron Speech Streaming en 0.6B, forced transducer log-likelihood of each candidate (asr_eval harness)
  lips   Auto-AVSR visual-only (the deployed checkpoint, not the GRID fine-tune), teacher-forced decoder+CTC score

The audio is degraded (white noise, or babble from three clips of the other speaker) at several SNRs; the video never
is, which is the point: noise that defeats the ears does not touch the lips.

Fusion treats the two as independent evidence: log p(c) = log_softmax(voice / T_v) + log_softmax(lips / T_l),
renormalised. Temperatures and the commit threshold are fitted on one speaker and tested on the other (both folds),
at a 2% wrong-commit target; below the threshold the agent abstains. Latency: the agents run in parallel, so the
fused latency is the slower agent's per clip.

Stages (each writes evaluations/results/av_fusion/<stage>.json; lips and voice run in different venvs):
  cands  candidate lists                          any venv
  lips   lip scores + latency                     app venv (Chaplin)
  voice  voice scores per condition + latency     NeMo venv (asr_eval)
  fuse   calibrate, decide, report                any venv with numpy
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GRID = ROOT / "data" / "grid"
OUT = ROOT / "evaluations" / "results" / "av_fusion"
CONDITIONS = ["clean", "white_10", "white_0", "white_-5", "babble_10", "babble_0", "babble_-5"]
SR = 16000


def truth(speaker: str, stem: str) -> str:
    words = [l.split()[2] for l in (GRID / "alignments" / speaker / f"{stem}.align").read_text().splitlines()]
    return " ".join(w for w in words if w not in ("sil", "sp"))


def load(name: str) -> dict:
    return json.loads((OUT / f"{name}.json").read_text())


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(f"wrote {OUT / f'{name}.json'}")


# ---------------------------------------------------------------- cands
def cands(speakers: list[str], per_speaker: int, n: int, seed: int):
    pool = sorted({truth(s.name, f.stem) for s in (GRID / "alignments").iterdir() if s.is_dir() for f in s.glob("*.align")})
    rng = random.Random(seed)
    clips = []
    for spk in speakers:
        files = sorted((GRID / spk).glob("*.mpg"))
        files = files[:: max(1, len(files) // per_speaker)][:per_speaker]
        for f in files:
            ref = truth(spk, f.stem)
            rw = ref.split()
            near = [p for p in pool if p != ref and len(p.split()) == len(rw) and sum(a != b for a, b in zip(p.split(), rw)) == 1]
            cs = [ref] + rng.sample(near, min(n - 1, len(near)))
            rng.shuffle(cs)
            clips.append({"id": f"{spk}/{f.stem}", "speaker": spk, "truth": ref, "cands": cs})
    save("cands", {"n": n, "distractors": "differ from the truth in exactly one word", "seed": seed, "clips": clips})


# ---------------------------------------------------------------- lips
def lips():
    import torch

    from services.vsr.app import MODEL_FPS, _load
    from services.vsr.phrases import score_phrases, tokenize
    from shared.service import decode_video_rgb

    m = _load()
    avsr = m["avsr"]
    piece_to_id = {p: i for i, p in enumerate(avsr.token_list)}
    rows = []
    for c in load("cands")["clips"]:
        spk, stem = c["id"].split("/")
        raw = (GRID / spk / f"{stem}.mpg").read_bytes()
        t0 = time.perf_counter()
        try:
            frames, _ = decode_video_rgb(raw, MODEL_FPS)
            det = m["detector"]
            lms = det.detect(frames, det.full_range_detector)
            if all(l is None for l in lms):
                lms = det.detect(frames, det.short_range_detector)
            x = m["transform"](torch.tensor(m["process"](frames, lms)))
            with torch.no_grad():
                enc = avsr.model.encode(x.to(avsr.device))
                s = score_phrases(avsr.model, enc, [tokenize(t, piece_to_id) for t in c["cands"]])
            scores = [float(v) for v in s]
        except Exception as exc:  # no face found: the lip agent has nothing to say
            print(f"  skip {c['id']}: {type(exc).__name__}")
            scores = None
        rows.append({"id": c["id"], "scores": scores, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        if len(rows) % 25 == 0:
            print(f"  lips {len(rows)} clips, median {statistics.median(r['ms'] for r in rows):.0f} ms", flush=True)
    save("lips", {"model": "Auto-AVSR LRS3_V_WER19.1 (deployed, not fine-tuned)", "device": str(avsr.device), "rows": rows})


# ---------------------------------------------------------------- voice
def audio_of(spk: str, stem: str) -> np.ndarray:
    import av

    with av.open(str(GRID / spk / f"{stem}.mpg")) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        pcm = [f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in rs.resample(fr)]
    return np.concatenate(pcm).astype(np.float32)


def degrade(wav: np.ndarray, cond: str, babble_src: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if cond == "clean":
        return wav
    kind, snr = cond.split("_")
    if kind == "white":
        noise = rng.standard_normal(len(wav)).astype(np.float32)
    else:  # three other-speaker utterances summed: someone else talking in the room
        noise = np.zeros(len(wav), np.float32)
        for b in rng.choice(len(babble_src), 3, replace=False):
            src = babble_src[b]
            reps = np.tile(src, int(np.ceil(len(wav) / len(src))))
            noise += reps[: len(wav)]
    p_sig, p_noise = float(np.mean(wav ** 2)) + 1e-12, float(np.mean(noise ** 2)) + 1e-12
    out = wav + noise * np.sqrt(p_sig / (p_noise * 10 ** (float(snr) / 10)))
    return (out / max(1.0, float(np.max(np.abs(out))))).astype(np.float32)


def voice(checkpoint: str):
    import torch

    sys.path.insert(0, str(ROOT / "asr_eval"))
    from harness.models import make

    model = make({"name": "nemotron-streaming-0.6b", "kind": "nemo_transducer", "checkpoint": checkpoint},
                 {"latency": {"dtype_cuda": "bfloat16"}})
    model.load()
    clips = load("cands")["clips"]
    wavs = {c["id"]: audio_of(*c["id"].split("/")) for c in clips}
    by_spk: dict[str, list[np.ndarray]] = {}
    for c in clips:
        by_spk.setdefault(c["speaker"], []).append(wavs[c["id"]])
    rng = np.random.default_rng(0)
    model.score_phrases(next(iter(wavs.values())), ["warm up"])  # first call compiles kernels
    rows = []
    for c in clips:
        other = [w for s, ws in by_spk.items() if s != c["speaker"] for w in ws]
        row = {"id": c["id"], "scores": {}, "ms": {}}
        for cond in CONDITIONS:
            w = degrade(wavs[c["id"]], cond, other, rng)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s = model.score_phrases(w, c["cands"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            row["ms"][cond] = round((time.perf_counter() - t0) * 1000, 1)
            row["scores"][cond] = [float(v[0]) for v in s]
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"  voice {len(rows)} clips, clean median {statistics.median(r['ms']['clean'] for r in rows):.0f} ms", flush=True)
    save("voice", {"model": checkpoint, "device": model.dev, "conditions": CONDITIONS, "rows": rows})


# ---------------------------------------------------------------- fuse
def _logsm(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _fit_t(scores: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising the NLL of the true candidate (grid search, log-spaced)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(50), 120))
    nll = [-_logsm(scores / t)[np.arange(len(y)), y].mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def _tau(logp: np.ndarray, y: np.ndarray, risk: float) -> float:
    """Lowest threshold whose wrong commits on calibration stay within risk x n (1.01 = never commit)."""
    pmax, ok = np.exp(logp.max(axis=1)), logp.argmax(axis=1) == y
    for t in sorted(set(np.round(pmax, 6))) + [1.01]:
        if ((pmax >= t) & ~ok).sum() <= int(risk * len(y)):
            return float(t)
    return 1.01


def fuse(risk: float, pooled: bool):
    cand = {c["id"]: c for c in load("cands")["clips"]}
    lip = {r["id"]: r for r in load("lips")["rows"]}
    vo = load("voice")
    voi = {r["id"]: r for r in vo["rows"]}
    ids = [i for i in cand if lip.get(i, {}).get("scores") and i in voi]
    y = np.array([cand[i]["cands"].index(cand[i]["truth"]) for i in ids])
    spk = np.array([cand[i]["speaker"] for i in ids])
    L = np.array([lip[i]["scores"] for i in ids])
    res = {"n_clips": len(ids), "lip_no_face": sum(1 for i in cand if not lip.get(i, {}).get("scores")), "risk_target": risk,
           "n_candidates": L.shape[1], "calibration": "pooled over all noise conditions" if pooled else "per condition",
           "conditions": {}}
    speakers = sorted(set(spk))
    VA = {c: np.array([voi[i]["scores"][c] for i in ids]) for c in vo["conditions"]}
    taus: dict = {}
    if pooled:  # one temperature pair and one threshold per policy, fitted on every condition of the other speaker
        for test in speakers:
            cal = spk != test
            Vc = np.concatenate([VA[c][cal] for c in vo["conditions"]])
            Lc, yc = np.concatenate([L[cal]] * len(vo["conditions"])), np.concatenate([y[cal]] * len(vo["conditions"]))
            tv, tl = _fit_t(Vc, yc), _fit_t(L[cal], y[cal])
            fused_c = _logsm(_logsm(Vc / tv) + _logsm(Lc / tl))
            taus[test] = (tv, tl, {"voice": _tau(_logsm(Vc / tv), yc, risk), "lips": _tau(_logsm(L[cal] / tl), y[cal], risk),
                                   "fused": _tau(fused_c, yc, risk)})
    for cond in vo["conditions"]:
        V = VA[cond]
        agg = {k: {"correct": 0, "wrong": 0, "abstain": 0, "top1": 0} for k in ("voice", "lips", "fused")}
        for test in speakers:  # calibrate on the other speaker, test on this one
            cal, tst = spk != test, spk == test
            tv, tl = taus[test][:2] if pooled else (_fit_t(V[cal], y[cal]), _fit_t(L[cal], y[cal]))
            lp = {"voice": lambda m: _logsm(V[m] / tv), "lips": lambda m: _logsm(L[m] / tl),
                  "fused": lambda m: _logsm(_logsm(V[m] / tv) + _logsm(L[m] / tl))}
            for k, f in lp.items():
                tau = taus[test][2][k] if pooled else _tau(f(cal), y[cal], risk)
                p = f(tst)
                commit, ok = np.exp(p.max(axis=1)) >= tau, p.argmax(axis=1) == y[tst]
                agg[k]["correct"] += int((commit & ok).sum())
                agg[k]["wrong"] += int((commit & ~ok).sum())
                agg[k]["abstain"] += int((~commit).sum())
                agg[k]["top1"] += int(ok.sum())
        n = len(ids)
        res["conditions"][cond] = {k: {**v, "correct_rate": round(v["correct"] / n, 4), "top1_rate": round(v["top1"] / n, 4)}
                                   for k, v in agg.items()}
    lip_ms = [lip[i]["ms"] for i in ids]
    voice_ms = [voi[i]["ms"]["clean"] for i in ids]
    both = [max(a, b) for a, b in zip(lip_ms, voice_ms)]
    q = lambda xs: {"p50": round(float(np.percentile(xs, 50)), 1), "p95": round(float(np.percentile(xs, 95)), 1)}
    res["latency_ms"] = {"voice": q(voice_ms), "lips": q(lip_ms), "fused_parallel": q(both),
                         "devices": {"voice": vo["device"], "lips": load("lips")["device"]}}
    save("fusion_pooled" if pooled else "fusion", res)
    print(f"\n{'condition':<11} {'':>6}" + "".join(f"{k:>24}" for k in ("voice", "lips", "fused")))
    for cond, r in res["conditions"].items():
        print(f"{cond:<11} correct" + "".join(f"{r[k]['correct_rate']:>17.1%} ({r[k]['wrong']:>2}w)" for k in ("voice", "lips", "fused")))
    print("latency ms", res["latency_ms"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("cands")
    c.add_argument("--speakers", nargs="+", default=["s1", "s2"])
    c.add_argument("--per-speaker", type=int, default=100)
    c.add_argument("--n", type=int, default=10)
    c.add_argument("--seed", type=int, default=0)
    sub.add_parser("lips")
    v = sub.add_parser("voice")
    v.add_argument("--checkpoint", default="nvidia/nemotron-speech-streaming-en-0.6b")
    f = sub.add_parser("fuse")
    f.add_argument("--risk", type=float, default=0.02)
    f.add_argument("--pooled", action="store_true", help="one calibration for all noise conditions (the system does not know the SNR)")
    a = ap.parse_args()
    if a.stage == "cands":
        cands(a.speakers, a.per_speaker, a.n, a.seed)
    elif a.stage == "lips":
        lips()
    elif a.stage == "voice":
        voice(a.checkpoint)
    else:
        fuse(a.risk, a.pooled)
