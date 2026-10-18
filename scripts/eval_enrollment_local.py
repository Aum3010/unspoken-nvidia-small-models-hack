"""Which way of using the person's own recordings reads their phrases best?

Runs on the stored enrollment features (data/enroll/, see services/vsr/enroll.py) of the 20 request
phrases (services/vsr/patient_phrases.txt), with folds by take: fold k holds out the k-th take of
every phrase, and every method is built from the other takes only. Methods, each giving a score
for all 20 phrases:

  dtw    nearest recording, DTW over cosine distance (what the lip agent did so far)
  cls    logistic regression on pooled features (mean, std, and the means of 3 time segments)
  phr    phrase-list score: log p(phrase | video) from the original decoder + CTC
  ft     the same score after fine-tuning the decoder + CTC on the training takes (encoder frozen)
  a+b    fusion: each method's 20 scores are z-normalised per clip and summed with equal weights
         (no weights are tuned on the test takes)

Features recorded before raw features were kept are unit-length per frame. They are rescaled by
RESCALE (the typical frame norm of the encoder output, ~9 on GRID) before the decoder sees them;
recordings that have raw features (raw_*.npy) use those.

    PYTHONPATH=. .venv/bin/python scripts/eval_enrollment_local.py [--epochs 12]
Writes evaluations/results/vsr_enrollment_local.json.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from services.vsr.enroll import ENROLL_DIR
from services.vsr.phrases import dtw, score_phrases, tokenize
from shared.provenance import ROOT

PHRASES = [l.strip() for l in (ROOT / "services/vsr/patient_phrases.txt").read_text().splitlines()
           if l.strip() and not l.startswith("#")]
RESCALE = 9.0
OUT = ROOT / "evaluations/results/vsr_enrollment_local.json"


def load_takes():
    takes = []  # (phrase index, take index, unit features, decoder features)
    for d in sorted(ENROLL_DIR.glob("*/")):
        name = (d / "phrase.txt").read_text().strip() if (d / "phrase.txt").exists() else ""
        if name not in PHRASES:
            continue
        for k, f in enumerate(sorted(d.glob("[0-9]*.npy"))):
            unit = np.load(f).astype(np.float32)
            raw_f = d / f"raw_{f.name}"
            dec = np.load(raw_f).astype(np.float32) if raw_f.exists() else unit * RESCALE
            takes.append((PHRASES.index(name), k, unit, dec))
    return takes


def pooled(u: np.ndarray) -> np.ndarray:
    thirds = [s.mean(0) for s in np.array_split(u, 3)]
    return np.concatenate([u.mean(0), u.std(0), *thirds])


def zs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)


def finetune(model, train, tokens, device, epochs: int, lr: float = 1e-4, ctc_weight: float = 0.3, bs: int = 8):
    from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, pad_list
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
    from espnet.nets.pytorch_backend.transformer.mask import target_mask

    for p in model.parameters():
        p.requires_grad = False
    params = list(model.decoder.parameters()) + list(model.ctc.parameters())
    for p in params:
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(0)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train))
        for i in range(0, len(order), bs):
            batch = [train[j] for j in order[i:i + bs]]
            feats = [torch.from_numpy(t[3]) for t in batch]
            hlens = torch.tensor([len(x) for x in feats])
            hs = pad_list(feats, 0.0).to(device)
            mask = make_non_pad_mask(hlens).to(device).unsqueeze(-2)
            ys = pad_list([torch.tensor(tokens[t[0]]) for t in batch], model.ignore_id).to(device)
            ys_in, ys_out = add_sos_eos(ys, model.sos, model.eos, model.ignore_id)
            pred, _ = model.decoder(ys_in, target_mask(ys_in, model.ignore_id), hs, mask)
            loss = (1 - ctc_weight) * model.criterion(pred, ys_out) + ctc_weight * model.ctc(hs, hlens.to(device), ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    from services.vsr.app import _load
    from sklearn.linear_model import LogisticRegression

    m = _load()
    model, device = m["avsr"].model, m["avsr"].device
    p2i = {p: i for i, p in enumerate(m["avsr"].token_list)}
    tokens = [tokenize(p, p2i) for p in PHRASES]
    base = copy.deepcopy({"decoder": model.decoder.state_dict(), "ctc": model.ctc.state_dict()})

    takes = load_takes()
    n_phr = len({t[0] for t in takes})
    print(f"{len(takes)} takes of {n_phr} phrases")
    folds = sorted({t[1] for t in takes})
    scores = {k: [None] * len(takes) for k in ("dtw", "cls", "phr", "ft")}
    ms = {k: [] for k in scores}

    def phrase_scores(dec):
        with torch.no_grad():
            return score_phrases(model, torch.from_numpy(dec).to(device), tokens).cpu().numpy()

    for i, t in enumerate(takes):  # the original decoder does not depend on the fold
        t0 = time.time(); scores["phr"][i] = phrase_scores(t[3]); ms["phr"].append(time.time() - t0)

    for f in folds:
        test = [i for i, t in enumerate(takes) if t[1] == f]
        train = [t for t in takes if t[1] != f]
        X = np.stack([pooled(t[2]) for t in train]); y = [t[0] for t in train]
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        t0 = time.time()
        finetune(model, train, tokens, device, a.epochs)
        print(f"fold {f}: {len(test)} held-out takes, {len(train)} training takes, fine-tune {time.time() - t0:.0f}s", flush=True)
        for i in test:
            u, dec = takes[i][2], takes[i][3]
            t0 = time.time()
            s = np.full(len(PHRASES), -np.inf)
            for t in train:
                s[t[0]] = max(s[t[0]], -dtw(u, t[2]))
            scores["dtw"][i] = s; ms["dtw"].append(time.time() - t0)
            t0 = time.time()
            p = np.full(len(PHRASES), -1e9)
            p[clf.classes_] = clf.predict_log_proba(pooled(u)[None])[0]
            scores["cls"][i] = p; ms["cls"].append(time.time() - t0)
            t0 = time.time(); scores["ft"][i] = phrase_scores(dec); ms["ft"].append(time.time() - t0)
        model.decoder.load_state_dict(base["decoder"]); model.ctc.load_state_dict(base["ctc"])

    truth = np.array([t[0] for t in takes])
    present = sorted(set(truth))  # rank only among phrases that have recordings

    def acc(mat):
        mat = np.stack(mat)[:, present]
        ranks = (mat > mat[np.arange(len(truth)), [present.index(x) for x in truth]][:, None]).sum(1)
        return {"top1": round(float((ranks == 0).mean()), 3), "top3": round(float((ranks < 3).mean()), 3)}

    results = {k: acc(v) for k, v in scores.items()}
    for r in range(2, 5):
        for combo in itertools.combinations(scores, r):
            fused = [sum(zs(np.stack(scores[k])[i][present]) for k in combo) for i in range(len(takes))]
            fused = [np.pad(x, (0, 0)) for x in fused]
            mat = np.stack(fused)
            idx = np.array([present.index(x) for x in truth])
            ranks = (mat > mat[np.arange(len(truth)), idx][:, None]).sum(1)
            results["+".join(combo)] = {"top1": round(float((ranks == 0).mean()), 3), "top3": round(float((ranks < 3).mean()), 3)}
    timing = {k: round(1000 * float(np.median(v)), 1) for k, v in ms.items()}

    print(f"\n{'method':16s} top-1  top-3")
    for k, v in sorted(results.items(), key=lambda kv: (-kv[1]["top1"], -kv[1]["top3"])):
        print(f"{k:16s} {v['top1']:.0%}   {v['top3']:.0%}")
    print("median ms per clip:", timing)
    OUT.write_text(json.dumps({"takes": len(takes), "phrases": n_phr, "folds": len(folds), "epochs": a.epochs,
                               "rescale": RESCALE, "results": results, "median_ms": timing, "device": str(device)}, indent=2))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
