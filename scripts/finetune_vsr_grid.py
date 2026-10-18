"""Adapt Auto-AVSR to GRID: frozen visual encoder, fine-tuned decoder + CTC head.

Stages (run in order; each caches its output under data/cache/vsr_ft/):
  extract  mouth crops -> frozen encoder features for every clip (one-time, slow part)
  train    fine-tune decoder + CTC head on training speakers (encoder stays frozen)
  eval     beam-search WER on held-out speakers: original vs fine-tuned decoder

Held-out test speakers (s1, s2) are the same ones used in eval_vsr_grid.py and are
never seen in training. Usage:
  PYTHONPATH=$PWD .venv/bin/python scripts/finetune_vsr_grid.py extract --speakers s1 s2 s3 s4 s5 s6 s7 s8
  PYTHONPATH=$PWD .venv/bin/python scripts/finetune_vsr_grid.py train --train s3 s4 s5 s6 s7 s8
  PYTHONPATH=$PWD .venv/bin/python scripts/finetune_vsr_grid.py eval --test s1 s2
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import jiwer
import numpy as np
import torch

from services.vsr.app import CHAPLIN, MODEL_FPS, _load
from services.vsr.phrases import score_phrases, tokenize
from shared.service import decode_video_rgb

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data" / "grid"
CACHE = ROOT / "data" / "cache" / "vsr_ft"
CKPT = CACHE / "decoder_ctc_ft.pt"
OUT = ROOT / "evaluations" / "results" / "vsr_finetune.json"


def truth(speaker: str, stem: str) -> str:
    words = [l.split()[2] for l in (GRID / "alignments" / speaker / f"{stem}.align").read_text().splitlines()]
    return " ".join(w for w in words if w not in ("sil", "sp"))


# ---------------------------------------------------------------- extract
def extract(speakers: list[str], per_speaker: int):
    m = _load()
    avsr = m["avsr"]
    for spk in speakers:
        out = CACHE / spk
        out.mkdir(parents=True, exist_ok=True)
        clips = sorted((GRID / spk).glob("*.mpg"))
        if per_speaker:
            clips = clips[:: max(1, len(clips) // per_speaker)][:per_speaker]
        t0, done, failed = time.time(), 0, 0
        for clip in clips:
            dst = out / f"{clip.stem}.npy"
            if dst.exists():
                continue
            try:
                frames, _ = decode_video_rgb(clip.read_bytes(), MODEL_FPS)
                det = m["detector"]
                lms = det.detect(frames, det.full_range_detector)
                if all(l is None for l in lms):
                    lms = det.detect(frames, det.short_range_detector)
                crops = m["process"](frames, lms)
                x = m["transform"](torch.tensor(crops))
                with torch.no_grad():
                    enc = avsr.model.encode(x.to(avsr.device))
                np.save(dst, enc.cpu().numpy().astype(np.float16))
                done += 1
            except Exception as exc:  # unreadable clip or no face: skip it
                failed += 1
                print(f"  skip {spk}/{clip.name}: {type(exc).__name__}")
            if done and done % 50 == 0:
                print(f"  {spk}: {done}/{len(clips)}  {(time.time() - t0) / done:.2f}s/clip", flush=True)
        print(f"{spk}: extracted {done}, failed {failed}", flush=True)


# ---------------------------------------------------------------- train
def load_split(speakers: list[str], piece_to_id):
    items = []
    for spk in speakers:
        for f in sorted((CACHE / spk).glob("*.npy")):
            if not (GRID / "alignments" / spk / f"{f.stem}.align").exists():
                continue  # a few GRID videos ship without a transcript
            text = truth(spk, f.stem)
            items.append((f, text, tokenize(text, piece_to_id)))
    return items


def batch_loss(model, batch, device, ctc_weight):
    from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, pad_list
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
    from espnet.nets.pytorch_backend.transformer.mask import target_mask

    feats = [torch.from_numpy(np.load(f).astype(np.float32)) for f, _, _ in batch]
    hlens = torch.tensor([len(x) for x in feats])
    hs = pad_list(feats, 0.0).to(device)
    hs_mask = make_non_pad_mask(hlens).to(device).unsqueeze(-2)
    ys = pad_list([torch.tensor(t) for _, _, t in batch], model.ignore_id).to(device)

    ys_in, ys_out = add_sos_eos(ys, model.sos, model.eos, model.ignore_id)
    pred, _ = model.decoder(ys_in, target_mask(ys_in, model.ignore_id), hs, hs_mask)
    loss_att = model.criterion(pred, ys_out)
    loss_ctc = model.ctc(hs, hlens.to(device), ys)
    return (1 - ctc_weight) * loss_att + ctc_weight * loss_ctc


def train(speakers: list[str], epochs: int, batch_size: int, lr: float, ctc_weight: float):
    m = _load()
    avsr, device = m["avsr"], m["avsr"].device
    model = avsr.model
    piece_to_id = {p: i for i, p in enumerate(avsr.token_list)}
    items = load_split(speakers, piece_to_id)
    random.seed(0)
    random.shuffle(items)
    n_val = max(20, len(items) // 20)
    val, tr = items[:n_val], items[n_val:]
    print(f"train {len(tr)} clips, val {len(val)} clips, speakers {speakers}")

    for p in model.parameters():
        p.requires_grad = False
    params = list(model.decoder.parameters()) + list(model.ctc.parameters())
    for p in params:
        p.requires_grad = True
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M (encoder frozen)")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    best = float("inf")
    for ep in range(epochs):
        model.decoder.train(); model.ctc.train()
        random.shuffle(tr)
        losses, t0 = [], time.time()
        for i in range(0, len(tr), batch_size):
            loss = batch_loss(model, tr[i:i + batch_size], device, ctc_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            losses.append(loss.item())
        model.decoder.eval(); model.ctc.eval()
        with torch.no_grad():
            vl = statistics.mean(batch_loss(model, val[i:i + batch_size], device, ctc_weight).item()
                                 for i in range(0, len(val), batch_size))
        print(f"epoch {ep + 1}/{epochs}  train {statistics.mean(losses):.3f}  val {vl:.3f}  {time.time() - t0:.0f}s", flush=True)
        if vl < best:
            best = vl
            torch.save({"decoder": model.decoder.state_dict(), "ctc": model.ctc.state_dict(),
                        "train_speakers": speakers, "epoch": ep + 1, "val_loss": vl}, CKPT)
    print(f"saved best (val {best:.3f}) -> {CKPT}")


# ---------------------------------------------------------------- eval
def decode_all(avsr, items, lm_weight: float, beam: int):
    from pipelines.model import get_beam_search_decoder

    models = CHAPLIN / "benchmarks" / "LRS3" / "language_models" / "lm_en_subword"
    lm = (str(models / "model.pth"), str(models / "model.json")) if lm_weight > 0 else (None, None)
    bs = get_beam_search_decoder(avsr.model, avsr.token_list, lm[0], lm[1], 0.0, 0.1, lm_weight, beam)
    bs.to(device=avsr.device).eval()
    rows = []
    for f, ref, _ in items:
        enc = torch.from_numpy(np.load(f).astype(np.float32)).to(avsr.device)
        t0 = time.time()
        with torch.no_grad():
            hyps = bs(enc)
        ms = int((time.time() - t0) * 1000)
        d = hyps[0].asdict() if hyps else {"yseq": []}
        top1 = "".join(avsr.token_list[int(i)] for i in d["yseq"][1:-1]).replace("▁", " ").replace("<eos>", "").strip().lower()
        rows.append({"clip": f"{f.parent.name}/{f.stem}", "truth": ref, "top1": top1,
                     "wer": jiwer.wer(ref, top1) if top1 else 1.0, "decode_ms": ms})
    return rows


def evaluate(speakers: list[str], beam: int):
    m = _load()
    avsr = m["avsr"]
    piece_to_id = {p: i for i, p in enumerate(avsr.token_list)}
    items = load_split(speakers, piece_to_id)
    print(f"test {len(items)} clips from held-out speakers {speakers}")
    results = {}

    def run(name, lm_weight):
        rows = decode_all(avsr, items, lm_weight, beam)
        results[name] = {"wer": round(statistics.mean(r["wer"] for r in rows), 4),
                         "decode_ms_median": int(statistics.median(r["decode_ms"] for r in rows)),
                         "examples": rows[:5]}
        print(f"{name:<34} WER {results[name]['wer']:.3f}  decode {results[name]['decode_ms_median']} ms", flush=True)

    run("original (LRS3 LM 0.3, deployed)", 0.3)
    ck = torch.load(CKPT, map_location="cpu")
    avsr.model.decoder.load_state_dict(ck["decoder"])
    avsr.model.ctc.load_state_dict(ck["ctc"])
    run("fine-tuned decoder, no LM", 0.0)
    run("fine-tuned decoder + LRS3 LM 0.3", 0.3)

    OUT.write_text(json.dumps({"dataset": "GRID", "test_speakers": speakers, "train_speakers": ck["train_speakers"],
                               "n_test_clips": len(items), "beam": beam, "method": "frozen Auto-AVSR encoder; "
                               "fine-tuned decoder + CTC head", "results": results}, indent=2))
    print(f"wrote {OUT}")


# ---------------------------------------------------------------- phrases
def phrases(speakers: list[str], sizes: list[int], max_diff: int = 0):
    """Closed-list recognition: is the true sentence ranked first among N GRID sentences?"""
    m = _load()
    avsr = m["avsr"]
    piece_to_id = {p: i for i, p in enumerate(avsr.token_list)}
    items = load_split(speakers, piece_to_id)
    pool = sorted({truth(s.name, f.stem) for s in (GRID / "alignments").iterdir() if s.is_dir()
                   for f in s.glob("*.align")})
    print(f"{len(items)} test clips, distractor pool {len(pool)} GRID sentences")

    def run(name):
        out = {}
        for n in sizes:
            rng = random.Random(n)
            top1 = top3 = 0
            for f, ref, toks in items:
                others = [p for p in pool if p != ref]
                if max_diff:  # hard mode: distractors differ from the truth in at most max_diff word slots
                    rw = ref.split()
                    others = [p for p in others if len(p.split()) == len(rw)
                              and sum(a != b for a, b in zip(p.split(), rw)) <= max_diff]
                cands = [ref] + rng.sample(others, min(n - 1, len(others)))
                enc = torch.from_numpy(np.load(f).astype(np.float32)).to(avsr.device)
                with torch.no_grad():
                    s = score_phrases(avsr.model, enc, [tokenize(c, piece_to_id) for c in cands])
                rank = int((s > s[0]).sum())
                top1 += rank == 0
                top3 += rank < 3
            out[f"N={n}"] = {"top1": round(top1 / len(items), 4), "top3": round(top3 / len(items), 4)}
            print(f"{name:<22} N={n:<4} top-1 {top1 / len(items):.1%}  top-3 {top3 / len(items):.1%}  (chance {1 / n:.1%})", flush=True)
        return out

    res = {"distractors": f"differ in <= {max_diff} words" if max_diff else "random", "original": run("original")}
    if CKPT.exists():
        ck = torch.load(CKPT, map_location="cpu")
        avsr.model.decoder.load_state_dict(ck["decoder"])
        avsr.model.ctc.load_state_dict(ck["ctc"])
        res["fine-tuned"] = run("fine-tuned")
    dst = OUT.with_name(f"vsr_phrases{'_hard' if max_diff else ''}.json")
    dst.write_text(json.dumps({"test_speakers": speakers, "n_clips": len(items), "method":
                               "rank closed phrase list by teacher-forced decoder+CTC log-likelihood", "results": res}, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("extract"); e.add_argument("--speakers", nargs="+", required=True); e.add_argument("--per-speaker", type=int, default=0)
    t = sub.add_parser("train"); t.add_argument("--train", nargs="+", required=True)
    t.add_argument("--epochs", type=int, default=8); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--ctc-weight", type=float, default=0.3)
    v = sub.add_parser("eval"); v.add_argument("--test", nargs="+", required=True); v.add_argument("--beam", type=int, default=12)
    p = sub.add_parser("phrases"); p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--max-diff", type=int, default=0, help="hard mode: distractors differ in at most this many words")
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.speakers, a.per_speaker)
    elif a.stage == "train":
        train(a.train, a.epochs, a.batch_size, a.lr, a.ctc_weight)
    elif a.stage == "phrases":
        phrases(a.test, a.sizes, a.max_diff)
    else:
        evaluate(a.test, a.beam)
