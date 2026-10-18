"""Can the lip agent be faster without losing accuracy? Same clips, same pipeline, one setting changed at a time.

    cd ~/unspoken && PYTHONPATH=. /workspace/venv-app/bin/python evaluations/modality_baselines/vsr_speed_accuracy.py

Loads the lip model exactly as services/vsr does, then decodes a fixed set of GRID clips (4 per speaker, every
speaker on disk, unseen by the LRS3-trained model) under: beam 12 / beam 6 x fp32 / bf16 autocast. Face detection
and the mouth crop are done once per clip and shared, so the settings see identical inputs. Reports WER, best-of-5
WER, CER and per-stage time. A setting is only acceptable if no accuracy number gets worse.
Writes evaluations/results/vsr_speed_accuracy.json.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import jiwer
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GRID = ROOT / "data" / "grid"
PER_SPEAKER = int(os.environ.get("PER_SPEAKER", 4))


def truth(speaker: str, stem: str) -> str:
    words = [l.split()[2] for l in (GRID / "alignments" / speaker / f"{stem}.align").read_text().splitlines()]
    return " ".join(w for w in words if w not in ("sil", "sp"))


def clips() -> list[Path]:
    out = []
    for spk in sorted((p for p in GRID.iterdir() if p.is_dir() and p.name[1:].isdigit()), key=lambda p: int(p.name[1:])):
        cs = sorted(spk.glob("*.mpg"))
        if not cs or not (GRID / "alignments" / spk.name).exists():
            continue
        out += cs[:: max(1, len(cs) // PER_SPEAKER)][:PER_SPEAKER]
    return out


def sync():
    torch.cuda.synchronize()


def main():
    from services.vsr import app as V  # puts third_party/chaplin on sys.path
    from shared.service import decode_video_rgb

    from pipelines.model import AVSR

    m = V._load()  # beam from UNSPOKEN_VSR_BEAM (default 12), exactly as the service
    models = V.CHAPLIN / "benchmarks" / "LRS3"
    beams = {V.BEAM: m["avsr"]}
    for b in (6,):
        if b not in beams:
            beams[b] = AVSR("video", str(models / "models/LRS3_V_WER19.1/model.pth"), str(models / "models/LRS3_V_WER19.1/model.json"),
                            str(models / "language_models/lm_en_subword/model.pth"), str(models / "language_models/lm_en_subword/model.json"),
                            penalty=0.0, ctc_weight=0.1, lm_weight=0.3, beam_size=b, device=m["avsr"].device)
    settings = [(b, dt) for b in sorted(beams, reverse=True) for dt in ("fp32", "bf16")]
    rows = {f"beam{b}_{dt}": [] for b, dt in settings}
    prep_ms = []
    cl = clips()
    print(f"{len(cl)} clips, settings {list(rows)}", flush=True)
    for k, clip in enumerate(cl):
        ref = truth(clip.parent.name, clip.stem)
        t0 = time.perf_counter()
        frames, _ = decode_video_rgb(clip.read_bytes(), V.MODEL_FPS)
        lms = m["detector"].detect(frames, m["detector"].full_range_detector)
        if all(lm is None for lm in lms):
            lms = m["detector"].detect(frames, m["detector"].short_range_detector)
        try:
            crops = m["process"](frames, lms)
        except Exception:
            continue
        x = m["transform"](torch.tensor(crops))
        prep_ms.append((time.perf_counter() - t0) * 1000)
        for b, dt in settings:
            avsr = beams[b]
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dt == "bf16" else torch.autocast("cuda", enabled=False)
            sync(); t1 = time.perf_counter()
            with torch.no_grad(), ctx:
                enc = avsr.model.encode(x.to(avsr.device))
                sync(); t2 = time.perf_counter()
                hyps = avsr.beam_search(enc)
            sync(); t3 = time.perf_counter()
            texts = [h.text.lower() for h in V._nbest_texts(avsr, hyps)]
            top1 = texts[0] if texts else ""
            rows[f"beam{b}_{dt}"].append({"clip": f"{clip.parent.name}/{clip.name}", "truth": ref, "top1": top1,
                                          "wer": jiwer.wer(ref, top1) if top1 else 1.0, "cer": jiwer.cer(ref, top1) if top1 else 1.0,
                                          "oracle_wer": min([jiwer.wer(ref, t) for t in texts] or [1.0]),
                                          "encode_ms": (t2 - t1) * 1000, "beam_ms": (t3 - t2) * 1000})
        if k % 20 == 0:
            print(f"  {k}/{len(cl)}", flush=True)

    base = f"beam{V.BEAM}_fp32"
    summary = {}
    for name, rs in rows.items():
        same = sum(1 for a, b in zip(rs, rows[base]) if a["top1"] == b["top1"])
        summary[name] = {"n": len(rs), "wer": round(statistics.fmean(r["wer"] for r in rs), 4), "cer": round(statistics.fmean(r["cer"] for r in rs), 4),
                         "oracle_wer": round(statistics.fmean(r["oracle_wer"] for r in rs), 4), "same_top1_as_baseline": f"{same}/{len(rs)}",
                         "encode_ms_p50": round(statistics.median(r["encode_ms"] for r in rs)), "beam_ms_p50": round(statistics.median(r["beam_ms"] for r in rs)),
                         "decode_ms_p50": round(statistics.median(r["encode_ms"] + r["beam_ms"] for r in rs))}
    out = {"date": time.strftime("%Y-%m-%d"), "gpu": torch.cuda.get_device_name(0), "baseline": base, "clips": len(cl),
           "prep_ms_p50 (decode video + face detection + mouth crop, shared)": round(statistics.median(prep_ms)),
           "summary": summary, "rows": rows}
    (ROOT / "evaluations" / "results" / "vsr_speed_accuracy.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
