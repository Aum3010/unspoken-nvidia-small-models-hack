"""Lip-reading baseline on GRID (CC BY 4.0): speakers and grammar unseen by the LRS3-trained model.

Also a still-face test: a clip made by freezing one frame, i.e. no articulation —
the system should flag it rather than present fluent text as a reading.
Requires the VSR service running (scripts/run_all.sh).
"""

from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

import av
import httpx
import jiwer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data" / "grid"
OUT = ROOT / "evaluations" / "results"
VSR = "http://127.0.0.1:8101/infer"


def truth(speaker: str, stem: str) -> str:
    words = [l.split()[2] for l in (GRID / "alignments" / speaker / f"{stem}.align").read_text().splitlines()]
    return " ".join(w for w in words if w not in ("sil", "sp"))


def still_clip(src: Path, seconds: float = 3.0) -> bytes:
    with av.open(str(src)) as c:
        frame = next(c.decode(video=0)).to_ndarray(format="rgb24")
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        s = out.add_stream("libx264", rate=25)
        s.width, s.height, s.pix_fmt = frame.shape[1], frame.shape[0], "yuv420p"
        for _ in range(int(25 * seconds)):
            noisy = np.clip(frame.astype(np.int16) + np.random.randint(-2, 3, frame.shape), 0, 255).astype(np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(noisy, format="rgb24")):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    return buf.getvalue()


def main(per_speaker: int = 20, still_n: int = 8):
    rows = []
    client = httpx.Client(timeout=180)
    # s1/s2 only: s3+ may be on disk as fine-tuning data (scripts/finetune_vsr_grid.py)
    speakers = [s for s in ("s1", "s2") if (GRID / s).is_dir()]
    for spk in speakers:
        clips = sorted((GRID / spk).glob("*.mpg"))
        for clip in clips[:: max(1, len(clips) // per_speaker)][:per_speaker]:
            ref = truth(spk, clip.stem)
            r = client.post(VSR, files={"video": (clip.name, clip.read_bytes(), "video/mpeg")}).json()
            hyps = [h["text"].lower() for h in r["hypotheses"]]
            top1 = hyps[0] if hyps else ""
            row = {"clip": f"{spk}/{clip.name}", "truth": ref, "top1": top1, "nbest": hyps,
                   "wer": jiwer.wer(ref, top1) if top1 else 1.0, "cer": jiwer.cer(ref, top1) if top1 else 1.0,
                   "oracle_wer": min([jiwer.wer(ref, h) for h in hyps] or [1.0]),
                   "flags": r["quality_flags"], "latency_ms": r["provenance"]["latency_ms"], "motion": r["quality"].get("mouth_motion")}
            rows.append(row)
            print(f"{row['clip']:<18} WER {row['wer']:.2f}  {ref!r:<38} -> {top1!r}", flush=True)

    still = []
    for clip in sorted((GRID / speakers[0]).glob("*.mpg"))[:still_n]:
        r = client.post(VSR, files={"video": ("still.mp4", still_clip(clip), "video/mp4")}).json()
        still.append({"flags": r["quality_flags"], "top1": r["hypotheses"][0]["text"] if r["hypotheses"] else "",
                      "motion": r["quality"].get("mouth_motion")})
        print("still:", still[-1], flush=True)

    flagged = [s for s in still if "low_mouth_motion" in s["flags"] or "no_face" in s["flags"]]
    res = {
        "dataset": "GRID corpus (unseen speakers)",
        "speakers": speakers,
        "n_clips": len(rows),
        "wer": round(statistics.mean(r["wer"] for r in rows), 4),
        "cer": round(statistics.mean(r["cer"] for r in rows), 4),
        "oracle_wer": round(statistics.mean(r["oracle_wer"] for r in rows), 4),
        "latency_ms_median": int(statistics.median(r["latency_ms"] for r in rows)),
        "speech_motion_median": round(statistics.median(r["motion"] for r in rows if r["motion"] is not None), 3),
        "still": {"n": len(still), "flag_rate": round(len(flagged) / max(1, len(still)), 3),
                  "false_phrase_rate": round(sum(1 for s in still if s["top1"] and s not in flagged) / max(1, len(still)), 3),
                  "motion_median": round(statistics.median(s["motion"] for s in still if s["motion"] is not None), 3) if still else None},
        "examples": sorted(rows, key=lambda r: r["wer"])[:3] + sorted(rows, key=lambda r: -r["wer"])[:3],
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vsr.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "examples")}, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
