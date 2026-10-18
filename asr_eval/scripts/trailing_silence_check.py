"""Does early voice decode change any decision? Own recordings, cut two ways, through the app's decision code.

The app ends an utterance after 700 ms of silence. With early decode (services/gateway, web/src/glass/sensors.ts)
the clip that is decoded and committed ends 350 ms after the speech instead. This checks, on the real recordings,
that the shorter trailing silence does not change the decision: each clip is cut 700 ms and 350 ms after its last
speech frame (Silero, the same gate the decision uses), and both go through harness.live.Bench, exactly as
services/audio runs it (Nemotron en, forced_llr, balanced).

    cd ~/unspoken/asr_eval && /workspace/venv-asr/bin/python scripts/trailing_silence_check.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import audio  # noqa: E402
from harness.config import load_config  # noqa: E402
from harness.live import Bench  # noqa: E402

SR = audio.SR
FRAME = 512
NAME = re.compile(r"^(?P<spk>[a-z0-9]+)_(?P<cond>[a-z]+)_(?P<label>[a-z0-9_]+)_(?P<take>\d+)\.wav$")


def main():
    cfg = load_config(ROOT / "configs" / "eval.yaml")
    own = Path("/workspace/asr_eval_data/own")
    files = sorted(p for p in own.glob("*.wav") if NAME.match(p.name))
    bench = Bench(cfg, "nemotron-streaming-0.6b", "forced_llr", "balanced")
    thr = bench.th["vad_t"]
    rows = []
    for k, p in enumerate(files):
        m = NAME.match(p.name)
        label = m["label"]
        wav = audio.load(p)
        probs = bench.vad.probs(wav)
        speech = np.flatnonzero(probs >= thr)
        if not len(speech):
            continue  # nothing for the browser's detector to cut either
        end = (speech[-1] + 1) * FRAME
        out = {"clip": p.name, "label": label, "condition": m["cond"], "tail_available_ms": round((len(wav) - end) / SR * 1000)}
        for name, tail_ms in (("cut700", 700), ("cut350", 350)):
            r = bench.recognize(wav[: min(len(wav), end + int(SR * tail_ms / 1000))])
            out[name] = r["phrase_id"] if r["decision"] == "candidate" else r["decision"]
        rows.append(out)
        if k % 50 == 0:
            print(f"  {k}/{len(files)}", flush=True)

    def score(key):
        pos = [r for r in rows if not r["label"].startswith("neg_")]
        neg = [r for r in rows if r["label"].startswith("neg_")]
        return {"correct_accept": f"{sum(r[key] == r['label'] for r in pos)}/{len(pos)}",
                "wrong_accept": f"{sum(r[key] not in ('unknown', 'no_speech', r['label']) for r in pos)}/{len(pos)}",
                "false_accept_on_negatives": f"{sum(r[key] not in ('unknown', 'no_speech') for r in neg)}/{len(neg)}"}

    changed = [r for r in rows if r["cut700"] != r["cut350"]]
    res = {"clips": len(rows), "cut700": score("cut700"), "cut350": score("cut350"),
           "decisions_changed": len(changed), "changed": changed,
           "short_tail_clips": sum(r["tail_available_ms"] < 700 for r in rows)}
    print(json.dumps({k: v for k, v in res.items() if k != "changed"}, indent=1))
    for r in changed:
        print("  changed:", r)
    (ROOT / "results" / "trailing_silence_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
