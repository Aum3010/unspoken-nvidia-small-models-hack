"""Phase 1.1: push-to-talk CLI recorder, 16 kHz mono WAV, phrases in random order.

    python scripts/record.py --speaker alice --condition normal
    python scripts/record.py --list-devices

Files: data/own/{speaker}_{condition}_{phraseid}_{take}.wav (negatives: phraseid=neg_{name}).
No browser processing, no gain control: the raw mic signal is what the product sees.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import audio  # noqa: E402
from harness.config import load_config  # noqa: E402

CONSENT = ("I agree that these recordings of my voice are stored locally (and on the project's evaluation GPU "
           "machine) and used only to evaluate this prototype's speech recognition. They will not be published. "
           "I can ask for them to be deleted at any time, and they will be deleted after the hackathon.")


def record_once(sd, device, max_s: float) -> np.ndarray:
    chunks, stop = [], threading.Event()

    def cb(indata, frames, t, status):
        chunks.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=audio.SR, channels=1, dtype="float32", device=device, callback=cb):
        timer = threading.Timer(max_s, stop.set)
        timer.start()
        waiter = threading.Thread(target=lambda: (input(), stop.set()), daemon=True)
        waiter.start()
        stop.wait()
        timer.cancel()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def main():
    import sounddevice as sd

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--speaker", help="pseudonym, letters/digits only (no real names)")
    ap.add_argument("--condition", default="normal")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--device", type=int)
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    if a.list_devices:
        print(sd.query_devices())
        return
    cfg = load_config(a.config)
    if not a.speaker or not re.fullmatch(r"[a-z0-9]+", a.speaker):
        sys.exit("--speaker must be lowercase letters/digits (it is part of the filename; no underscores)")
    if a.condition not in cfg["own"]["conditions"]:
        sys.exit(f"--condition must be one of {cfg['own']['conditions']}")

    out = cfg.data / "own"
    out.mkdir(parents=True, exist_ok=True)
    consent_log = out / "consent.csv"
    seen = consent_log.exists() and any(r[0] == a.speaker for r in csv.reader(consent_log.open()))
    if not seen:
        print("\n" + CONSENT + "\n")
        if input("Type 'yes' to consent: ").strip().lower() != "yes":
            sys.exit("No consent, nothing recorded.")
        with consent_log.open("a", newline="") as f:
            csv.writer(f).writerow([a.speaker, time.strftime("%Y-%m-%d %H:%M"), CONSENT])

    items = [(pid, text) for pid, text in cfg.phrases.items()]
    if not a.no_negatives:
        items += [(f"neg_{k}", f"(say) {v}") for k, v in cfg.hard_negatives.items()]
        prompts = {"silence": "(stay silent)", "breathing": "(breathe audibly)", "cough": "(cough or clear your throat)",
                   "unrelated": "(say any unrelated sentence)"}
        items += [(f"neg_{k}", prompts.get(k, k)) for k in cfg["own"]["negatives"]]
    plan = [it for it in items for _ in range(a.takes)]
    random.shuffle(plan)

    print(f"\n{len(plan)} prompts · condition '{a.condition}' · device {a.device if a.device is not None else 'default'}")
    print("For each prompt: Enter = start, speak, Enter = stop. Then Enter = keep, r = redo, s = skip, q = quit.\n")
    i = 0
    while i < len(plan):
        pid, text = plan[i]
        cmd = input(f"[{i + 1}/{len(plan)}] {text!r}  — Enter to start (s skip, q quit): ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue
        print("   recording… Enter to stop")
        x = record_once(sd, a.device, a.max_seconds)
        print(f"   {len(x) / audio.SR:.1f} s, level {audio.rms_db(x):.0f} dBFS")
        if input("   keep? (Enter / r): ").strip().lower() == "r":
            continue
        take = 1
        while (out / f"{a.speaker}_{a.condition}_{pid}_{take}.wav").exists():
            take += 1
        audio.save(out / f"{a.speaker}_{a.condition}_{pid}_{take}.wav", x)
        i += 1
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
