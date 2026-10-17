"""Weak-speech baseline: raw Whisper vs RNNoise+gain Whisper on the same utterances.

Utterances are synthetic (Kokoro voices) degraded to controlled levels/SNRs, so
the reference text is exact. This tests the pipeline's behaviour, not dysarthric
speech — real dysarthria needs its own consented evaluation.
"""

from __future__ import annotations

import io
import json
import statistics
from pathlib import Path

import httpx
import jiwer
import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluations" / "results"
AUDIO = "http://127.0.0.1:8102/infer"
SENTENCES = [
    "I need some water please", "Please call my daughter", "My back hurts when I lie down", "Can you turn off the light",
    "I would like to sit up", "Thank you for staying with me", "I feel too cold", "Where is the nurse",
    "Please move my pillow", "I want to go home",
]
VOICES = ["af_heart", "am_michael", "bf_emma", "am_fenrir", "af_bella"]
CONDITIONS = [
    ("Normal level, quiet room", -20, 40),
    ("Very quiet (-42 dBFS), quiet room", -42, 30),
    ("Normal level, noisy (SNR 5 dB)", -22, 5),
    ("Whisper-level + noise (SNR 0 dB)", -40, 0),
    ("Whisper-level + heavy noise (SNR -5 dB)", -40, -5),
    ("Whisper-level + very heavy noise (SNR -10 dB)", -40, -10),
]
rng = np.random.default_rng(3)


def pink(n: int) -> np.ndarray:
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    x = lfilter(b, a, rng.standard_normal(n))
    return x / (np.sqrt(np.mean(x**2)) + 1e-9)


def at_db(x: np.ndarray, db: float) -> np.ndarray:
    return x / (np.sqrt(np.mean(x**2)) + 1e-9) * 10 ** (db / 20)


def wav_bytes(x: np.ndarray, sr=16000) -> bytes:
    b = io.BytesIO()
    sf.write(b, np.clip(x, -1, 1), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def norm(t: str) -> str:
    return " ".join("".join(c for c in w.lower() if c.isalnum() or c == "'") for w in t.split())


def main():
    from mlx_audio.tts.utils import load_model

    tts = load_model("mlx-community/Kokoro-82M-bf16")
    clean = []
    for i, s in enumerate(SENTENCES):
        v = VOICES[i % len(VOICES)]
        a = np.concatenate([np.asarray(r.audio).reshape(-1) for r in tts.generate(text=s, voice=v, speed=1.0, lang_code=v[0])])
        clean.append(resample_poly(a, 2, 3).astype(np.float32))
    client = httpx.Client(timeout=120)
    conds, examples = [], []
    for name, level, snr in CONDITIONS:
        refs, raws, enhs = [], [], []
        for s, a in zip(SENTENCES, clean):
            pad = np.zeros(8000, np.float32)
            sp = at_db(np.concatenate([pad, a, pad]), level)
            x = sp + at_db(pink(len(sp)), level - snr)
            r = client.post(AUDIO, files={"audio": ("x.wav", wav_bytes(x), "audio/wav")}, data={"enhance": "true"}).json()
            q = r["quality"]
            raw = (q.get("raw") or {}).get("text", "") if r["hypotheses"] else ""
            enh = (q.get("enhanced") or {}).get("text", "") if r["hypotheses"] else ""
            refs.append(norm(s)); raws.append(norm(raw)); enhs.append(norm(enh))
            if len(examples) < 8 and norm(raw) != norm(enh):
                examples.append({"condition": name, "truth": s, "raw": raw, "enhanced": enh, "flags": r["quality_flags"]})
        ins_raw = jiwer.process_words(refs, [r or "∅" for r in raws]).insertions
        ins_enh = jiwer.process_words(refs, [e or "∅" for e in enhs]).insertions
        conds.append({"name": name, "level_dbfs": level, "snr_db": snr, "n": len(refs),
                      "wer_raw": round(jiwer.wer(refs, [r or "∅" for r in raws]), 4),
                      "wer_enhanced": round(jiwer.wer(refs, [e or "∅" for e in enhs]), 4),
                      "fabricated_raw": ins_raw, "fabricated_enh": ins_enh})
        print(conds[-1], flush=True)

    # noise only: any words at all are fabricated
    fab_raw = fab_enh = declined = 0
    for i in range(10):
        x = at_db(pink(16000 * 3), -45 + i)
        r = client.post(AUDIO, files={"audio": ("n.wav", wav_bytes(x), "audio/wav")}, data={"enhance": "true"}).json()
        q = r["quality"]
        if not r["hypotheses"]:
            declined += 1
        fab_raw += len(norm((q.get("raw") or {}).get("text", "")).split()) if r["hypotheses"] else 0
        fab_enh += len(norm((q.get("enhanced") or {}).get("text", "")).split()) if r["hypotheses"] else 0
    conds.append({"name": "Noise only, no speech (10 clips)", "level_dbfs": None, "snr_db": None, "n": 10,
                  "wer_raw": None, "wer_enhanced": None, "fabricated_raw": fab_raw, "fabricated_enh": fab_enh, "declined": declined})
    print(conds[-1])
    res = {"conditions": conds, "examples": examples,
           "note": "Synthetic Kokoro utterances (5 voices) degraded with pink noise; reference text exact. 'Fabricated' = inserted words "
                   "(noise-only: any word surfaced as a candidate). Noise-only clips the voice agent declined to transcribe: "
                   f"{declined}/10. Not a dysarthria benchmark."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
