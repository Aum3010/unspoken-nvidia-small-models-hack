"""Residual-voice specialist: weak speech -> raw ASR and enhanced ASR, side by side.

Whisper (MLX, Metal) transcribes the untouched audio and an RNNoise-denoised,
loudness-normalised copy. Both are kept; disagreements are marked, never merged.
Words absent from every usable input are never inferred: if voice activity is
absent, no hypotheses are returned (Whisper otherwise hallucinates on silence).
"""

from __future__ import annotations

import base64
import difflib
import io
import os
import time

import numpy as np
import soundfile as sf
from fastapi import File, Form, UploadFile

from shared.schemas import Evidence, Hypothesis, Provenance
from shared.service import Readiness, decode_audio_mono, make_app

SR = 16000
WHISPER_REPO = os.environ.get("UNSPOKEN_WHISPER", "mlx-community/whisper-base-mlx")
LOW_WORD_PROB = 0.5

app = make_app("unspoken-audio")


def _load():
    import mlx_whisper

    # warm-up: downloads/compiles the model once
    mlx_whisper.transcribe(np.zeros(SR, np.float32), path_or_hf_repo=WHISPER_REPO, language="en")
    return mlx_whisper


model = Readiness("audio", _load)


@app.on_event("startup")
def _startup():
    model.start()


@app.get("/health")
def health():
    return {"service": "audio", **model.health(), "checkpoint": WHISPER_REPO, "enhancer": "RNNoise (pyrnnoise)", "device": "mlx-metal"}


def _rnnoise(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (denoised 16k audio, per-10ms speech probability)."""
    from pyrnnoise import RNNoise

    den = RNNoise(sample_rate=SR)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)[None, :]
    probs, chunks = [], []
    for p, frame in den.denoise_chunk(pcm, partial=True):
        probs.append(np.atleast_1d(p).mean())
        chunks.append(np.asarray(frame).reshape(-1))
    out = np.concatenate(chunks).astype(np.float32) / 32767 if chunks else audio.copy()
    return out[: len(audio)], np.asarray(probs, np.float32)


def _normalize(audio: np.ndarray, target_dbfs: float = -20.0) -> tuple[np.ndarray, float]:
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-9)
    gain = 10 ** (target_dbfs / 20) / rms
    gain = float(min(gain, 30.0))  # cap at ~+30 dB so we do not amplify pure noise
    return np.clip(audio * gain, -1, 1), 20 * np.log10(gain)


def _dbfs(x: np.ndarray) -> float:
    return round(float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)), 1)


def _transcribe(mlx_whisper, audio: np.ndarray) -> dict:
    r = mlx_whisper.transcribe(
        audio, path_or_hf_repo=WHISPER_REPO, language="en", word_timestamps=True,
        temperature=0.0, condition_on_previous_text=False,
    )
    words = [
        {"word": w["word"].strip(), "p": round(float(w["probability"]), 3), "start": round(w["start"], 2), "end": round(w["end"], 2)}
        for s in r.get("segments", []) for w in s.get("words", [])
    ]
    segs = r.get("segments", [])
    return {
        "text": r.get("text", "").strip(),
        "words": words,
        "avg_logprob": round(float(np.mean([s["avg_logprob"] for s in segs])), 3) if segs else None,
        "no_speech_prob": round(float(np.max([s["no_speech_prob"] for s in segs])), 3) if segs else None,
    }


def _norm_words(text: str) -> list[str]:
    return ["".join(c for c in w.lower() if c.isalnum() or c == "'") for w in text.split() if w.strip()]


def _diff(a: str, b: str) -> list[dict]:
    """Word-level alignment of raw vs enhanced transcripts."""
    wa, wb = _norm_words(a), _norm_words(b)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb).get_opcodes():
        out.append({"op": op, "raw": wa[i1:i2], "enhanced": wb[j1:j2]})
    return out


def _wav_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/infer")
def infer(audio: UploadFile = File(...), session_id: str = Form("demo-session"), enhance: bool = Form(True)):
    t0 = time.time()
    mw = model.get()
    wav = decode_audio_mono(audio.file.read(), SR)
    flags: list[str] = []
    quality: dict = {"duration_s": round(len(wav) / SR, 2), "input_dbfs": _dbfs(wav) if len(wav) else None}
    prov = Provenance(service="audio", checkpoint=WHISPER_REPO, device="mlx-metal",
                      preprocess="pyav->16k mono; enhanced = RNNoise + loudness normalisation (-20 dBFS, <=+30 dB)")

    if len(wav) < 0.4 * SR:
        flags.append("too_short")
        return _evidence(session_id, [], flags, quality, prov, t0)

    denoised, vad = _rnnoise(wav)
    speech_ratio = float((vad > 0.5).mean()) if len(vad) else 0.0
    quality["speech_ratio"] = round(speech_ratio, 3)
    if len(vad):
        frames = wav[: len(vad) * (len(wav) // max(1, len(vad)))].reshape(len(vad), -1)
        e = (frames**2).mean(1) + 1e-12
        sp, ns = e[vad > 0.5], e[vad <= 0.5]
        if len(sp) and len(ns):
            quality["snr_db_est"] = round(float(10 * np.log10(sp.mean() / ns.mean())), 1)
    if quality["input_dbfs"] is not None and quality["input_dbfs"] < -45:
        flags.append("low_signal")

    with model.lock:
        raw = _transcribe(mw, wav)
        enhanced_audio, gain_db = _normalize(denoised)
        enh = _transcribe(mw, enhanced_audio) if enhance else None
    quality["raw"] = raw
    quality["enhanced"] = enh
    quality["gain_db"] = round(gain_db, 1)
    quality["enhanced_audio"] = _wav_b64(enhanced_audio) if enhance else None

    no_voice = speech_ratio < 0.04 and max(raw["no_speech_prob"] or 1, 0) > 0.4
    if no_voice or (not raw["text"] and not (enh and enh["text"])):
        flags.append("no_speech_detected")
        return _evidence(session_id, [], flags, quality, prov, t0)

    hyps: list[Hypothesis] = []
    for label, r in (("raw", raw), ("enhanced", enh)):
        if r and r["text"] and r["text"] not in [h.text for h in hyps]:
            hyps.append(Hypothesis(text=r["text"], score=r["avg_logprob"] or -9.0,
                                   score_kind=f"whisper avg log-prob ({label} audio), uncalibrated",
                                   words=r["words"]))
    if enh:
        quality["diff"] = _diff(raw["text"], enh["text"])
        if any(d["op"] != "equal" for d in quality["diff"]):
            flags.append("enhancer_disagrees")
    if any(w["p"] < LOW_WORD_PROB for h in hyps for w in (h.words or [])):
        flags.append("low_word_confidence")
    return _evidence(session_id, hyps, flags, quality, prov, t0)


def _evidence(session_id, hyps, flags, quality, prov, t0) -> Evidence:
    prov.latency_ms = int((time.time() - t0) * 1000)
    status = "unknown" if not hyps else ("ambiguous" if {"enhancer_disagrees", "low_word_confidence"} & set(flags) else "proposed")
    return Evidence(session_id=session_id, modality="audio", source_mode="live", source_ref="browser-microphone-clip",
                    hypotheses=hyps, quality_flags=flags, quality=quality, provenance=prov, status=status)
