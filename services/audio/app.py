"""Residual-voice specialist: a weak-speech clip -> one of the fixed request phrases, `unknown`, or no speech.

NVIDIA Nemotron Speech Streaming en 0.6B transcribes the clip and scores every phrase with its own RNNT
loss (forced_llr), behind a Silero VAD gate. This is the configuration chosen in asr_eval/ (results/loso.md):
the decision comes from the same code and the thresholds frozen there (results/thresholds.json), so the
agent makes exactly the decisions the evaluation measured.

It proposes phrases, never free text: a fixed phrase list cannot hallucinate words, and the raw
forced_llr scores (not probabilities) are passed on. Nothing is inferred from silence or from a clip in
which the model heard no words (plan.md). The patient's confirmation stays the final say.
"""

from __future__ import annotations

import os
import sys
import time

from fastapi import File, Form, UploadFile

from shared.provenance import ROOT
from shared.schemas import Evidence, Hypothesis, Provenance
from shared.service import Readiness, decode_audio_mono, make_app

ASR_EVAL = ROOT / "asr_eval"
sys.path.insert(0, str(ASR_EVAL))

SR = 16000
MODEL = os.environ.get("UNSPOKEN_ASR_MODEL", "nemotron-streaming-0.6b")
METHOD = os.environ.get("UNSPOKEN_ASR_METHOD", "forced_llr")
# balanced (<=5% false accepts per negative type on calibration) is the live default: every candidate is
# still confirmed by the person; strict (<=1%) is the headline operating point
POINT = os.environ.get("UNSPOKEN_ASR_POINT", "balanced")
MIN_S = 0.25

app = make_app("unspoken-audio")


def _load():
    from harness.config import load_config
    from harness.live import Bench

    bench = Bench(load_config(ASR_EVAL / "configs" / "eval.yaml"), MODEL, METHOD, POINT)
    if not bench.th:
        raise RuntimeError(f"no frozen thresholds for {bench.th_key(MODEL, METHOD, POINT)} in asr_eval/results/thresholds.json")
    return bench


model = Readiness("audio", _load)


@app.on_event("startup")
def _startup():
    model.start()


@app.get("/health")
def health():
    out = {"service": "audio", **model.health(), "checkpoint": "nvidia/nemotron-speech-streaming-en-0.6b",
           "method": METHOD, "point": POINT, "vad": "Silero VAD"}
    if model.state == "ready":
        info = model.value.model.info()
        out.update(device=info.get("device"), params=info.get("params"), phrases=len(model.value.vocab["phrases20"]))
    return out


@app.post("/infer")
def infer(audio: UploadFile = File(...), session_id: str = Form("demo-session"), enhance: bool = Form(True)):
    # `enhance` is accepted for API compatibility; the evaluated pipeline scores the raw audio
    t0 = time.time()
    bench = model.get()
    wav = decode_audio_mono(audio.file.read(), SR)
    quality: dict = {"duration_s": round(len(wav) / SR, 2)}
    prov = Provenance(service="audio", checkpoint="nvidia/nemotron-speech-streaming-en-0.6b", device=bench.model.info().get("device", "cpu"),
                      preprocess=f"pyav->16k mono; Silero VAD gate; {METHOD} phrase scoring, {POINT} thresholds",
                      extra={"model": MODEL, "method": METHOD, "point": POINT})
    if len(wav) < MIN_S * SR:
        return _evidence(session_id, [], ["too_short"], "unknown", quality, prov, t0)

    r = bench.recognize(wav)
    quality.update(decision=r["decision"], reason=r["reason"], transcript=r["transcript"], margin=r["margin"],
                   top3=r["top3"], vad_speech_s=r["vad"].get(f"vad_speech_s@{bench.th['vad_t']}"), timings_ms=r["timings_ms"])
    if r["decision"] == "no_speech":
        return _evidence(session_id, [], ["no_speech_detected"], "unknown", quality, prov, t0)
    if r["reason"] in ("empty_transcript", "no_hypothesis"):
        # the model heard no words: forced scoring must not create them
        return _evidence(session_id, [], ["low_word_confidence"], "unknown", quality, prov, t0)

    hyps = [Hypothesis(text=p["text"], score=p["score"], score_kind=f"{METHOD} (Nemotron RNNT, per-token log-likelihood ratio), uncalibrated")
            for p in r["top3"]]
    if r["decision"] == "candidate":
        return _evidence(session_id, hyps, [], "proposed", quality, prov, t0)
    # unknown (low score, small margin, negation): offer the top 3 so the person chooses, never pick for them
    return _evidence(session_id, hyps, ["low_word_confidence"], "ambiguous", quality, prov, t0)


def _evidence(session_id, hyps, flags, status, quality, prov, t0) -> Evidence:
    prov.latency_ms = int((time.time() - t0) * 1000)
    return Evidence(session_id=session_id, modality="audio", source_mode="live", source_ref="browser-microphone-clip",
                    hypotheses=hyps, quality_flags=flags, quality=quality, provenance=prov, status=status)
