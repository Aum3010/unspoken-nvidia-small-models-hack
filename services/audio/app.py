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
from pathlib import Path
import re
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
EXTRA_PHRASES = Path(os.environ.get("UNSPOKEN_ASR_EXTRA_PHRASES", Path(__file__).with_name("extra_phrases.txt")))

app = make_app("unspoken-audio")


def _load():
    from harness.config import load_config
    from harness.live import Bench

    cfg = load_config(ASR_EVAL / "configs" / "eval.yaml")
    # extra phrases (services/audio/extra_phrases.txt) join the 20 evaluated ones for the live agent only
    for line in EXTRA_PHRASES.read_text().splitlines() if EXTRA_PHRASES.exists() else []:
        text = line.strip()
        if text and not text.startswith("#"):
            cfg.raw["phrases"]["x_" + "_".join(re.findall(r"[a-z0-9]+", text.lower()))[:40]] = text
    bench = Bench(cfg, MODEL, METHOD, POINT)
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
def infer(audio: UploadFile = File(...), session_id: str = Form("demo-session"), enhance: bool = Form(True),
          lips_moving: bool = Form(False)):
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

    # lips_moving: the browser saw the lips moving with this clip (sent by the gateway). A whisper has no voicing,
    # so the VAD reads it as no speech; with the lips moving, only the VAD gate is passed (score, margin and
    # negation gates still decide), so noise without mouth movement is still rejected.
    r = bench.recognize(wav, assume_speech=lips_moving)
    if r.get("vad_passed_by_lips"):
        quality["vad_passed_by_lips"] = True
    # one line per clip in .run/logs/audio.log: where a clip stops (no speech / low score / accepted) is visible
    vad_s = r["vad"].get(f"vad_speech_s@{bench.th['vad_t']}")
    top = [(p["text"], p["score"]) for p in r["top3"][:2]]
    print(f"[voice] {len(wav) / SR:.2f}s rms={float((wav ** 2).mean() ** 0.5):.4f} decision={r['decision']} "
          f"reason={r['reason']} vad_speech_s={vad_s} lips_moving={lips_moving} heard={r['transcript']!r} top={top}", flush=True)
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
        # accepted: it passed the score and margin gates, so the runners-up are not plausible readings;
        # one option keeps the question as small as possible (they stay in quality.top3 for the audit)
        return _evidence(session_id, hyps[:1], [], "proposed", quality, prov, t0)
    if r["reason"] == "small_margin":
        # a phrase scored well but another came close: offer the top 3 so the person chooses, never pick for them
        return _evidence(session_id, hyps, ["low_word_confidence"], "ambiguous", quality, prov, t0)
    # low_score (words, but none of the phrases: background talk, unrelated speech) or negation ("I don't need
    # water"): no reading. With always-on capture, offering the top 3 here turned any nearby talk into a guess.
    # The top 3 stay in quality.top3 for the audit.
    return _evidence(session_id, [], ["low_word_confidence"], "unknown", quality, prov, t0)


def _evidence(session_id, hyps, flags, status, quality, prov, t0) -> Evidence:
    prov.latency_ms = int((time.time() - t0) * 1000)
    return Evidence(session_id=session_id, modality="audio", source_mode="live", source_ref="browser-microphone-clip",
                    hypotheses=hyps, quality_flags=flags, quality=quality, provenance=prov, status=status)
