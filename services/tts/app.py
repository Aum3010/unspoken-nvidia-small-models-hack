"""Local TTS (Kokoro-82M on MLX). An output renderer, never the authority on content.

Refuses any request that lacks a gateway-issued token for exactly this
(message_id, text) pair, so nothing but explicitly confirmed text is spoken.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from shared.confirmation import verify_token
from shared.service import Readiness, make_app

REPO = os.environ.get("UNSPOKEN_TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
VOICES = {
    "af_heart": "Heart · warm (US)", "af_bella": "Bella · bright (US)", "am_michael": "Michael · calm (US)",
    "am_fenrir": "Fenrir · deep (US)", "bf_emma": "Emma · (UK)", "bm_george": "George · (UK)",
}
SR = 24000

app = make_app("unspoken-tts")


def _load():
    from mlx_audio.tts.utils import load_model

    m = load_model(REPO)
    list(m.generate(text="Ready.", voice="af_heart", speed=1.0, lang_code="a"))  # warm-up
    return m


model = Readiness("tts", _load)


@app.on_event("startup")
def _startup():
    model.start()


@app.get("/health")
def health():
    return {"service": "tts", **model.health(), "checkpoint": REPO, "device": "mlx-metal", "voices": VOICES}


class SpeakRequest(BaseModel):
    message_id: str
    text: str
    token: str
    voice: str = "af_heart"
    speed: float = 1.0


@app.post("/speak")
def speak(req: SpeakRequest):
    if not verify_token(req.message_id, req.text, req.token):
        raise HTTPException(403, "speech not authorized: text was not explicitly confirmed")
    voice = req.voice if req.voice in VOICES else "af_heart"
    t0 = time.time()
    m = model.get()
    with model.lock:
        chunks = [np.asarray(r.audio).reshape(-1) for r in m.generate(
            text=req.text, voice=voice, speed=max(0.6, min(req.speed, 1.4)), lang_code=voice[0])]
    audio = np.concatenate(chunks) if chunks else np.zeros(SR // 10, np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Latency-Ms": str(int((time.time() - t0) * 1000)), "X-Voice": voice})
