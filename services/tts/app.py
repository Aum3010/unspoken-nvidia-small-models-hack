"""Local TTS (Kokoro-82M: MLX on Apple Silicon, PyTorch `kokoro` on CUDA). An output renderer, never the authority on content.

Refuses any request that lacks a gateway-issued token for exactly this
(message_id, text) pair, so nothing but explicitly confirmed text is spoken.
"""

from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from shared.confirmation import verify_token
from shared.service import Readiness, make_app, pick_device

BACKEND = os.environ.get("UNSPOKEN_TTS_BACKEND", "mlx" if sys.platform == "darwin" else "kokoro")
REPO = os.environ.get("UNSPOKEN_TTS_MODEL", "mlx-community/Kokoro-82M-bf16" if BACKEND == "mlx" else "hexgrad/Kokoro-82M")
DEVICE = "mlx-metal" if BACKEND == "mlx" else pick_device("UNSPOKEN_TTS_DEVICE")
VOICES = {
    "af_heart": "Heart · warm (US)", "af_bella": "Bella · bright (US)", "am_michael": "Michael · calm (US)",
    "am_fenrir": "Fenrir · deep (US)", "bf_emma": "Emma · (UK)", "bm_george": "George · (UK)",
}
SR = 24000

app = make_app("unspoken-tts")


def _load():
    """Returns synth(text, voice, speed) -> iterable of float audio chunks at 24 kHz."""
    if BACKEND == "mlx":
        from mlx_audio.tts.utils import load_model

        m = load_model(REPO)

        def synth(text, voice, speed):
            return (r.audio for r in m.generate(text=text, voice=voice, speed=speed, lang_code=voice[0]))
    else:
        from kokoro import KModel, KPipeline

        km = KModel(repo_id=REPO).to(DEVICE).eval()
        # one G2P pipeline per accent ('a' US, 'b' UK), sharing the same weights
        pipes = {lang: KPipeline(lang_code=lang, repo_id=REPO, model=km) for lang in sorted({v[0] for v in VOICES})}

        def synth(text, voice, speed):
            return (r.audio.cpu().numpy() if hasattr(r.audio, "cpu") else r.audio for r in pipes[voice[0]](text, voice=voice, speed=speed))

    list(synth("Ready.", "af_heart", 1.0))  # warm-up
    return synth


model = Readiness("tts", _load)


@app.on_event("startup")
def _startup():
    model.start()


@app.get("/health")
def health():
    return {"service": "tts", **model.health(), "checkpoint": REPO, "device": DEVICE, "voices": VOICES}


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
    synth = model.get()
    with model.lock:
        chunks = [np.asarray(a).reshape(-1) for a in synth(req.text, voice, max(0.6, min(req.speed, 1.4)))]
    audio = np.concatenate(chunks) if chunks else np.zeros(SR // 10, np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Latency-Ms": str(int((time.time() - t0) * 1000)), "X-Voice": voice})
