"""Small helpers every specialist service uses: app factory, health state, media decoding."""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class Readiness:
    """Tracks a model that loads in a background thread so /health answers immediately."""

    def __init__(self, name: str, loader: Callable[[], object]):
        self.name = name
        self.state = "loading"
        self.error: str | None = None
        self.value: object | None = None
        self.loaded_s: float | None = None
        self._loader = loader
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._load, daemon=True, name=f"load-{self.name}").start()

    def _load(self) -> None:
        t = time.time()
        try:
            self.value = self._loader()
            self.state = "ready"
        except Exception as exc:  # surfaced in /health, never hidden
            logging.getLogger(self.name).exception("load failed")
            self.state = "unavailable"
            self.error = f"{type(exc).__name__}: {exc}"
        self.loaded_s = round(time.time() - t, 2)

    def get(self):
        if self.state != "ready":
            raise RuntimeError(f"{self.name} is {self.state}" + (f": {self.error}" if self.error else ""))
        return self.value

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def health(self) -> dict:
        return {"state": self.state, "error": self.error, "load_seconds": self.loaded_s}


def make_app(title: str) -> FastAPI:
    app = FastAPI(title=title)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    return app


def decode_video_rgb(data: bytes, target_fps: float = 25.0) -> tuple[np.ndarray, float]:
    """Decode any browser container (webm/mp4) to RGB frames resampled to a constant fps.

    MediaRecorder output has variable frame timing, so frames are picked by
    presentation timestamp rather than by index.
    """
    import av

    frames, times = [], []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            times.append(float(frame.pts * frame.time_base))
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        return np.zeros((0, 1, 1, 3), np.uint8), 0.0
    times_arr = np.asarray(times) - times[0]
    duration = float(times_arr[-1]) + 1.0 / target_fps
    grid = np.arange(0.0, duration, 1.0 / target_fps)
    idx = np.clip(np.searchsorted(times_arr, grid, side="right") - 1, 0, len(frames) - 1)
    return np.stack([frames[i] for i in idx]), duration


def decode_audio_mono(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode any browser audio container to mono float32 at target_sr (PyAV bundles ffmpeg)."""
    import av

    out = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                out.append(rf.to_ndarray().reshape(-1))
        for rf in resampler.resample(None):
            out.append(rf.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
