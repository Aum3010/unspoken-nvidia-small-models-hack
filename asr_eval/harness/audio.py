"""Audio I/O, degradation effects and generated non-speech. CPU only."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfiltfilt

SR = 16000


def load(path, sr: int = SR, offset_s: float = 0.0, dur_s: float | None = None) -> np.ndarray:
    info = sf.info(str(path))
    start = int(offset_s * info.samplerate)
    frames = -1 if dur_s is None else int(dur_s * info.samplerate)
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True, start=start, frames=frames)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def save(path, x: np.ndarray, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1, 1), sr, subtype="PCM_16")


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12))


def at_db(x: np.ndarray, db: float) -> np.ndarray:
    return (x / (np.sqrt(np.mean(x**2)) + 1e-12) * 10 ** (db / 20)).astype(np.float32)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    return at_db(lfilter(b, a, rng.standard_normal(n)), 0)


# ---- degradations (one at a time, never stacked) ---------------------------------------------

def gain(x: np.ndarray, db: float) -> np.ndarray:
    return (x * 10 ** (db / 20)).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, noise: np.ndarray) -> np.ndarray:
    """Mix noise at an SNR measured over the whole clip (speech level includes its pauses)."""
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    return np.clip(x + at_db(noise, rms_db(x) - snr_db), -1, 1).astype(np.float32)


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    sos = butter(6, hz, btype="low", fs=SR, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def tempo(x: np.ndarray, rate: float) -> np.ndarray:
    import librosa

    return librosa.effects.time_stretch(x, rate=rate).astype(np.float32)


def reverb(x: np.ndarray, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Approximate room: exponentially decaying noise impulse response (60 dB decay at rt60)."""
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)
    rir[0] = 1.0
    y = fftconvolve(x, rir)[: len(x)]
    return at_db(y.astype(np.float32), rms_db(x))


# ---- generated non-speech --------------------------------------------------------------------------

def digital_silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), np.float32)


def mic_noise(seconds: float, level_db: float, rng: np.random.Generator) -> np.ndarray:
    """Low-level room tone: pink noise plus faint mains hum, like an idle laptop mic."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    hum = 0.2 * np.sin(2 * np.pi * rng.choice([50.0, 60.0]) * t)
    return at_db(pink(n, rng) + hum, level_db)
