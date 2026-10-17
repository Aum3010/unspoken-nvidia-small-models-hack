"""Phase 2: Silero VAD speech gate.

Frame probabilities (512 samples = 32 ms) are computed once per clip; speech
duration at each threshold is the total length of runs of frames >= threshold
that last at least min_speech_ms. A low minimum keeps one-word "yes"/"no".
"""

from __future__ import annotations

import numpy as np

FRAME_S = 512 / 16000


class Vad:
    def __init__(self):
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self.model = load_silero_vad()

    def probs(self, wav: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            self.model.reset_states()
            return self.model.audio_forward(torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)), 16000)[0].numpy()


def speech_seconds(probs: np.ndarray, threshold: float, min_speech_ms: float) -> float:
    above = np.concatenate([[False], probs >= threshold, [False]])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    runs = edges[1::2] - edges[0::2]
    min_frames = max(1, int(np.ceil(min_speech_ms / 1000 / FRAME_S)))
    return float(runs[runs >= min_frames].sum() * FRAME_S)


def stats(probs: np.ndarray, thresholds: list[float], min_speech_ms: float) -> dict:
    out = {"vad_prob_max": round(float(probs.max()) if len(probs) else 0.0, 4)}
    for t in thresholds:
        out[f"vad_speech_s@{t}"] = round(speech_seconds(probs, t, min_speech_ms), 3)
    return out
