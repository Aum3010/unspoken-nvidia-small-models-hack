"""Personal enrollment: the user records their own phrases; new clips are matched against them.

Each recording is stored only as Auto-AVSR encoder features (no video) under
data/enroll/ on this machine. Matching is nearest-neighbour DTW over those
features (services/vsr/phrases.py). The recordings serve only this matching:
they are not uploaded and do not train any model.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from pathlib import Path

import numpy as np

from services.vsr.phrases import dtw_batch
from shared.provenance import ROOT

ENROLL_DIR = Path(os.environ.get("UNSPOKEN_ENROLL_DIR", ROOT / "data" / "enroll"))


def _slug(phrase: str) -> str:
    return hashlib.sha1(phrase.lower().encode()).hexdigest()[:12]


class EnrollmentStore:
    def __init__(self, root: Path = ENROLL_DIR):
        self.root = root
        self.lock = threading.Lock()
        self._cache: dict[str, list[np.ndarray]] | None = None
        self._packed: tuple[list[str], np.ndarray, np.ndarray] | None = None

    def _load(self) -> dict[str, list[np.ndarray]]:
        if self._cache is None:
            self._cache, self._packed = {}, None
            for d in sorted(self.root.glob("*/")) if self.root.exists() else []:
                name = d / "phrase.txt"
                if name.exists():
                    clips = [np.load(f).astype(np.float32) for f in sorted(d.glob("*.npy"))]
                    if clips:
                        self._cache[name.read_text().strip()] = clips
        return self._cache

    def phrases(self) -> list[dict]:
        with self.lock:
            return [{"phrase": p, "count": len(c)} for p, c in self._load().items()]

    def add(self, phrase: str, feats: np.ndarray) -> int:
        phrase = " ".join(phrase.split())
        with self.lock:
            d = self.root / _slug(phrase)
            d.mkdir(parents=True, exist_ok=True)
            (d / "phrase.txt").write_text(phrase)
            normed = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            n = len(list(d.glob("*.npy")))
            np.save(d / f"{n:03d}.npy", normed.astype(np.float16))
            self._cache = None
            return n + 1

    def remove(self, phrase: str):
        with self.lock:
            shutil.rmtree(self.root / _slug(" ".join(phrase.split())), ignore_errors=True)
            self._cache = None

    def _pack(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Every recording zero-padded into one array, so a clip is compared with all of them at once."""
        cache = self._load()
        if self._packed is None:
            names = [p for p, clips in cache.items() for _ in clips]
            clips = [c for cs in cache.values() for c in cs]
            lens = np.array([len(c) for c in clips])
            bs = np.zeros((len(clips), lens.max(initial=0), clips[0].shape[1] if clips else 0), np.float32)
            for k, c in enumerate(clips):
                bs[k, :len(c)] = c
            self._packed = names, bs, lens
        return self._packed

    def match(self, feats: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        """Phrases ranked by their closest recording (smaller DTW distance is better)."""
        x = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        with self.lock:
            names, bs, lens = self._pack()
            if not names:
                return []
            best: dict[str, float] = {}
            for p, dist in zip(names, dtw_batch(x, bs, lens).tolist()):
                best[p] = min(dist, best.get(p, np.inf))
        return sorted(best.items(), key=lambda t: t[1])[:top]
