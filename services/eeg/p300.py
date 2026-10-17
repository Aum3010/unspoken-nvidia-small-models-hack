"""P300 decoding on BNCI2014-008 (8 ALS participants, Riccio et al. 2013).

Recording facts (from the dataset description, verified against the files):
  256 Hz, 8 channels (Fz Cz Pz Oz P3 P4 PO7 PO8), 6x6 Farwell-Donchin matrix,
  rows/cols flashed 125 ms on / 125 ms off (4 Hz SOA), 10 repetitions x 12 flashes
  = 120 flashes per character, 35 characters (7 words x 5).
  y_stim 1-6 = Row1..Row6, 7-12 = Col1..Col6; y 2 = target flash.
  Runs 1-3 (characters 0-14) were the original calibration set; runs 4-7
  (characters 15-34) were online test runs. We keep exactly that split, so no
  repeated flash of a test character is ever seen during training.

The glyph layout (A-Z, 1-9, _) is the BCI2000 default; decoding the ground-truth
row/column pairs with it yields the Italian copy-spelling words of the protocol
(e.g. GATTO, VIOLA, REBUS), which is how the layout was checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from shared.provenance import ROOT

DATA_DIR = ROOT / "data" / "bnci2014_008"
CACHE_DIR = ROOT / "data" / "cache"
MATRIX = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_"
FS = 256
FLASHES_PER_CHAR = 120
REPS = 10
N_CALIB = 15
TMIN, TMAX = -0.1, 0.7
BAND = (1.0, 20.0)
CACHE_VERSION = 3


@dataclass
class Recording:
    subject: str
    eeg: np.ndarray        # (samples, 8) microvolts, band-passed
    onsets: np.ndarray     # (4200,) sample index of each flash onset
    codes: np.ndarray      # (4200,) 1..12
    is_target: np.ndarray  # (4200,) 0/1  -- ground truth, used only for training/scoring
    channels: list[str]
    meta: dict


def available_subjects() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("A0*.mat"))


def load(subject: str) -> Recording:
    d = sio.loadmat(DATA_DIR / f"{subject}.mat", squeeze_me=True, struct_as_record=False)["data"]
    ys, y = d.y_stim.astype(int), d.y.astype(int)
    onsets = np.where(np.diff((ys > 0).astype(int)) == 1)[0] + 1
    sos = butter(4, BAND, btype="bandpass", fs=FS, output="sos")
    eeg = sosfiltfilt(sos, d.X.astype(np.float64), axis=0)
    return Recording(
        subject=subject, eeg=eeg, onsets=onsets, codes=ys[onsets], is_target=(y[onsets] == 2).astype(int),
        channels=[str(c) for c in d.channels],
        meta={"sex": str(d.gender), "age": str(d.age), "alsfrs_r": str(d.ALSfrs), "onset": str(d.onsetALS)},
    )


def epochs(rec: Recording) -> np.ndarray:
    """(n_flashes, channels, samples) baseline-corrected epochs, TMIN..TMAX around each onset."""
    a, b, base = int(TMIN * FS), int(TMAX * FS), int(-TMIN * FS)
    X = np.stack([rec.eeg[o + a:o + b].T for o in rec.onsets])
    return X - X[:, :, :base].mean(axis=2, keepdims=True)


def char_index(i_flash: int) -> int:
    return i_flash // FLASHES_PER_CHAR


def truth_char(rec: Recording, ch: int) -> str:
    sl = slice(ch * FLASHES_PER_CHAR, (ch + 1) * FLASHES_PER_CHAR)
    codes = set(rec.codes[sl][rec.is_target[sl] == 1].tolist())
    row = next(c for c in codes if c <= 6) - 1
    col = next(c for c in codes if c > 6) - 7
    return MATRIX[row * 6 + col]


def _riemann():
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace

    return make_pipeline(XdawnCovariances(nfilter=4, estimator="lwf", xdawn_estimator="lwf"),
                         TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=2000))


class _DecimatedLDA:
    """Classic speller baseline: decimate epochs to ~20 Hz, concatenate channels, shrinkage LDA."""

    def __init__(self):
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    @staticmethod
    def _feat(X):
        n = X.shape[2] // 12 * 12
        return X[:, :, :n].reshape(X.shape[0], X.shape[1], -1, 12).mean(-1).reshape(X.shape[0], -1)

    def fit(self, X, y):
        self.lda.fit(self._feat(X), y)
        return self

    def decision_function(self, X):
        return self.lda.decision_function(self._feat(X))


def decode_char(scores: np.ndarray, codes: np.ndarray, reps: int = REPS) -> dict:
    """Accumulate classifier scores over the first `reps` repetitions -> row/col/letter."""
    n = reps * 12
    s, c = scores[:n], codes[:n]
    row_ev = np.array([s[c == r].sum() for r in range(1, 7)])
    col_ev = np.array([s[c == k].sum() for k in range(7, 13)])
    r, k = int(row_ev.argmax()), int(col_ev.argmax())
    srt_r, srt_c = np.sort(row_ev), np.sort(col_ev)
    return {
        "row": r, "col": k, "char": MATRIX[r * 6 + k],
        "row_evidence": row_ev.round(3).tolist(), "col_evidence": col_ev.round(3).tolist(),
        "margin": round(float(min(srt_r[-1] - srt_r[-2], srt_c[-1] - srt_c[-2])), 3),
    }


def train_and_evaluate(subject: str) -> dict:
    """Fit on calibration characters, score held-out characters. Cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"eeg_{subject}_v{CACHE_VERSION}.joblib"
    if path.exists():
        return joblib.load(path)

    rec = load(subject)
    X = epochs(rec)
    ch = np.arange(len(X)) // FLASHES_PER_CHAR
    train, test = ch < N_CALIB, ch >= N_CALIB

    results = {}
    for name, model in (("xdawn_riemann_lr", _riemann()), ("decimated_lda", _DecimatedLDA())):
        model.fit(X[train], rec.is_target[train])
        scores = model.decision_function(X)
        test_chars = sorted(set(ch[test].tolist()))
        curve = []
        for reps in range(1, REPS + 1):
            correct = sum(
                decode_char(scores[c * 120:(c + 1) * 120], rec.codes[c * 120:(c + 1) * 120], reps)["char"] == truth_char(rec, c)
                for c in test_chars
            )
            curve.append(round(correct / len(test_chars), 3))
        results[name] = {
            "model": model,
            "scores": scores.astype(np.float32),
            "test_auc": round(float(roc_auc_score(rec.is_target[test], scores[test])), 4),
            "accuracy_by_reps": curve,
        }

    # Grand-average ERPs on calibration data only (what the classifier learned from)
    tgt = X[train & (rec.is_target == 1)].mean(0)
    non = X[train & (rec.is_target == 0)].mean(0)
    times = (np.arange(X.shape[2]) / FS + TMIN).round(4)

    out = {
        "subject": subject,
        "meta": rec.meta,
        "channels": rec.channels,
        "n_train_chars": N_CALIB,
        "n_test_chars": int(35 - N_CALIB),
        "models": {k: {kk: vv for kk, vv in v.items() if kk not in ("model", "scores")} for k, v in results.items()},
        "scores": {k: v["scores"] for k, v in results.items()},
        "erp": {"times": times.tolist(), "target": tgt.round(3).tolist(), "nontarget": non.round(3).tolist()},
        "primary_model": "xdawn_riemann_lr",
    }
    joblib.dump(out, path)
    return out


def trial_payload(subject: str, char: int, model_name: str = "xdawn_riemann_lr") -> dict:
    """Everything the UI needs to replay one recorded character selection."""
    rec = load(subject)
    trained = train_and_evaluate(subject)
    scores = trained["scores"][model_name]
    sl = slice(char * FLASHES_PER_CHAR, (char + 1) * FLASHES_PER_CHAR)
    onsets = rec.onsets[sl]
    t0 = int(onsets[0])
    flashes = [
        {"i": i, "t": round((int(o) - t0) / FS, 4), "code": int(c), "score": round(float(s), 3)}
        for i, (o, c, s) in enumerate(zip(onsets, rec.codes[sl], scores[sl]))
    ]
    by_reps = [decode_char(scores[sl], rec.codes[sl], r) for r in range(1, REPS + 1)]

    # downsampled raw trace (64 Hz) for the scrolling signal view
    end = int(onsets[-1]) + int(TMAX * FS)
    trace = rec.eeg[t0:end:4].T
    return {
        "subject": subject,
        "char_index": char,
        "split": "calibration" if char < N_CALIB else "held_out_test",
        "model": model_name,
        "flashes": flashes,
        "decoded_by_reps": by_reps,
        "prediction": by_reps[-1],
        "trace": {"fs": FS / 4, "channels": rec.channels, "data": trace.round(2).tolist()},
        # revealed by the UI only after the replay finishes; never used for prediction
        "ground_truth": {"char": truth_char(rec, char)},
        "source_mode": "prerecorded",
        "source_ref": f"BNCI2014-008/{subject}.mat#char{char}",
    }


def summary(subject: str) -> dict:
    t = train_and_evaluate(subject)
    rec = load(subject)
    return {k: v for k, v in t.items() if k != "scores"} | {
        "chars": [{"index": c, "split": "calibration" if c < N_CALIB else "held_out_test"} for c in range(35)],
        "words_hidden": True,
        "n_flashes": int(len(rec.onsets)),
    }
