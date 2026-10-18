"""Study F (bigP3BCI) flash-level decoding -> frozen evidence artifacts.

Study-F specific facts (measured; see docs/experiments/EEG_DATASET_AUDIT.md): 36 characters, each flash lights 3-5 of them,
a repetition is a randomised partition of all 36 characters (10 flashes, then 8, alternating), 14 repetitions
(= 7 sequences of 18 flashes, Mainsah et al. 2015) = 126 flashes. Row/column decoding does not apply; evidence is
aggregated per character from the characters each flash lit (`stopping.py`).

Leakage rules enforced here:
  * preprocessing is a fixed causal band-pass (no fitted parameters);
  * the decoder is fitted on calibration-role recordings only (sessions 1-2); session 3 scores come from that frozen model;
  * score->LLR (class-conditional Gaussians) and, later, alpha/thresholds are fitted on out-of-fold calibration scores
    (train on one calibration session, score the other), never on session 3;
  * ground truth lives in `ground_truth.csv`, never in the evidence tables.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

FS = 256
BAND = (1.0, 20.0)
PRE, POST, DECIM = 26, 180, 4     # -0.1 s .. +0.7 s at 256 Hz; average blocks of 4 samples -> 64 Hz
N_CHARS = 36
CAL_ROLE, EVAL_ROLES = "calibration", ("evaluation", "evaluation_secondary")
DECODERS = ("xdawn_riemann_lr", "shrinkage_lda")
CONFIG = {"band_hz": BAND, "filter": "butter4 causal sosfilt", "epoch_s": [-0.1, 0.7], "baseline_s": [-0.1, 0.0],
          "decimation": DECIM, "decoders": DECODERS, "llr": "pooled-variance gaussian on out-of-fold calibration scores",
          "oof": "fit on one calibration session, score the other (sessions 1,2)"}


def load_eeg(path: Path) -> np.ndarray:
    import pyedflib

    r = pyedflib.EdfReader(str(path))
    try:
        idx = [i for i, l in enumerate(r.getSignalLabels()) if l.startswith("EEG_")]
        return np.stack([r.readSignal(i) for i in idx])
    finally:
        r.close()


def bandpass(x: np.ndarray) -> np.ndarray:
    sos = butter(4, BAND, btype="bandpass", fs=FS, output="sos")
    return sosfilt(sos, x, axis=-1)


def epoch(x: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """(channels, n) + onset samples -> (n_flash, channels, T) baseline-corrected, decimated. Raises if a window leaves the file."""
    out = []
    for o in onsets:
        if o - PRE < 0 or o + POST > x.shape[1]:
            raise ValueError(f"epoch window at sample {o} leaves the recording")
        seg = x[:, o - PRE:o + POST]
        seg = seg - seg[:, :PRE].mean(1, keepdims=True)
        post = seg[:, PRE:]
        T = post.shape[1] // DECIM * DECIM
        out.append(post[:, :T].reshape(post.shape[0], -1, DECIM).mean(-1))
    return np.stack(out).astype(np.float32)


def make_decoder(name: str):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    if name == "xdawn_riemann_lr":
        from services.eeg.p300 import _riemann  # existing pipeline: XdawnCovariances -> TangentSpace -> LogisticRegression

        return _riemann()
    if name == "shrinkage_lda":
        return make_pipeline(FunctionTransformer(lambda X: X.reshape(len(X), -1)),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    raise KeyError(name)


def score(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.decision_function(X), float)


class GaussianLLR:
    """log p(s|target) - log p(s|non-target), class-conditional Gaussians with pooled variance (linear in s)."""

    def fit(self, s: np.ndarray, y: np.ndarray) -> "GaussianLLR":
        s1, s0 = s[y == 1], s[y == 0]
        self.m1, self.m0 = float(s1.mean()), float(s0.mean())
        n1, n0 = len(s1), len(s0)
        self.var = float(((s1 - self.m1) ** 2).sum() + ((s0 - self.m0) ** 2).sum()) / max(n1 + n0 - 2, 1)
        return self

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return ((s - self.m0) ** 2 - (s - self.m1) ** 2) / (2 * self.var)

    def params(self) -> dict:
        return {"m_target": self.m1, "m_nontarget": self.m0, "var": self.var}


def parse_chars(cell: str) -> np.ndarray:
    v = np.zeros(N_CHARS, bool)
    v[[int(c) - 1 for c in cell.split(",")]] = True
    return v


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_participant(pid: str, tabs: dict, raw_root: Path, decoders=DECODERS, log=print):
    """Return (flash_rows, rep_blocks, calib_meta) for one participant. `tabs` holds the audit tables as DataFrames."""
    split = tabs["splits"].query("participant_id == @pid and role in ['calibration','evaluation','evaluation_secondary']")
    rec = tabs["recordings"].set_index("run_id")
    sel = tabs["selections"].query("participant_id == @pid").set_index("selection_id")
    ev = tabs["events"]
    lab = tabs["labels"].set_index(["selection_id", "flash_index"])["is_target_flash"]
    ev = ev[ev.selection_id.isin(sel.index)].copy()
    ev["run_id"] = sel.loc[ev.selection_id, "run_id"].values

    X, meta = [], []
    for run_id, role in zip(split.run_id, split.role):
        e = ev[ev.run_id == run_id]
        x = bandpass(load_eeg(raw_root / rec.loc[run_id, "path"]))
        X.append(epoch(x, e.onset_sample.values))
        m = e[["selection_id", "flash_index", "repetition_index", "onset_sample", "onset_s", "chars_shown"]].copy()
        m["role"], m["run_id"], m["session"] = role, run_id, int(sel.loc[e.selection_id.iloc[0], "session"])
        meta.append(m)
        log(f"  {run_id}: {len(e)} flashes")
    X, M = np.concatenate(X), pd.concat(meta, ignore_index=True)
    # labels: only ever indexed for calibration rows below
    cal = (M.role == CAL_ROLE).values
    y = np.full(len(M), -1)
    y[cal] = lab.loc[list(zip(M.selection_id[cal], M.flash_index[cal]))].values
    assert (y[~cal] == -1).all()

    flash_out, calib_meta = [], {}
    for dec in decoders:
        s = np.full(len(M), np.nan)
        oof = np.full(len(M), np.nan)
        for held in (1, 2):                      # out-of-fold calibration scores
            tr, te = cal & (M.session != held).values, cal & (M.session == held).values
            oof[te] = score(make_decoder(dec).fit(X[tr], y[tr]), X[te])
        final = make_decoder(dec).fit(X[cal], y[cal])
        ne = ~cal
        s[ne] = score(final, X[ne])
        s[cal] = oof[cal]                         # calibration rows carry their out-of-fold score
        g = GaussianLLR().fit(oof[cal], y[cal])
        calib_meta[dec] = {"llr": g.params(), "n_cal_flashes": int(cal.sum()), "n_cal_targets": int((y[cal] == 1).sum())}
        out = M[["selection_id", "flash_index", "repetition_index", "onset_sample", "onset_s", "chars_shown", "role", "session"]].copy()
        out["decoder"], out["score"], out["llr"] = dec, s, g(s)
        out["participant_id"] = pid
        flash_out.append(out)
    return pd.concat(flash_out, ignore_index=True), calib_meta


def rep_blocks(flash: pd.DataFrame, n_reps_max: int = 14):
    """flash table (one decoder) -> dict selection_id -> arrays. Increments, per-repetition end timing, availability."""
    res = {}
    for sid, g in flash.groupby("selection_id", sort=False):
        delta = np.full((n_reps_max, N_CHARS), np.nan, np.float32)
        last_on = np.full(n_reps_max, np.nan)
        n_fl = np.zeros(n_reps_max, int)
        for r, gr in g.groupby("repetition_index"):
            if r >= n_reps_max:
                continue
            shown = np.stack([parse_chars(c) for c in gr.chars_shown])
            if not shown.any(0).all():            # partial repetition: not every character lit -> not a usable unit
                continue
            delta[r] = (gr.llr.values[:, None] * shown).sum(0)
            last_on[r] = gr.onset_s.max()
            n_fl[r] = len(gr)
        res[sid] = {"delta": delta, "last_onset_s": last_on, "n_flashes": n_fl, "first_onset_s": float(g.onset_s.min())}
    return res
