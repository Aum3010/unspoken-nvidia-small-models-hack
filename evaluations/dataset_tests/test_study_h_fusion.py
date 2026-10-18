"""Study H fusion helpers: EEG evidence is causal in decision time, gaze uses only samples up to t, filename parsing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark import study_h_fusion as sh
from services.eeg.datasets.bigp3bci import FILE_RE

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "evaluations/results/eeg_gaze_fusion"


def test_filename_with_digits_in_condition():
    m = FILE_RE.match("H_04_SE001_CBGaze01_Test02.edf")
    assert m and m["p"] == "H_04" and m["c"] == "CBGaze01" and m["t"] == "Test"
    assert FILE_RE.match("H_04_SE001_CB_Train01.edf")["c"] == "CB"


def test_eeg_track_counts_flash_only_after_response_window():
    f = 256.0
    onsets = np.array([0, 32, 64])                       # 125 ms SOA
    shown = np.zeros((3, 4), bool); shown[0, 0] = shown[1, 1] = shown[2, 0] = True
    llr = np.array([1.0, 2.0, 3.0])
    times = np.array([0.6, 0.7, 0.825, 0.95])
    L = sh.eeg_track(llr, shown, onsets, f, times)
    assert np.allclose(L[0], 0)                          # no window has closed before 0.7 s
    assert np.allclose(L[1], [1, 0, 0, 0])
    assert np.allclose(L[2], [1, 2, 0, 0])
    assert np.allclose(L[3], [4, 2, 0, 0])


def test_gaze_track_ignores_future_and_invalid_samples():
    f = 256
    n = 4 * f
    gaze = np.zeros((2, n)); gaze[:, f:] = 1.0           # gaze jumps after 1 s
    valid = np.ones(n, bool); valid[: f // 2] = False   # first 0.5 s invalid
    run = {"fs": f, "gaze": gaze, "valid": valid}
    s = {"onsets": np.array([0])}
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # identity map
    tr = sh.gaze_track(run, s, A, np.array([0.25, 0.9, 2.0]))
    assert np.isnan(tr[0]).all()                         # nothing valid yet
    assert np.allclose(tr[1], 0.0)                       # the later jump is not visible at 0.9 s
    assert 0.0 < tr[2, 0] < 1.0


def test_gaze_ll_prefers_nearest_key_and_is_flat_without_gaze():
    pos = np.array([[0, 0], [1, 0], [5, 5]], float)
    ll = sh.gaze_ll(np.array([[0.9, 0.1], [np.nan, np.nan]]), pos, 0.5)
    assert ll[0].argmax() == 1
    assert np.allclose(ll[1], 0.0)


@pytest.mark.skipif(not (RES / "chosen_params.json").exists(), reason="run study_h_fusion.py evaluate first")
def test_committed_params_cover_every_fold_and_stress_level():
    ch = json.loads((RES / "chosen_params.json").read_text())
    assert {(c["fold"], c["gaze_offset_sd"]) for c in ch} == {(k, s) for k in range(3) for s in sh.STRESS}
