"""Dataset-layer tests. Validator tests use hand-built invalid tables (to prove the checks fail loudly);
real-data tests are skipped, not faked, when the recordings are not on disk."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.datasets import bigp3bci, qa  # noqa: E402
from services.eeg.datasets.schema import AGENT_SAFE_TABLES, COLUMNS, VERIFIER_ONLY_TABLES, AuditResult  # noqa: E402

RAW = ROOT / "data" / "raw" / "bigp3bci"
HAVE_F = any(RAW.rglob("StudyF/**/*.edf")) if RAW.exists() else False


def _spec():
    p = ROOT / "scripts" / "data" / "audit_eeg_datasets.py"
    s = importlib.util.spec_from_file_location("audit_eeg_datasets", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- schema separation
def test_agent_safe_tables_never_contain_hidden_labels():
    hidden = {"target_index", "target_char", "online_selected_index", "online_selected_char", "is_target_flash"}
    for t in AGENT_SAFE_TABLES:
        assert not hidden & set(COLUMNS[t]), t
    assert set(VERIFIER_ONLY_TABLES) == {"verifier_targets", "verifier_flash_labels"}


# ---------------------------------------------------------------- validators fail loudly
def _res(**tables):
    r = AuditResult()
    r.tables.update(tables)
    return r


def test_duplicate_selection_ids_are_errors():
    r = _res(selections=[{"selection_id": "a"}, {"selection_id": "a"}])
    qa.check_unique_ids(r)
    assert any(i.check == "duplicate_ids" for i in r.errors)


def test_stimulus_outside_recording_and_selection_are_errors():
    r = _res(recordings=[{"recording_id": "r", "n_samples": 100}],
             selections=[{"selection_id": "s", "recording_id": "r", "start_sample": 10, "end_sample": 50}],
             stimulus_events=[{"selection_id": "s", "flash_index": 0, "onset_sample": 120, "duration_samples": 8}])
    qa.check_timestamps(r)
    checks = {i.check for i in r.errors}
    assert {"stimulus_outside_recording", "stimulus_outside_selection"} <= checks


def test_non_monotonic_and_zero_duration_events_are_errors():
    ev = [{"selection_id": "s", "flash_index": 0, "onset_sample": 20, "duration_samples": 8},
          {"selection_id": "s", "flash_index": 1, "onset_sample": 20, "duration_samples": 0}]
    r = _res(recordings=[{"recording_id": "r", "n_samples": 100}],
             selections=[{"selection_id": "s", "recording_id": "r", "start_sample": 10, "end_sample": 50}], stimulus_events=ev)
    qa.check_timestamps(r)
    assert sum(i.check == "invalid_timestamps" for i in r.errors) == 2


def test_missing_target_is_error():
    r = _res(verifier_targets=[{"selection_id": "s", "target_index": None, "target_char": None}])
    qa.check_targets(r)
    assert r.errors


def test_fixed_length_run_with_varying_flashes_is_error_but_dyn_is_not():
    runs = [{"run_id": "a", "condition": "Static"}, {"run_id": "b", "condition": "Dyn"}]
    sels = [{"run_id": "a", "n_flashes": 126}, {"run_id": "a", "n_flashes": 40},
            {"run_id": "b", "n_flashes": 20}, {"run_id": "b", "n_flashes": 30}]
    r = _res(runs=runs, selections=sels)
    qa.check_sequences(r)
    assert [i.where for i in r.errors] == ["a"]


def test_session_leakage_detected():
    rows = [{"role": "calibration", "participant_id": "p", "session": 1, "run_id": "r1"},
            {"role": "evaluation", "participant_id": "p", "session": 1, "run_id": "r2"}]
    r = _res(selections=[])
    qa.check_split_manifest(r, rows)
    assert any(i.check == "session_leakage" for i in r.errors)


def test_evidence_beyond_recorded_stop_raises():
    sel = {"selection_id": "s", "n_flashes": 20}
    qa.assert_evidence_within(sel, 19)
    with pytest.raises(IndexError):
        qa.assert_evidence_within(sel, 20)


def test_repetition_blocks_rule():
    # 6 characters; two complete repetitions of 2 flashes, then one flash of a third repetition
    shown = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1],
                      [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1],
                      [1, 1, 0, 0, 0, 0]])
    assert bigp3bci.repetition_blocks(shown) == [(2, True), (2, True), (1, False)]


# ---------------------------------------------------------------- BNCI adapter on a hand-built Recording (code path only)
def test_bnci_adapter_emits_6x6_rows_and_separates_targets():
    from services.eeg import p300
    from services.eeg.datasets.bnci import rows_from_recording

    rng = np.random.default_rng(0)
    n_char = 35
    codes, tgt = [], []
    for c in range(n_char):
        row, col = c % 6, (c * 5) % 6
        for _ in range(p300.REPS):
            order = rng.permutation(12) + 1
            codes += order.tolist()
            tgt += [int(x == row + 1 or x == 7 + col) for x in order]
    onsets = np.arange(len(codes)) * 64 + 100
    rec = p300.Recording("A99", np.zeros((int(onsets[-1]) + 300, 8)), onsets, np.array(codes), np.array(tgt), ["Fz"] * 8, {})
    res = AuditResult()
    out = rows_from_recording(rec, res)
    assert not res.errors and len(out["selections"]) == 35
    sel0 = out["selections"][0]
    assert sel0["grid_rows"] == 6 and sel0["n_flashes"] == 120 and "target_char" not in sel0
    assert all(len(e["chars_shown"].split(",")) == 6 for e in out["events"])


# ---------------------------------------------------------------- real Study F data (skipped if absent)
@pytest.fixture(scope="module")
def audit():
    mod = _spec()
    res = AuditResult()
    stats = mod.audit_bigp3bci(res)
    qa.check_unique_ids(res); qa.check_timestamps(res); qa.check_targets(res); qa.check_sequences(res)
    splits = mod.split_manifest(res)
    qa.check_split_manifest(res, splits)
    return res, stats, splits


@pytest.mark.skipif(not HAVE_F, reason="Study F EDF files not downloaded (scripts/data/fetch_bigp3bci.py)")
def test_study_f_audit_has_no_errors(audit):
    res, _, _ = audit
    assert not res.errors, [f"{i.check}@{i.where}: {i.detail}" for i in res.errors][:5]


@pytest.mark.skipif(not HAVE_F, reason="Study F EDF files not downloaded")
def test_study_f_protocol_is_measured_not_assumed(audit):
    res, stats, _ = audit
    assert stats["n_char_channels"] == {36: stats["files"]}   # 6x6 labels, not the 9x8 in the overview table
    assert stats["protocol_max_flashes"] == 126
    rec = res.tables["recordings"]
    assert {r["fs_hz"] for r in rec} == {256} and {r["n_eeg_channels"] for r in rec} == {16}
    assert not any(r["has_eye_tracking"] for r in rec)


@pytest.mark.skipif(not HAVE_F, reason="Study F EDF files not downloaded")
def test_study_f_static_runs_complete_dyn_runs_truncated(audit):
    res, _, _ = audit
    runs = {r["run_id"]: r for r in res.tables["runs"]}
    for s in res.tables["selections"]:
        cond = runs[s["run_id"]]["condition"]
        if cond in ("Static", "CBCol"):
            assert s["n_flashes"] == 126 and s["complete_repetitions"] == 14 and not s["evidence_truncated"]
        else:
            assert s["evidence_truncated"] and s["n_flashes"] < 126


@pytest.mark.skipif(not HAVE_F, reason="Study F EDF files not downloaded")
def test_study_f_split_manifest_is_session_clean(audit):
    res, _, splits = audit
    cal = {(r["participant_id"], r["session"]) for r in splits if r["role"] == "calibration"}
    ev = {(r["participant_id"], r["session"]) for r in splits if r["role"].startswith("evaluation")}
    assert cal and ev and not cal & ev
    assert all(r["session"] in (1, 2) for r in splits if r["role"] == "calibration")
    assert all(r["condition"].startswith("Dyn") for r in splits if r["role"] == "dynamic_replay_only")
