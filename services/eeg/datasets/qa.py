"""Fail-loud validators over the canonical tables. Each returns nothing and records ERROR/WARN/INFO on the AuditResult."""

from __future__ import annotations

from collections import Counter, defaultdict

from services.eeg.datasets.schema import AuditResult


def check_unique_ids(res: AuditResult) -> None:
    for table, key in (("recordings", "recording_id"), ("runs", "run_id"), ("selections", "selection_id")):
        dup = [k for k, n in Counter(r[key] for r in res.tables.get(table, [])).items() if n > 1]
        if dup:
            res.error("duplicate_ids", table, f"{len(dup)} duplicate {key}: {dup[:3]}")
    ev = Counter((e["selection_id"], e["flash_index"]) for e in res.tables.get("stimulus_events", []))
    if any(n > 1 for n in ev.values()):
        res.error("duplicate_ids", "stimulus_events", "duplicate (selection_id, flash_index)")


def check_timestamps(res: AuditResult) -> None:
    n_samples = {r["recording_id"]: r["n_samples"] for r in res.tables.get("recordings", [])}
    rec_of = {s["selection_id"]: s["recording_id"] for s in res.tables.get("selections", [])}
    sel = {s["selection_id"]: s for s in res.tables.get("selections", [])}
    last = {}
    for e in res.tables.get("stimulus_events", []):
        sid = e["selection_id"]
        n = n_samples.get(rec_of.get(sid))
        if n is None:
            res.error("orphan_event", sid, "stimulus event without recording")
            continue
        if not 0 <= e["onset_sample"] < n:
            res.error("stimulus_outside_recording", sid, f"onset {e['onset_sample']} not in [0,{n})")
        s = sel[sid]
        if not s["start_sample"] <= e["onset_sample"] < s["end_sample"]:
            res.error("stimulus_outside_selection", sid, f"onset {e['onset_sample']} outside {s['start_sample']}..{s['end_sample']}")
        if sid in last and e["onset_sample"] <= last[sid]:
            res.error("invalid_timestamps", sid, f"non-increasing onset at flash {e['flash_index']}")
        last[sid] = e["onset_sample"]
        if e["duration_samples"] <= 0:
            res.error("invalid_timestamps", sid, f"non-positive flash duration at flash {e['flash_index']}")
    for s in sel.values():
        n = n_samples.get(s["recording_id"])
        if n is not None and not 0 <= s["start_sample"] < s["end_sample"] <= n:
            res.error("invalid_timestamps", s["selection_id"], "selection bounds outside recording")


def check_targets(res: AuditResult) -> None:
    for t in res.tables.get("verifier_targets", []):
        if t["target_index"] is None or not t["target_char"]:
            res.error("missing_target_label", t["selection_id"], "no usable target label")


def check_sequences(res: AuditResult) -> None:
    """Flag selections whose evidence stops early (dynamic stopping) or is inconsistent with the run's protocol."""
    runs = {r["run_id"]: r for r in res.tables.get("runs", [])}
    full = defaultdict(list)
    for s in res.tables.get("selections", []):
        full[s["run_id"]].append(s["n_flashes"])
    for rid, ns in full.items():
        if len(set(ns)) > 1 and not runs[rid]["condition"].startswith("Dyn"):
            res.error("incomplete_sequence", rid, f"fixed-length run has varying flashes/selection {sorted(set(ns))}")


def check_split_manifest(res: AuditResult, split_rows: list[dict]) -> None:
    """Session leakage and selection-level separation between calibration and evaluation."""
    by_role = defaultdict(list)
    for r in split_rows:
        by_role[r["role"]].append(r)
    cal, ev = by_role.get("calibration", []), by_role.get("evaluation", [])
    leak_sessions = {(r["participant_id"], r["session"]) for r in cal} & {(r["participant_id"], r["session"]) for r in ev}
    if leak_sessions:
        res.error("session_leakage", "splits", f"sessions in both calibration and evaluation: {sorted(leak_sessions)}")
    leak_runs = {r["run_id"] for r in cal} & {r["run_id"] for r in ev}
    if leak_runs:
        res.error("session_leakage", "splits", f"runs in both roles: {sorted(leak_runs)[:3]}")
    sel_roles = defaultdict(set)
    for s in res.tables.get("selections", []):
        for r in split_rows:
            if r["run_id"] == s["run_id"]:
                sel_roles[s["selection_id"]].add(r["role"])
    both = [k for k, v in sel_roles.items() if {"calibration", "evaluation"} <= v]
    if both:
        res.error("session_leakage", "splits", f"{len(both)} selections assigned to both roles")


def assert_evidence_within(selection: dict, flash_index_requested: int) -> None:
    """Replay guard for future environments: evidence beyond the recorded stopping point does not exist."""
    if flash_index_requested >= selection["n_flashes"]:
        raise IndexError(f"{selection['selection_id']}: flash {flash_index_requested} was never recorded "
                         f"(recording stops after {selection['n_flashes']} flashes)")
