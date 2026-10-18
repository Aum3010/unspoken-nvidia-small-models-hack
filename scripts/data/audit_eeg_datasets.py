"""Audit BNCI2014-008 and bigP3BCI recordings already on disk and write canonical manifests.

    python scripts/data/audit_eeg_datasets.py                # audits whatever is present under data/raw
    python scripts/data/audit_eeg_datasets.py --out data/manifests

Reads only. Never downloads. Exits non-zero if any ERROR-level check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.datasets import bigp3bci, qa  # noqa: E402
from services.eeg.datasets.schema import AGENT_SAFE_TABLES, COLUMNS, VERIFIER_ONLY_TABLES, AuditResult  # noqa: E402

RAW_BIG = ROOT / "data" / "raw" / "bigp3bci"
BNCI_DIR = ROOT / "data" / "bnci2014_008"
CAL_SESSIONS, EVAL_SESSION = (1, 2), 3


def audit_bigp3bci(res: AuditResult) -> dict:
    stats: dict = {"files": 0}
    if not RAW_BIG.exists():
        res.info("bigp3bci_present", str(RAW_BIG), "no bigP3BCI files on disk")
        return stats
    chars_by_rec, rec_rows, hdr = {}, [], defaultdict(dict)
    run_rows, sel_rows, ev_rows, tgt_rows, lab_rows = [], [], [], [], []
    for p in sorted(RAW_BIG.glob("StudyF/**/*.edf")):     # the Study F benchmark audit; other studies have their own scripts
        out = bigp3bci.read_recording(p, RAW_BIG, res)
        stats["files"] += 1
        if out is None:
            continue
        rec = out["recording"]
        chars_by_rec[rec["recording_id"]] = rec.pop("_chars")
        hdr[rec["participant_id"]][rec["session"]] = (rec.pop("_sex"), rec.pop("_als_raw"), rec.pop("_alsfrs"))
        rec.pop("_n_chars")
        rec_rows.append(rec)
        m = out["meta"]
        run_rows.append({"run_id": rec["run_id"], "recording_id": rec["recording_id"], "dataset": "bigp3bci",
                         "participant_id": m["participant"], "session": m["session"], "run_type": m["run_type"],
                         "condition": m["condition"], "run_index": m["run_index"], "n_selections": len(out["selections"]),
                         "evidence_complete": not m["condition"].startswith("Dyn")})
        sel_rows += out["selections"]; ev_rows += out["events"]; tgt_rows += out["targets"]; lab_rows += out["flash_labels"]
    # protocol facts per condition, measured not assumed
    max_flashes = defaultdict(int)
    for s in sel_rows:
        max_flashes[s["run_id"]] = max(max_flashes[s["run_id"]], s["n_flashes"])
    fixed = {r["run_id"] for r in run_rows if r["evidence_complete"]}
    fixed_counts = Counter(s["n_flashes"] for s in sel_rows if s["run_id"] in fixed)
    protocol_max = fixed_counts.most_common(1)[0][0] if fixed_counts else None
    n_chars = Counter(len(v) for v in chars_by_rec.values())
    for s in sel_rows:
        s["grid_rows"], s["grid_cols"] = (6, 6) if n_chars and 36 in n_chars else (None, None)
        s["max_flashes_in_protocol"] = protocol_max
        s["evidence_truncated"] = s["n_flashes"] < (protocol_max or 10**9)
    gaps = Counter()
    durs = Counter()
    by_sel = defaultdict(list)
    for e in ev_rows:
        durs[e["duration_samples"]] += 1
        by_sel[e["selection_id"]].append(e["onset_sample"])
    for v in by_sel.values():
        gaps.update(b - a for a, b in zip(v, v[1:]))
    stats.update(protocol_max_flashes=protocol_max, flash_gap_samples=gaps.most_common(4), flash_duration_samples=durs.most_common(3),
                 n_char_channels=dict(n_chars))
    # participants / sessions
    part_rows, sess_rows = [], []
    for pid, by_s in sorted(hdr.items()):
        sexes = {v[0] for v in by_s.values()}
        scores = {s: v[2] for s, v in by_s.items()}
        raw = {v[1] for v in by_s.values()}
        part_rows.append({"participant_id": pid, "dataset": "bigp3bci", "sex": ";".join(sorted(sexes)),
                          "als_flag": any(r.startswith("ALS") for r in raw),
                          "alsfrs_r": next(iter({v for v in scores.values() if v is not None}), None) if len({v for v in scores.values() if v is not None}) == 1 else None,
                          "alsfrs_r_by_session": json.dumps(scores), "n_sessions": len(by_s)})
        if len({v for v in scores.values()}) > 1:
            res.warn("alsfrs_varies_by_session", pid, f"ALSFRS-R differs across sessions: {scores}")
        for s, v in sorted(by_s.items()):
            sess_rows.append({"session_id": f"bigp3bci/{pid}/SE{s:03d}", "participant_id": pid, "dataset": "bigp3bci", "session": s,
                              "n_runs": sum(1 for r in run_rows if r["participant_id"] == pid and r["session"] == s), "alsfrs_r": v[2]})
    res.tables.update(recordings=rec_rows, participants=part_rows, sessions=sess_rows, runs=run_rows, selections=sel_rows,
                      stimulus_events=ev_rows, verifier_targets=tgt_rows, verifier_flash_labels=lab_rows)
    return stats


def participant_stats(res: AuditResult) -> list[dict]:
    """One row per participant: sessions, runs/selections per condition, fixed vs dynamic trials, channels, fs, repetitions."""
    runs = {r["run_id"]: r for r in res.tables.get("runs", [])}
    recs = defaultdict(list)
    for r in res.tables.get("recordings", []):
        recs[r["participant_id"]].append(r)
    by_p = defaultdict(list)
    for s in res.tables.get("selections", []):
        by_p[s["participant_id"]].append(s)
    part = {p["participant_id"]: p for p in res.tables.get("participants", [])}
    rows = []
    for pid, ss in sorted(by_p.items()):
        cond = Counter(runs[s["run_id"]]["condition"] for s in ss)
        fixed = [s for s in ss if not runs[s["run_id"]]["condition"].startswith("Dyn")]
        dyn = [s for s in ss if runs[s["run_id"]]["condition"].startswith("Dyn")]
        rows.append({
            "participant_id": pid, "sessions": sorted({s["session"] for s in ss}).__len__(),
            "runs": len({s["run_id"] for s in ss}),
            "selections_by_condition": json.dumps(dict(sorted(cond.items()))),
            "n_fixed_length_selections": len(fixed), "n_dynamic_selections": len(dyn),
            "n_truncated_fixed_selections": sum(1 for s in fixed if s["evidence_truncated"]),
            "dyn_flashes_min_median_max": json.dumps([min(s["n_flashes"] for s in dyn), sorted(s["n_flashes"] for s in dyn)[len(dyn) // 2],
                                                      max(s["n_flashes"] for s in dyn)]),
            "eeg_channels": sorted({r["n_eeg_channels"] for r in recs[pid]})[0], "fs_hz": sorted({r["fs_hz"] for r in recs[pid]})[0],
            "has_eye_tracking": any(r["has_eye_tracking"] for r in recs[pid]),
            "total_flashes": sum(s["n_flashes"] for s in ss), "alsfrs_r": part[pid]["alsfrs_r"],
        })
    return rows


def split_manifest(res: AuditResult) -> list[dict]:
    """Session-level split per participant. Roles: calibration | evaluation | held_out_unused | dynamic_replay_only."""
    rows = []
    for r in res.tables.get("runs", []):
        if r["dataset"] != "bigp3bci":
            continue
        if not r["evidence_complete"]:
            role, why = "dynamic_replay_only", "online dynamic stopping truncated the evidence; usable only to compare with the recorded stop point"
        elif r["session"] in CAL_SESSIONS:
            role, why = "calibration", f"session {r['session']} complete fixed-length run"
        elif r["session"] == EVAL_SESSION and r["run_type"] == "Test":
            role, why = "evaluation", "session 3 complete fixed-length Test run (primary)"
        elif r["session"] == EVAL_SESSION:
            role, why = "evaluation_secondary", "session 3 complete calibration-style run; never used for training"
        else:
            role, why = "held_out_unused", "session outside the 1-2 / 3 scheme"
        rows.append({"participant_id": r["participant_id"], "session": r["session"], "run_id": r["run_id"], "run_type": r["run_type"],
                     "condition": r["condition"], "role": role, "reason": why})
    return rows


def audit_bnci(res: AuditResult) -> dict:
    files = sorted(BNCI_DIR.glob("A0*.mat")) if BNCI_DIR.exists() else []
    if not files:
        res.info("bnci_present", str(BNCI_DIR), "no BNCI2014-008 files on disk; BNCI audit not run (source host unreachable from the audit container)")
        return {"present": False}
    from services.eeg.datasets.bnci import audit_bnci_files

    return audit_bnci_files(files, res)


def write_tables(res: AuditResult, out: Path, split_rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in (*AGENT_SAFE_TABLES, *VERIFIER_ONLY_TABLES):
        rows = res.tables.get(name, [])
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS[name], extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    ps = participant_stats(res)
    if ps:
        with open(out / "participant_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ps[0])); w.writeheader(); w.writerows(ps)
    with open(out / "splits_session_3way.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant_id", "session", "run_id", "run_type", "condition", "role", "reason"])
        w.writeheader(); w.writerows(split_rows)
    (out / "audit_issues.json").write_text(json.dumps([i.__dict__ for i in res.issues], indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "manifests"))
    ap.add_argument("--skip-bnci", action="store_true", help="audit bigP3BCI only (the Study F/H pipeline does not use BNCI)")
    a = ap.parse_args()
    res = AuditResult()
    stats = {"bigp3bci": audit_bigp3bci(res), "bnci2014_008": {"present": False} if a.skip_bnci else audit_bnci(res)}
    qa.check_unique_ids(res); qa.check_timestamps(res); qa.check_targets(res); qa.check_sequences(res)
    splits = split_manifest(res)
    qa.check_split_manifest(res, splits)
    write_tables(res, Path(a.out), splits)
    (Path(a.out) / "audit_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    for i in res.issues:
        print(f"[{i.level}] {i.check} @ {i.where}: {i.detail}")
    print(json.dumps(stats, default=str))
    print(f"tables: " + ", ".join(f"{k}={len(v)}" for k, v in res.tables.items()))
    return 1 if res.errors else 0


if __name__ == "__main__":
    sys.exit(main())
