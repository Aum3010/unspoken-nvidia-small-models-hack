"""BNCI2014-008 adapter: reuses services/eeg/p300.load (unchanged) and emits the same canonical rows."""

from __future__ import annotations

from pathlib import Path

from services.eeg.datasets.schema import AuditResult


def rows_from_recording(rec, res: AuditResult, path: str = "", sha: str = "", nbytes: int = 0) -> dict:
    """rec is a services.eeg.p300.Recording. 35 characters x 120 flashes, 6x6 row/col matrix, 10 repetitions."""
    from services.eeg import p300

    sub = rec.subject
    n_flash, per = len(rec.onsets), p300.FLASHES_PER_CHAR
    if n_flash != 35 * per:
        res.error("bnci_flash_count", sub, f"{n_flash} flashes, expected {35 * per}")
    rec_id = f"bnci2014_008/{sub}"
    chars_per_code = {c: [r * 6 + k + 1 for r in range(6) for k in range(6) if (r + 1 == c if c <= 6 else k + 7 == c)]
                      for c in range(1, 13)}
    sel, ev, tgt, lab = [], [], [], []
    for ch in range(n_flash // per):
        sid = f"{rec_id}#c{ch:02d}"
        sl = slice(ch * per, (ch + 1) * per)
        try:
            tc = p300.truth_char(rec, ch)
        except StopIteration:
            res.error("missing_target_label", sid, "no target row/column flash found")
            tc = None
        sel.append({"selection_id": sid, "run_id": f"{rec_id}/run{(ch // 5) + 1}", "recording_id": rec_id, "dataset": "bnci2014_008",
                    "participant_id": sub, "session": 1, "trial_index": ch, "start_sample": int(rec.onsets[sl][0]),
                    "end_sample": int(rec.onsets[sl][-1]) + 1, "n_flashes": per, "max_flashes_in_protocol": per,
                    "evidence_truncated": False, "min_flashes_per_character": 10, "max_flashes_per_character": 10,
                    "grid_rows": 6, "grid_cols": 6, "complete_repetitions": 10, "flashes_in_partial_repetition": 0,
                    "repetition_rule": "12_flashes_per_repetition"})
        tgt.append({"selection_id": sid, "target_index": (p300.MATRIX.index(tc) + 1) if tc else None, "target_char": tc,
                    "online_selected_index": None, "online_selected_char": None})
        for j, (o, c, t) in enumerate(zip(rec.onsets[sl], rec.codes[sl], rec.is_target[sl])):
            ev.append({"selection_id": sid, "flash_index": j, "onset_sample": int(o), "onset_s": round(o / p300.FS, 4),
                       "duration_samples": 32, "chars_shown": ",".join(map(str, chars_per_code[int(c)])), "repetition_index": j // 12})
            lab.append({"selection_id": sid, "flash_index": j, "is_target_flash": int(t)})
    return {"selections": sel, "events": ev, "targets": tgt, "flash_labels": lab, "rec_id": rec_id,
            "recording": {"recording_id": rec_id, "dataset": "bnci2014_008", "participant_id": sub, "session": 1, "run_id": rec_id,
                          "path": path, "sha256": sha, "bytes": nbytes, "fs_hz": p300.FS, "n_samples": int(len(rec.eeg)),
                          "duration_s": round(len(rec.eeg) / p300.FS, 3), "n_eeg_channels": len(rec.channels),
                          "eeg_channels": ";".join(rec.channels), "has_eye_tracking": False, "equipment": ""}}


def audit_bnci_files(files: list[Path], res: AuditResult) -> dict:
    from services.eeg import p300
    from services.eeg.datasets.bigp3bci import sha256

    n = 0
    for f in files:
        out = rows_from_recording(p300.load(f.stem), res, f.as_posix(), sha256(f), f.stat().st_size)
        res.tables.setdefault("recordings", []).append(out["recording"])
        for key, table in (("selections", "selections"), ("events", "stimulus_events"), ("targets", "verifier_targets"),
                           ("flash_labels", "verifier_flash_labels")):
            res.tables.setdefault(table, []).extend(out[key])
        n += 1
    return {"present": True, "recordings": n}
