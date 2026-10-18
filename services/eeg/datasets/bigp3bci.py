"""bigP3BCI (PhysioNet v1.0.0, CC BY 4.0) EDF+ parser -> canonical tables.

Everything is read from the files. Observed on Study F / F_03 (see docs/EEG_DATASET_AUDIT.md):
62 signals @256 Hz = 16 EEG + 36 per-character "presented" flags (label <char>_<row>_<col>) + 10 BCI2000 state channels
(StimulusType, SelectedTarget, SelectedRow, SelectedColumn, PhaseInSequence, StimulusBegin, StimulusCode, CurrentTarget,
FakeFeedback, DisplayResults). Flash = rising edge of StimulusBegin; a selection = one PhaseInSequence==2 segment.
Nothing is assumed from BNCI2014-008: no row/column decoding is used, each flash lists the characters it showed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from services.eeg.datasets.schema import AuditResult, Protocol

STATE = ["StimulusType", "SelectedTarget", "SelectedRow", "SelectedColumn", "PhaseInSequence", "StimulusBegin",
         "StimulusCode", "CurrentTarget", "FakeFeedback", "DisplayResults"]
KNOWN_PHASES = {0, 1, 2, 3}
FILE_RE = re.compile(r"(?P<p>[A-Z0-9]+_\d+)_SE(?P<s>\d{3})_(?:(?P<c>[A-Za-z][A-Za-z0-9]*)_)?(?P<t>Train|Test)(?P<i>\d+)\.edf$")
CHAR_RE = re.compile(r"^(?P<ch>.+)_(?P<r>\d)_(?P<c>\d)$")
ET_PREFIX = "ET"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_name(path: Path) -> dict:
    m = FILE_RE.search(path.name)
    if not m:
        raise ValueError(f"unrecognised bigP3BCI file name: {path.name}")
    d = m.groupdict()
    cond = d["c"] or ("CB" if d["t"] else "")
    return {"participant": d["p"], "session": int(d["s"]), "run_type": d["t"], "condition": cond, "run_index": int(d["i"])}


def _runs(x: np.ndarray, value: int) -> list[tuple[int, int]]:
    m = (x == value).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def read_recording(path: Path, root: Path, res: AuditResult, protocol_hint: dict | None = None) -> dict | None:
    """Parse one EDF+. Returns a dict with recording/run/selection/event rows, or None (with an ERROR) if unreadable."""
    import pyedflib

    rel = path.relative_to(root).as_posix()
    try:
        meta = parse_name(path)
        r = pyedflib.EdfReader(str(path))
    except Exception as e:  # corrupt / truncated / unexpected name
        res.error("edf_readable", rel, f"{type(e).__name__}: {e}")
        return None
    try:
        labels = r.getSignalLabels()
        fs = {float(r.getSampleFrequency(i)) for i in range(len(labels))}
        n = {int(x) for x in r.getNSamples()}
        hdr = r.getHeader()
        if len(fs) != 1 or len(n) != 1:
            res.error("uniform_sampling", rel, f"fs={fs} n_samples={n}")
            return None
        fs_hz, n_samples = int(fs.pop()), n.pop()
        eeg_idx = [i for i, l in enumerate(labels) if l.startswith("EEG_")]
        char_idx = [i for i, l in enumerate(labels) if CHAR_RE.match(l) and not l.startswith("EEG_") and i not in eeg_idx
                    and l not in STATE and not l.startswith(ET_PREFIX)]
        missing = [s for s in STATE if s not in labels]
        if missing:
            res.error("state_channels", rel, f"missing {missing}")
            return None
        S = {s: r.readSignal(labels.index(s)) for s in STATE}
        eeg = np.stack([r.readSignal(i) for i in eeg_idx])
        flags = np.stack([r.readSignal(i) for i in char_idx], 1)
    finally:
        r.close()

    chars = [labels[i] for i in char_idx]
    if not np.isfinite(eeg).all():
        res.error("eeg_finite", rel, "NaN/inf in EEG channels")
    als = re.search(r"ALS_(\d+)", hdr.get("patient_additional", "") + " " + hdr.get("patientname", ""))
    rec_id = "bigp3bci/" + rel.removeprefix("bigp3bci/")
    rec_id = f"bigp3bci/{meta['participant']}/SE{meta['session']:03d}/{meta['run_type']}/{meta['condition']}{meta['run_index']:02d}"
    run_id = rec_id
    recording = {
        "recording_id": rec_id, "dataset": "bigp3bci", "participant_id": meta["participant"], "session": meta["session"],
        "run_id": run_id, "path": rel, "sha256": sha256(path), "bytes": path.stat().st_size, "fs_hz": fs_hz,
        "n_samples": n_samples, "duration_s": round(n_samples / fs_hz, 3), "n_eeg_channels": len(eeg_idx),
        "eeg_channels": ";".join(labels[i] for i in eeg_idx),
        "has_eye_tracking": any(l.startswith(ET_PREFIX) for l in labels), "equipment": hdr.get("equipment", ""),
        "_sex": hdr.get("sex", ""), "_als_raw": hdr.get("patient_additional", ""), "_alsfrs": int(als.group(1)) if als else None,
        "_n_chars": len(chars), "_chars": chars,
    }

    sb = (S["StimulusBegin"] > 0).astype(np.int8)
    d = np.diff(sb, prepend=0)
    onsets = np.where(d == 1)[0]
    offsets = np.where(d == -1)[0]
    if len(offsets) < len(onsets):
        offsets = np.append(offsets, n_samples)
    unknown_state = {}
    for name, ok in (("StimulusType", {0, 1}), ("PhaseInSequence", KNOWN_PHASES)):
        bad = set(np.unique(S[name]).astype(int).tolist()) - ok
        if bad:
            unknown_state[name] = sorted(bad)
    if unknown_state:
        res.error("unknown_event_codes", rel, f"unexpected state values {unknown_state}")
    if not set(np.unique(flags).astype(int).tolist()) <= {0, 1}:
        res.error("unknown_event_codes", rel, "character flags not binary")

    selections, events, targets, labels_rows = [], [], [], []
    segs = _runs(S["PhaseInSequence"].astype(int), 2)
    for k, (a, b) in enumerate(segs):
        sel_id = f"{rec_id}#t{k:02d}"
        idx = np.where((onsets >= a) & (onsets < b))[0]
        on = onsets[idx]
        shown = flags[on].astype(int)
        cur = np.unique(S["CurrentTarget"][a:b].astype(int))
        cur = cur[cur > 0]
        if len(cur) != 1:
            res.error("target_label", sel_id, f"CurrentTarget takes values {cur.tolist()} inside the trial")
            tgt = None
        else:
            tgt = int(cur[0])
            if not 1 <= tgt <= len(chars):
                res.error("target_label", sel_id, f"CurrentTarget={tgt} outside 1..{len(chars)}")
                tgt = None
        stype = S["StimulusType"][on].astype(int)
        if tgt is not None:
            expect = shown[:, tgt - 1]
            if not (expect == stype).all():
                res.error("character_mapping", sel_id,
                          f"StimulusType disagrees with 'target shown' in {(expect != stype).sum()} flashes (target idx {tgt})")
        post = S["SelectedTarget"][b:min(b + 1024, n_samples)].astype(int)
        post = post[post > 0]
        online = int(post[0]) if len(post) else None
        per_char = shown.sum(0) if len(on) else np.zeros(len(chars), int)
        blocks = repetition_blocks(shown) if len(on) else []
        rep_of = np.repeat(np.arange(len(blocks)), [nf for nf, _ in blocks]) if blocks else []
        selections.append({
            "selection_id": sel_id, "run_id": run_id, "recording_id": rec_id, "dataset": "bigp3bci",
            "participant_id": meta["participant"], "session": meta["session"], "trial_index": k,
            "start_sample": int(a), "end_sample": int(b), "n_flashes": int(len(on)),
            "min_flashes_per_character": int(per_char.min()), "max_flashes_per_character": int(per_char.max()),
            "complete_repetitions": sum(1 for _, c in blocks if c),
            "flashes_in_partial_repetition": sum(nf for nf, c in blocks if not c),
            "repetition_rule": "disjoint_cover_blocks",
        })
        targets.append({"selection_id": sel_id, "target_index": tgt, "target_char": chars[tgt - 1].split("_")[0] if tgt else None,
                        "online_selected_index": online,
                        "online_selected_char": chars[online - 1].split("_")[0] if online and online <= len(chars) else None})
        for j, (o, off) in enumerate(zip(on, offsets[idx])):
            events.append({"selection_id": sel_id, "flash_index": j, "onset_sample": int(o), "onset_s": round(o / fs_hz, 4),
                           "duration_samples": int(off - o), "chars_shown": ",".join(str(c + 1) for c in np.where(shown[j])[0]),
                           "repetition_index": int(rep_of[j])})
            labels_rows.append({"selection_id": sel_id, "flash_index": j, "is_target_flash": int(stype[j])})
    # flashes outside any trial are suspicious (stimulus without a selection)
    inside = sum(1 for o in onsets if any(a <= o < b for a, b in segs))
    if inside != len(onsets):
        res.error("stimulus_outside_selection", rel, f"{len(onsets) - inside} flashes outside PhaseInSequence==2")
    return {"recording": recording, "meta": meta, "selections": selections, "events": events, "targets": targets,
            "flash_labels": labels_rows, "n_samples": n_samples, "fs": fs_hz}


def repetition_blocks(shown: np.ndarray) -> list[tuple[int, bool]]:
    """Split a selection's flashes into repetitions. Rule (derived from the data, see docs/EEG_DATASET_AUDIT.md):
    a repetition is a maximal run of consecutive flashes whose character sets are disjoint; it is *complete* when their
    union is every character. Returns [(n_flashes, complete)]."""
    blocks, cur, n = [], np.zeros(shown.shape[1], bool), 0
    for row in shown.astype(bool):
        if (cur & row).any():
            blocks.append((n, bool(cur.all())))
            cur, n = np.zeros_like(cur), 0
        cur |= row
        n += 1
    if n:
        blocks.append((n, bool(cur.all())))
    return blocks


def study_f_protocol(fs: int, soa: int, flash: int, max_flashes: int | None, n_chars: int) -> Protocol:
    return Protocol("bigP3BCI Study F (observed)", (6, 6) if n_chars == 36 else (0, n_chars), "checkerboard_subset", soa, flash,
                    fs, max_flashes, "grid inferred from 36 per-character channels (label <char>_<row>_<col>); "
                                       "the PhysioNet overview table lists 9x8 for Study F: UNRESOLVED")
