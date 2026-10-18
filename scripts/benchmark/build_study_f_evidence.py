"""Build frozen Study F evidence artifacts from the audited recordings.

    python scripts/benchmark/build_study_f_evidence.py [--participants F_03 F_05 ...] [--audit-dir data/processed/audit]

Reads data/raw/bigp3bci (EDF) + the audit tables; writes data/evidence/study_f/:
    flash_evidence.csv.gz      agent-visible per-flash scores/LLRs (no labels)
    rep_evidence_<decoder>.npz agent-visible per-repetition character log-evidence increments + timestamps
    ground_truth.csv           verifier-only target per selection
    manifest.json              config, per-participant calibration parameters, sha256 of every input and output
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import evidence as ev  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default=str(ROOT / "data/processed/audit"))
    ap.add_argument("--raw", default=str(ROOT / "data/raw/bigp3bci"))
    ap.add_argument("--out", default=str(ROOT / "data/evidence/study_f"))
    ap.add_argument("--participants", nargs="*")
    ap.add_argument("--decoders", nargs="*", default=list(ev.DECODERS))
    a = ap.parse_args()
    ad, out = Path(a.audit_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tabs = {"splits": pd.read_csv(ad / "splits_session_3way.csv"), "recordings": pd.read_csv(ad / "recordings.csv"),
            "selections": pd.read_csv(ad / "selections.csv"), "events": pd.read_csv(ad / "stimulus_events.csv"),
            "labels": pd.read_csv(ad / "verifier_flash_labels.csv")}
    pids = a.participants or sorted(tabs["recordings"].participant_id.unique())
    flash, calib = [], {}
    for pid in pids:
        print(pid, flush=True)
        f, c = ev.build_participant(pid, tabs, Path(a.raw), a.decoders)
        flash.append(f); calib[pid] = c
    flash = pd.concat(flash, ignore_index=True)
    flash.to_csv(out / "flash_evidence.csv.gz", index=False, float_format="%.6g")
    for dec in a.decoders:
        blocks = ev.rep_blocks(flash[flash.decoder == dec])
        ids = sorted(blocks)
        np.savez_compressed(out / f"rep_evidence_{dec}.npz", selection_id=np.array(ids),
                            delta=np.stack([blocks[i]["delta"] for i in ids]),
                            last_onset_s=np.stack([blocks[i]["last_onset_s"] for i in ids]),
                            n_flashes=np.stack([blocks[i]["n_flashes"] for i in ids]),
                            first_onset_s=np.array([blocks[i]["first_onset_s"] for i in ids]))
    gt = pd.read_csv(ad / "verifier_targets.csv")
    gt = gt[gt.selection_id.isin(flash.selection_id.unique())]
    gt.to_csv(out / "ground_truth.csv", index=False)
    roles = flash[["selection_id", "participant_id", "session", "role"]].drop_duplicates("selection_id")
    roles.to_csv(out / "selection_roles.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    manifest = {"config": ev.CONFIG, "git_commit": commit, "calibration": calib,
                "source_recordings": {r.run_id: r.sha256 for r in tabs["recordings"].itertuples() if r.participant_id in pids},
                "artifacts": {p.name: ev.sha(p) for p in sorted(out.glob("*")) if p.name != "manifest.json"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
