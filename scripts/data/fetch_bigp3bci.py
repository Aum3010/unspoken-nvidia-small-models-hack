"""Fetch selected bigP3BCI recordings from the official PhysioNet S3 mirror, verified against SHA256SUMS.txt.

Usage: python scripts/data/fetch_bigp3bci.py --study F --participant F_03 [--dry-run]
Downloads only the requested participant into data/raw/bigp3bci/ (original EDF+ files, unchanged), resumes partial
files, verifies every checksum, and writes data/manifests/provenance_bigp3bci_<participant>.json.
For manually supplied EDFs: copy them under data/raw/bigp3bci/Study<S>/<P>/... and run with --verify-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIRROR = "https://physionet-open.s3.amazonaws.com/bigp3bci/1.0.0/"
PAGE = "https://physionet.org/content/bigp3bci/1.0.0/"
RAW = ROOT / "data" / "raw" / "bigp3bci"
MANIFESTS = ROOT / "data" / "manifests"


def _get(url: str, start: int = 0) -> urllib.request.addinfourl:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    return urllib.request.urlopen(req, timeout=60)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sums() -> dict[str, str]:
    cache = MANIFESTS / "bigp3bci_SHA256SUMS.txt"
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        cache.write_bytes(_get(MIRROR + "SHA256SUMS.txt").read())
    out = {}
    for line in cache.read_text().splitlines():
        digest, name = line.split(None, 1)
        out[name.strip()] = digest
    return out


def download(rel: str, expected: str) -> dict:
    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256(dest) == expected:
        return {"file": rel, "bytes": dest.stat().st_size, "sha256": expected, "status": "already_present"}
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    resp = _get(MIRROR + "bigP3BCI-data/" + rel, have)
    mode = "ab" if have and resp.status == 206 else "wb"
    with open(part, mode) as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    got = sha256(part)
    if got != expected:
        raise SystemExit(f"CHECKSUM MISMATCH {rel}: expected {expected}, got {got} (left as {part.name})")
    part.rename(dest)
    return {"file": rel, "bytes": dest.stat().st_size, "sha256": got, "status": "downloaded"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    prefix = f"Study{a.study}/{a.participant}/"
    sums = load_sums()
    files = sorted(n.removeprefix("bigP3BCI-data/") for n in sums if n.startswith("bigP3BCI-data/" + prefix))
    if not files:
        known = sorted({n.split("/")[2] for n in sums if f"/Study{a.study}/" in n})
        sys.exit(f"no files for {prefix}; participants present in Study {a.study}: {known}")
    if a.dry_run:
        print(f"{len(files)} files for {a.participant}")
        return
    rows = []
    for rel in files:
        exp = sums["bigP3BCI-data/" + rel]
        if a.verify_only:
            p = RAW / rel
            rows.append({"file": rel, "status": "ok" if p.exists() and sha256(p) == exp else "MISSING_OR_BAD"})
        else:
            rows.append(download(rel, exp))
        print(rows[-1]["status"], rel, flush=True)
    prov = {
        "dataset": "bigP3BCI", "version": "1.0.0", "doi": "10.13026/0byy-ry86", "license": "CC BY 4.0",
        "source_page": PAGE, "download_mirror": MIRROR, "checksums": "SHA256SUMS.txt from the same mirror",
        "acquired_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "study": a.study, "participant": a.participant, "files": rows,
    }
    (MANIFESTS / f"provenance_bigp3bci_{a.participant}.json").write_text(json.dumps(prov, indent=1))
    bad = [r for r in rows if r["status"] == "MISSING_OR_BAD"]
    if bad:
        sys.exit(f"{len(bad)} files missing or failing checksum")


if __name__ == "__main__":
    main()
