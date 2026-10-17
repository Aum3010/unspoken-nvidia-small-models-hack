"""Phase 1: download public datasets into data/raw/ and record provenance in data/provenance.json.

    python scripts/fetch_data.py                     # all
    python scripts/fetch_data.py librispeech esc50   # some

OpenSLR archives come from the ELDA mirror first (the main server measured ~0.3 MB/s
from Brev vs ~10 MB/s for ELDA), with the main server as fallback.
Archives are stream-extracted with a member filter, so only the parts we use hit
the disk (e.g. MUSAN noise/, Speech Commands _background_noise_). Each dataset's
landing page is saved next to it and lines mentioning its licence are logged, as
evidence of what the terms said on the day of download. Read them before use.
"""

from __future__ import annotations

import json
import re
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.config import load_config  # noqa: E402

DATASETS = {
    "librispeech": {
        "landing": "https://www.openslr.org/12/",
        "urls": [["https://openslr.elda.org/resources/12/test-clean.tar.gz", "https://www.openslr.org/resources/12/test-clean.tar.gz"]],
        "keep": lambda n: True,
        "license": "CC BY 4.0", "use": "unknown_speech negatives + WER sanity",
    },
    "musan": {
        "landing": "https://www.openslr.org/17/",
        "urls": [["https://openslr.elda.org/resources/17/musan.tar.gz", "https://www.openslr.org/resources/17/musan.tar.gz"]],
        "keep": lambda n: "/noise/" in n,
        "license": "CC BY 4.0 (per OpenSLR page; individual files carry their own attributions)", "use": "non_speech negatives + additive noise",
    },
    "speech_commands": {
        "landing": "https://www.tensorflow.org/datasets/catalog/speech_commands",
        "urls": ["http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"],
        "keep": lambda n: "_background_noise_" in n,
        "license": "CC BY 4.0", "use": "non_speech negatives (_background_noise_ only)",
    },
    "esc50": {
        "landing": "https://github.com/karolpiczak/ESC-50",
        "urls": ["https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"],
        "keep": lambda n: n.endswith(".wav") or n.endswith("esc50.csv"),
        "license": "CC BY-NC 3.0", "use": "non_speech negatives (selected categories)",
    },
    "torgo": {
        "landing": "https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html",
        "urls": [f"https://www.cs.toronto.edu/~complingweb/data/TORGO/{s}.tar.bz2" for s in ("F", "FC", "M", "MC")],
        "keep": lambda n: any(k in n for k in ("/wav_arrayMic/", "/wav_headMic/", "/prompts/")),
        "license": "Academic / non-profit research use; cite Rudzicz, Namasivayam & Wolff (2012), LRE 46(4)",
        "use": "dysarthric surrogate closed-vocab task + unknown-speech sentences",
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def licence_evidence(url: str, dest: Path) -> list[str]:
    try:
        html = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read().decode("utf-8", "ignore")
    except Exception as exc:
        return [f"landing page not fetched: {type(exc).__name__}: {exc}"]
    (dest / "landing.html").write_text(html, encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", html)
    hits = [" ".join(s.split())[:240] for s in re.split(r"(?<=[.!?])\s", text)
            if re.search(r"licen[cs]e|creative commons|cc[- ]by|academic|non-?commercial|copyright|cite", s, re.I)]
    return hits[:8]


def stream_extract(url: str, dest: Path, keep) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    n = 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        if url.endswith(".zip"):
            tmp = dest / "_download.zip"
            with open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(tmp) as z:
                for m in z.namelist():
                    if keep(m) and not m.endswith("/"):
                        z.extract(m, dest)
                        n += 1
            tmp.unlink()
            return n
        mode = "r|bz2" if url.endswith(".bz2") else "r|gz"
        with tarfile.open(fileobj=resp, mode=mode) as tar:
            for m in tar:
                if m.isfile() and keep("/" + m.name):
                    tar.extract(m, dest, filter="data")
                    n += 1
    return n


def fetch(name: str, data: Path, prov: dict, lock: threading.Lock) -> None:
    d = DATASETS[name]
    dest = data / "raw" / name
    dest.mkdir(parents=True, exist_ok=True)
    done = dest / ".done"
    evidence = licence_evidence(d["landing"], dest)
    entry = {"source": d["landing"], "urls": d["urls"], "license": d["license"], "use": d["use"],
             "licence_evidence": evidence, "notes": ""}
    if done.exists():
        entry["notes"] = "already downloaded"
    else:
        try:
            total = 0
            for url in d["urls"]:
                mirrors = url if isinstance(url, list) else [url]  # official mirrors, fastest first
                for i, u in enumerate(mirrors):
                    try:
                        log(f"{name}: downloading {u}")
                        total += stream_extract(u, dest, d["keep"])
                        entry["downloaded_from"] = entry.get("downloaded_from", []) + [u]
                        break
                    except Exception as exc:
                        if i == len(mirrors) - 1:
                            raise
                        log(f"{name}: {u} failed ({type(exc).__name__}), trying next mirror")
            done.write_text(time.strftime("%Y-%m-%d %H:%M"))
            entry["notes"] = f"{total} files extracted {time.strftime('%Y-%m-%d')}"
            log(f"{name}: {total} files")
        except Exception as exc:
            entry["notes"] = f"FAILED: {type(exc).__name__}: {exc}"
            log(f"{name}: FAILED {exc}" + (" — try a Hugging Face mirror; record its repo id and licence" if name == "torgo" else ""))
    with lock:
        prov[name] = entry


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(DATASETS)
    cfg_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--config=")), "configs/eval.yaml")
    cfg = load_config(cfg_arg)
    path = cfg.data / "provenance.json"
    prov = json.loads(path.read_text()) if path.exists() else {}
    lock = threading.Lock()
    threads = [threading.Thread(target=fetch, args=(n, cfg.data, prov, lock)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2))
    for n in names:
        log(f"{n}: {prov[n]['notes']}")
        for line in prov[n]["licence_evidence"][:3]:
            log(f"    licence text: {line}")


if __name__ == "__main__":
    main()
