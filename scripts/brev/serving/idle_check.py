"""Seconds since the app was last used, from the gateway's logs (prints a number).

The gateway logs every call it makes to an agent (httpx, INFO). Use = a POST: a voice or lip clip, a plan, speech,
EEG. GETs are left out: the page polls /health and /enroll while it is open, even when nobody uses it.
Also left out: the windows when these experiments ran (.run/serving_self_windows.txt, "start end" epochs), and
logs saved before an experiment restarted the gateway (.run/logs/gateway.pre_*.log) are read too.
No use in the logs at all: the time since the gateway's first log line (it may have been used before it restarted).
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / ".run" / "logs"
WINDOWS = ROOT / ".run" / "serving_self_windows.txt"
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ httpx INFO HTTP Request: POST ")


def windows() -> list[tuple[float, float]]:
    out = []
    if WINDOWS.exists():
        for line in WINDOWS.read_text().split("\n"):
            parts = line.split()
            if parts:
                out.append((float(parts[0]), float(parts[1]) if len(parts) > 1 else time.time()))
    return out


def last_use() -> float | None:
    own = windows()
    last = first = None
    for f in [LOGS / "gateway.log", *LOGS.glob("gateway.pre_*.log")]:
        if not f.exists():
            continue
        for line in f.read_text(errors="replace").splitlines():
            if first is None and f.name == "gateway.log" and re.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,", line):
                first = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").timestamp()
            m = LINE.match(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()   # the box runs in UTC, like its logs
            if any(a - 5 <= t <= b + 5 for a, b in own):
                continue
            last = t if last is None or t > last else last
    return last if last is not None else first


if __name__ == "__main__":
    t = last_use()
    print(0 if t is None else int(time.time() - t))
    sys.exit(0)
