"""Restart only vLLM, from the running vLLM's own command line and environment (/proc), with changes.

    python3 scripts/brev/serving/relaunch_vllm.py --set --max-num-seqs 64 --add --speculative-config '{...}' \
        --env CUDA_MPS_PIPE_DIRECTORY=/tmp/x --log .run/logs/vllm_serving.log --out /tmp/relaunch.json

The first relaunch saves the original command line and environment (the one start_app.sh launched) to
.run/serving_vllm_orig.json; later relaunches start from that saved copy, so changes never accumulate. The new
pid goes into .run/pids/vllm.pid, so `start_app.sh stop` stops it. Exit 0 once /v1/models answers (10 min max).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PIDFILE = ROOT / ".run" / "pids" / "vllm.pid"
ORIG = ROOT / ".run" / "serving_vllm_orig.json"
URL = "http://127.0.0.1:8001/v1/models"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def vllm_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-f", r"[v]llm serve|VLLM::EngineCore"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def ready() -> bool:
    try:
        return urllib.request.urlopen(URL, timeout=3).status == 200
    except Exception:
        return False


def stop_vllm(timeout: float = 90) -> None:
    """SIGTERM the server (it shuts its engine down), then SIGKILL anything left of vLLM."""
    pids = vllm_pids()
    if PIDFILE.exists():
        pids = [int(PIDFILE.read_text().strip())] + pids
    for p in pids[:1]:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < timeout and any(alive(p) for p in vllm_pids()):
        time.sleep(1)
    for p in vllm_pids():
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(3)  # the driver releases the memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=2, action="append", default=[], metavar=("FLAG", "VALUE"), help="replace or add FLAG VALUE")
    ap.add_argument("--add", nargs=2, action="append", default=[], metavar=("FLAG", "VALUE"), help="append FLAG VALUE")
    ap.add_argument("--env", action="append", default=[], help="K=V to set in the environment")
    ap.add_argument("--log", default=str(ROOT / ".run" / "logs" / "vllm_serving.log"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    if ORIG.exists():
        orig = json.loads(ORIG.read_text())
    else:
        pid = int(PIDFILE.read_text().strip())
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        env = dict(kv.split(b"=", 1) for kv in Path(f"/proc/{pid}/environ").read_bytes().rstrip(b"\0").split(b"\0") if b"=" in kv)
        orig = {"argv": [a.decode() for a in argv], "env": {k.decode(): v.decode() for k, v in env.items()}}
        ORIG.write_text(json.dumps(orig, indent=1))
    argv = list(orig["argv"])
    for flag, val in args.set:
        if flag in argv:
            argv[argv.index(flag) + 1] = val
        else:
            argv += [flag, val]
    for flag, val in args.add:
        argv += [flag, val]
    env = dict(orig["env"])
    for kv in args.env:
        k, v = kv.split("=", 1)
        env[k] = v
        if k == "CUDA_MPS_PIPE_DIRECTORY" or k == "CUDA_MPS_LOG_DIRECTORY":
            Path(v).mkdir(parents=True, exist_ok=True)

    stop_vllm()
    t0 = time.time()
    with open(args.log, "w") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(ROOT), start_new_session=True)
    PIDFILE.write_text(str(proc.pid))
    ok = False
    while time.time() - t0 < args.timeout:
        if proc.poll() is not None:
            break
        if ready():
            ok = True
            break
        time.sleep(2)
    res = {"ok": ok, "pid": proc.pid, "exit_code": proc.poll(), "ready_s": round(time.time() - t0, 1), "argv": argv,
           "env_changes": args.env, "log": args.log}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "argv"}), flush=True)
    print("argv:", " ".join(argv), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
