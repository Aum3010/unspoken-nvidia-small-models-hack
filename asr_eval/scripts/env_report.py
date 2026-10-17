"""Phase 0: log the machine, drivers and package versions to results/env.md."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGES = ["torch", "torchaudio", "transformers", "nemo_toolkit", "faster-whisper", "ctranslate2", "silero-vad",
            "librosa", "soundfile", "rapidfuzz", "pronouncing", "jiwer", "pandas", "numpy", "scipy", "kokoro", "sherpa-onnx"]


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def main():
    from harness.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    cfg = load_config(ap.parse_args().config)
    out = cfg.results / "env.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Environment", "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", ""]
    lines += ["## Machine", "", f"- Host OS: {platform.platform()}", f"- Python: {sys.version.split()[0]} ({sys.executable})",
              f"- CPU: {platform.processor() or 'n/a'} · {os.cpu_count()} logical cores"]
    try:
        import psutil

        lines.append(f"- RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    except ImportError:
        pass
    du = shutil.disk_usage(cfg.data if cfg.data.exists() else ROOT)
    lines.append(f"- Disk free at data dir: {du.free / 1e9:.0f} GB of {du.total / 1e9:.0f} GB")
    lines.append(f"- Git commit: {sh(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'])}")
    lines.append(f"- HF_HOME: {os.environ.get('HF_HOME', '(default)')} · HF_TOKEN set: {'yes' if os.environ.get('HF_TOKEN') else 'no'}")

    lines += ["", "## GPU", ""]
    try:
        import torch

        lines.append(f"- torch {torch.__version__}, CUDA runtime {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                lines.append(f"- cuda:{i} {p.name} · {p.total_memory / 1e9:.1f} GB · compute capability {p.major}.{p.minor}")
        else:
            lines.append("- No CUDA device visible to torch (CPU run)")
    except ImportError:
        lines.append("- torch not installed")
    smi = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    lines += [f"- nvidia-smi: {smi or 'n/a'}", "", "```", sh(["nvidia-smi"]) or "nvidia-smi not available", "```"]

    lines += ["", "## Packages", "", "| Package | Version |", "|---|---|"]
    for p in PACKAGES:
        try:
            lines.append(f"| {p} | {md.version(p)} |")
        except md.PackageNotFoundError:
            lines.append(f"| {p} | not installed |")
    lock = ROOT / "requirements.lock"
    lines += ["", f"Full pin list: `requirements.lock` ({'present' if lock.exists() else 'not generated yet — run setup_brev.sh'})."]

    prov = cfg.data / "provenance.json"
    if prov.exists():
        lines += ["", "## Datasets", "", "| Dataset | Source | Licence | Notes |", "|---|---|---|---|"]
        for name, d in json.loads(prov.read_text()).items():
            lines.append(f"| {name} | {d.get('source', '')} | {d.get('license', '')} | {d.get('notes', '')} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
