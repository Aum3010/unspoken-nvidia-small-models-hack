"""Phase 1.5: synthetic positives + hard negatives with Kokoro, every English voice. SANITY ONLY.

    python scripts/make_tts.py                 # Kokoro (Linux/Brev)
    python scripts/make_tts.py --engine sapi   # Windows built-in voices, for laptop plumbing tests only

Writes data/tts/{voice}/{id}.wav (16 kHz). Rows get source=tts and never enter headline tables.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import audio  # noqa: E402
from harness.config import load_config  # noqa: E402


def kokoro_voices() -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files("hexgrad/Kokoro-82M")
    return sorted(m.group(1) for f in files if (m := re.fullmatch(r"voices/([ab][fm]_\w+)\.pt", f)))


def run_kokoro(texts: dict[str, str], out: Path) -> None:
    from kokoro import KPipeline
    from scipy.signal import resample_poly

    pipes = {}
    for voice in kokoro_voices():
        lang = voice[0]
        pipes.setdefault(lang, KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M"))
        for tid, text in texts.items():
            dst = out / voice / f"{tid}.wav"
            if dst.exists():
                continue
            chunks = [np.asarray(r.audio).reshape(-1) for r in pipes[lang](text, voice=voice, speed=1.0)]
            y = resample_poly(np.concatenate(chunks), 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz
            audio.save(dst, np.concatenate([np.zeros(4000, np.float32), y, np.zeros(4000, np.float32)]))
        print("  voice", voice, flush=True)


def run_sapi(texts: dict[str, str], out: Path) -> None:
    lines = ["Add-Type -AssemblyName System.Speech",
             "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)",
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
             "foreach ($v in $s.GetInstalledVoices()) { if ($v.VoiceInfo.Culture.Name -like 'en-*') {",
             "  $name = 'sapi' + ($v.VoiceInfo.Name -replace 'Microsoft ','' -replace ' Desktop','' -replace '[^A-Za-z]','').ToLower()",
             "  $s.SelectVoice($v.VoiceInfo.Name)"]
    for tid, text in texts.items():
        lines.append(f"  $d = Join-Path '{out}' $name; New-Item -ItemType Directory -Force $d | Out-Null")
        lines.append(f"  $s.SetOutputToWaveFile((Join-Path $d '{tid}.wav'), $fmt); $s.Speak('{text.replace(chr(39), chr(39) * 2)}')")
    lines += ["}}", "$s.SetOutputToNull()"]
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", f.name], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--engine", choices=["kokoro", "sapi"], default="kokoro")
    a = ap.parse_args()
    cfg = load_config(a.config)
    texts = {**cfg.phrases, **cfg.hard_negatives}
    out = cfg.data / "tts"
    (run_kokoro if a.engine == "kokoro" else run_sapi)(texts, out)
    print(f"wrote {sum(1 for _ in out.rglob('*.wav'))} clips under {out}")


if __name__ == "__main__":
    main()
