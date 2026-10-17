"""Phase 1.6: degradation grid, applied separately (never stacked) to positive clips.

Called by build_manifest.py after the speaker split, so every copy inherits its
source's split. Standalone: python scripts/degrade.py in.wav out_dir/
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import audio  # noqa: E402


def effects(cfg: dict, noise_pool: list[Path]) -> list[tuple[str, callable]]:
    """(condition_name, fn(x, rng)) for every grid point in configs/eval.yaml `degrade`."""
    out = [(f"gain_{db}dB", lambda x, r, db=db: audio.gain(x, db)) for db in cfg["gain_db"]]
    noise_kind = "musan" if noise_pool else "pink"
    for snr in cfg["snr_db"]:
        def fn(x, r, snr=snr):
            n = audio.load(noise_pool[r.integers(len(noise_pool))]) if noise_pool else audio.pink(len(x), r)
            if len(n) > len(x):
                start = r.integers(0, len(n) - len(x))
                n = n[start:start + len(x)]
            return audio.add_noise(x, snr, n)
        out.append((f"noise_{noise_kind}_snr{snr}", fn))
    lo, hi = cfg["bandpass_hz"]
    out.append((f"bandpass_{lo}-{hi}", lambda x, r: audio.bandpass(x, lo, hi)))
    out.append((f"lowpass_{cfg['lowpass_hz']}", lambda x, r: audio.lowpass(x, cfg["lowpass_hz"])))
    out += [(f"tempo_{t}x", lambda x, r, t=t: audio.tempo(x, t)) for t in cfg["tempo"]]
    out += [(f"reverb_rt60_{t}s", lambda x, r, t=t: audio.reverb(x, t, r)) for t in cfg["rt60_s"]]
    return out


def degrade_file(src: Path, out_dir: Path, cfg: dict, noise_pool: list[Path], overwrite: bool = False) -> list[tuple[str, Path]]:
    x = None
    made = []
    for name, fn in effects(cfg, noise_pool):
        dst = out_dir / name / f"{src.stem}.wav"
        if overwrite or not dst.exists():
            if x is None:
                x = audio.load(src)
            rng = np.random.default_rng(zlib.crc32(f"{src}|{name}".encode()))
            audio.save(dst, fn(x, rng))
        made.append((name, dst))
    return made


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())["degrade"]
    for cond, path in degrade_file(Path(sys.argv[1]), Path(sys.argv[2]), cfg, []):
        print(cond, path)
