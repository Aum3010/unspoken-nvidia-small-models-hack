"""Load configs/eval.yaml into a small object with resolved paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    raw: dict
    path: Path

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def data(self) -> Path:
        return (ROOT / self.raw["paths"]["data"]).resolve()

    @property
    def results(self) -> Path:
        return (ROOT / self.raw["paths"]["results"]).resolve()

    @property
    def sr(self) -> int:
        return self.raw["sample_rate"]

    @property
    def phrases(self) -> dict[str, str]:
        return {str(k): v for k, v in self.raw["phrases"].items()}

    @property
    def hard_negatives(self) -> dict[str, str]:
        return dict(self.raw["hard_negatives"])

    def model(self, name: str) -> dict:
        return next(m for m in self.raw["models"] if m["name"] == name)

    def enabled_models(self) -> list[dict]:
        return [m for m in self.raw["models"] if m.get("enabled", True)]


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p) if not p.exists() else p.resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    for parent in raw.pop("extends", []) or []:  # small overlay configs (e.g. local smoke test)
        base = load_config(p.parent / parent).raw
        raw = _merge(base, raw)
    return Config(raw=raw, path=p)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out
