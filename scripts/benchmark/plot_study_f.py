"""Accuracy-vs-evidence and error-vs-throughput plots from evaluations/results/eeg_stopping/aggregate.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {"fixed": ("fixed_", "k", "tab:gray", "fixed repetitions"), "bayes": ("bayes_tau", "t", "tab:blue", "Bayesian stopping"),
            "calibrated": ("calibrated_tau", "t", "tab:green", "calibrated confidence stopping"),
            "calibrated_abstain": ("calibrated_abstain_tau", "t", "tab:red", "calibrated stopping + abstention")}


def family(df: pd.DataFrame, key: str) -> pd.DataFrame:
    prefix = FAMILIES[key][0]
    m = df.policy.str.startswith(prefix)
    if key == "fixed":
        m &= ~df.policy.str.contains("abstain")
    if key in ("bayes", "calibrated"):
        m &= ~df.policy.str.contains("abstain")
    return df[m].sort_values("mean_reps")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "evaluations/results/eeg_stopping"))
    a = ap.parse_args()
    res = Path(a.results)
    agg = pd.read_csv(res / "aggregate.csv")
    for dec in sorted(agg.decoder.unique()):
        for split in ("evaluation", "evaluation_secondary"):
            d = agg[(agg.decoder == dec) & (agg.split == split)]
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
            for key, (_, _, col, label) in FAMILIES.items():
                f = family(d, key)
                ax[0].plot(f.mean_flashes, f.accuracy, "-o", ms=3, color=col, label=label)
                if key == "fixed":
                    ax[0].fill_between(f.mean_flashes, f.accuracy_lo, f.accuracy_hi, color=col, alpha=0.15)
                ax[1].plot(f.ccpm, f.error_rate, "-o", ms=3, color=col, label=label)
                ax[2].plot(f.mean_latency_s, f.accuracy, "-o", ms=3, color=col, label=label)
            ax[0].set(xlabel="mean flashes consumed per character", ylabel="character accuracy (abstain = wrong)",
                      title="Accuracy vs evidence")
            ax[1].set(xlabel="correct characters / min", ylabel="incorrect commits / selection", title="Error vs throughput")
            ax[2].set(xlabel="mean decision latency (s, from recorded timestamps)", ylabel="character accuracy", title="Accuracy vs latency")
            ax[0].legend(fontsize=7)
            fig.suptitle(f"Study F, {split}, decoder {dec}: mean over 10 participants (band = 95% participant bootstrap, fixed only)")
            fig.tight_layout()
            fig.savefig(res / f"curves_{split}_{dec}.png", dpi=130)
            plt.close(fig)
    per = pd.read_csv(res / "per_participant.csv")
    for dec in sorted(per.decoder.unique()):
        d = per[(per.decoder == dec) & (per.split == "evaluation") & per.policy.str.match(r"fixed_(0[1-9]|1[0-4])$")]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for pid, g in d.groupby("participant_id"):
            g = g.sort_values("mean_reps")
            ax.plot(g.mean_reps, g.accuracy, "-o", ms=3, label=pid)
        ax.set(xlabel="repetitions (of 14)", ylabel="accuracy", title=f"Per-participant fixed-repetition accuracy ({dec}, Static session 3)")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(res / f"per_participant_fixed_{dec}.png", dpi=130); plt.close(fig)
    print("plots written to", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
