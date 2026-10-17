"""Figures for the Study H EEG x gaze fusion results (reads evaluations/results/eeg_gaze_fusion)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "evaluations/results/eeg_gaze_fusion"
COL = {"eeg": "#7b3fbf", "gaze": "#1f8a70", "fused": "#d9480f"}


def bootstrap_budget(n_boot=2000, risk=0.02):
    """Participant-level bootstrap of correct-commit rate, fused minus EEG, per time budget (real gaze)."""
    d = pd.read_csv(RES / "per_selection_budget.csv.gz")
    d = d[(d.gaze_offset_sd == 0) & (d.risk_target == risk)]
    pp = d.groupby(["budget_s", "modality", "participant_id"]).agg(c=("correct", "sum"), w=("wrong", "sum"), n=("correct", "size"))
    rng = np.random.default_rng(0)
    pids = sorted(d.participant_id.unique())
    rows = []
    for D in sorted(d.budget_s.unique()):
        t = pp.loc[D].unstack(0)
        idx = rng.integers(0, len(pids), (n_boot, len(pids)))
        out = {"budget_s": D, "risk_target": risk}
        for m in ("eeg", "gaze", "fused"):
            c, n, w = t[("c", m)].to_numpy(), t[("n", m)].to_numpy(), t[("w", m)].to_numpy()
            out[f"{m}_correct"] = c.sum() / n.sum(); out[f"{m}_wrong"] = int(w.sum())
            bs = c[idx].sum(1) / n[idx].sum(1)
            out[f"{m}_lo"], out[f"{m}_hi"] = np.percentile(bs, [2.5, 97.5])
        cf, ce, n = t[("c", "fused")].to_numpy(), t[("c", "eeg")].to_numpy(), t[("n", "fused")].to_numpy()
        diff = (cf[idx].sum(1) - ce[idx].sum(1)) / n[idx].sum(1)
        out["fused_minus_eeg"] = (cf.sum() - ce.sum()) / n.sum()
        out["diff_lo"], out["diff_hi"] = np.percentile(diff, [2.5, 97.5])
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "budget_bootstrap_ci.csv", index=False)
    return df


def main():
    print(bootstrap_budget().round(3).to_string(index=False))
    fx = pd.read_csv(RES / "argmax_accuracy_at_fixed_times.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ("eeg", "gaze", "fused"):
        ax.plot(fx.fixed_time, fx[m], "o-", color=COL[m], label=m)
    ax.axhline(1 / 72, color="grey", ls=":", lw=1, label="chance (1/72)")
    ax.set(xlabel="decision time after first flash (s)", ylabel="argmax accuracy (no stopping rule)", ylim=(0, 1),
           title="Study H, held-out Train runs, 16 non-ALS participants")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "accuracy_vs_time.png", dpi=150)

    bud = pd.read_csv(RES / "budget_frontier.csv")
    bud = bud[bud.risk_target == 0.02]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    b0 = bud[bud.gaze_offset_sd == 0]
    for m in ("eeg", "gaze", "fused"):
        d = b0[b0.modality == m].sort_values("budget_s")
        axes[0].plot(d.budget_s, d.accuracy, "o-", color=COL[m], label=f"{m} correct")
        axes[0].plot(d.budget_s, d.wrong / d.n, "x--", color=COL[m], alpha=0.6, label=f"{m} wrong")
    axes[0].set(xlabel="time budget (s); yield if not confident by then", ylabel="share of selections",
                title="Real gaze, risk target 2% (rest yield)")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    e = b0[b0.modality == "eeg"].sort_values("budget_s")
    axes[1].plot(e.budget_s, e.accuracy, "o-", color=COL["eeg"], lw=2, label="EEG alone")
    cm = plt.get_cmap("Oranges")
    for i, lv in enumerate(sorted(bud.gaze_offset_sd.unique())):
        d = bud[(bud.modality == "fused") & (bud.gaze_offset_sd == lv)].sort_values("budget_s")
        axes[1].plot(d.budget_s, d.accuracy, "o-", color=cm(0.35 + 0.65 * (1 - i / 5)), label=f"fused, offset SD {lv:g}")
    axes[1].set(xlabel="time budget (s)", title="Fused correct commits as gaze is degraded (synthetic offset)")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "budget_and_stress.png", dpi=150)


if __name__ == "__main__":
    main()
