"""Benchmark panels with units and Wilson intervals."""

from pathlib import Path
import numpy as np


def plot_summary(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [
        ("success_rate", "Success probability"),
        ("mean_capped_time", "Capped first-hit time (s)"),
        ("lateral_accel_cost_success_mean", "Lateral cost on successes (m²/s³)"),
        ("control_variation_success_mean", "Action variation on successes"),
    ]
    labels = [r["algorithm"] + " / " + r["scenario"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(rows) * 1.2), 7))
    for ax, (key, label) in zip(axes.flat, fields):
        vals = [np.nan if r.get(key) is None else r[key] for r in rows]
        ax.bar(np.arange(len(rows)), vals)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=30, ha="right")
        ax.set_ylabel(label)
        if key == "success_rate":
            ax.set_ylim(0.0, 1.0)
            low = [max(0., r["success_rate"] - r["wilson_low"]) for r in rows]
            high = [max(0., r["wilson_high"] - r["success_rate"]) for r in rows]
            ax.errorbar(
                np.arange(len(rows)),
                vals,
                yerr=[low, high],
                fmt="none",
                color="black",
                capsize=3,
            )
        for i, value in enumerate(vals):
            if not np.isfinite(value):
                ax.text(i, 0.05, "N/A", transform=ax.get_xaxis_transform(), ha="center")
    fig.suptitle("Per scenario results; Wilson intervals describe evaluation episodes")
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
