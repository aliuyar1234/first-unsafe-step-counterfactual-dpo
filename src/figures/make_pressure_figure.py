"""Generate the mandated-vs-incentivized pressure sensitivity figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.common import ensure_parent, load_csv


def main(csv_path: Path, out_path: Path) -> None:
    df = load_csv(csv_path).sort_values("method")
    methods = df["method"].tolist()
    mandated = df["hs_mr_mandated"].astype(float).to_numpy()
    incentivized = df["hs_mr_incentivized"].astype(float).to_numpy()
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, mandated, width=width, label="Mandated", color="#2f6db0")
    ax.bar(x + width / 2, incentivized, width=width, label="Incentivized", color="#d07a35")
    ax.set_ylabel("High-Severity Misalignment Rate")
    ax.set_title("Mandated vs Incentivized High-Severity Misalignment")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylim(0, max(0.05, float(max(mandated.max(initial=0), incentivized.max(initial=0))) * 1.2))
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
