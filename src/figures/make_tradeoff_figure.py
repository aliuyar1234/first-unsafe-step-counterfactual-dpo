"""Generate the safety-utility frontier figure from `results/metrics/test_summary.csv`."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.common import ensure_parent, load_csv


def main(csv_path: Path, out_path: Path) -> None:
    df = load_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for _, row in df.iterrows():
        ax.scatter(float(row["safe_kpi_sr"]), float(row["hs_mr"]), s=80)
        ax.annotate(str(row["method"]), (float(row["safe_kpi_sr"]), float(row["hs_mr"])), xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Safe-KPI-SR")
    ax.set_ylabel("HS-MR")
    ax.set_title("Held-Out Safety-Utility Frontier")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
