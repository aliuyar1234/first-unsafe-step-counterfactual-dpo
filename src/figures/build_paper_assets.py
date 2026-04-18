"""Build paper-facing tables and figures from the pivot-path evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eval.aggregate_metrics import aggregate_metrics
from src.eval.manual_audit import manual_audit_manifest_is_resolved
from src.figures.make_ablation_table import main as make_ablation_table
from src.figures.make_error_taxonomy_table import main as make_error_taxonomy_table
from src.figures.make_main_results_table import main as make_main_results_table
from src.figures.make_per_stem_results_table import main as make_per_stem_results_table
from src.figures.make_pressure_figure import main as make_pressure_figure
from src.figures.make_qualitative_figure import main as make_qualitative_figure
from src.figures.make_tradeoff_figure import main as make_tradeoff_figure
from src.utils.io import repo_root


def _combine_csvs(inputs: list[Path], out_path: Path, sort_columns: list[str]) -> None:
    frames = [pd.read_csv(path) for path in inputs]
    combined = pd.concat(frames, ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values(sort_columns, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)


def _write_qualitative_placeholder(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        "Qualitative trajectory figure pending manual audit resolution",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        0.5,
        0.38,
        "Resolve prompt_only and sparse_pilots audit queues, merge audited labels, and rebuild assets.",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _combine_manual_audits(root: Path) -> bool:
    audits_root = root / "results" / "audits"
    prompt_manifest = audits_root / "manual_audit_manifest_prompt_only.json"
    sparse_manifest = audits_root / "manual_audit_manifest_sparse_pilots.json"
    prompt_csv = audits_root / "manual_audit_prompt_only.csv"
    sparse_csv = audits_root / "manual_audit_sparse_pilots.csv"
    out_csv = audits_root / "manual_audit.csv"
    out_manifest = audits_root / "manual_audit_manifest.json"

    prompt_resolved = prompt_csv.exists() and manual_audit_manifest_is_resolved(prompt_manifest)
    sparse_resolved = sparse_csv.exists() and manual_audit_manifest_is_resolved(sparse_manifest)
    if not (prompt_resolved and sparse_resolved):
        status = {
            "status": "pending",
            "source_csvs": [str(prompt_csv), str(sparse_csv)],
            "prompt_only_resolved": prompt_resolved,
            "sparse_pilots_resolved": sparse_resolved,
        }
        out_manifest.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return False

    _combine_csvs(
        [prompt_csv, sparse_csv],
        out_csv,
        ["resolved_method", "scenario_stem", "variant", "blinded_id"],
    )
    out_manifest.write_text(
        json.dumps(
            {
                "status": "resolved",
                "source_csvs": [str(prompt_csv), str(sparse_csv)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def build_paper_assets() -> None:
    root = repo_root()
    metrics_root = root / "results" / "metrics"
    audits_root = root / "results" / "audits"
    paper_tables = root / "paper" / "tables"
    paper_figures = root / "paper" / "figures"

    _combine_csvs(
        [
            metrics_root / "test_runs_prompt_only.csv",
            metrics_root / "test_runs_sparse_pilots.csv",
        ],
        metrics_root / "test_runs.csv",
        ["method", "scenario_stem", "variant", "seed"],
    )
    aggregate_metrics(root / "configs" / "eval_main.yaml")

    make_main_results_table(metrics_root / "test_summary.csv", paper_tables / "main_results.tex")
    make_ablation_table(metrics_root / "test_summary.csv", paper_tables / "ablation.tex")
    make_error_taxonomy_table(metrics_root / "test_runs.csv", paper_tables / "error_taxonomy.tex")
    make_per_stem_results_table(metrics_root / "per_stem_results.csv", paper_tables / "per_stem_results.tex")
    make_pressure_figure(metrics_root / "test_summary.csv", paper_figures / "pressure_sensitivity.pdf")
    make_tradeoff_figure(metrics_root / "test_summary.csv", paper_figures / "tradeoff.pdf")

    if _combine_manual_audits(root):
        make_qualitative_figure(root / "results" / "runs", audits_root / "manual_audit.csv", paper_figures / "qualitative_trajectory.pdf")
    else:
        _write_qualitative_placeholder(paper_figures / "qualitative_trajectory.pdf")


if __name__ == "__main__":
    build_paper_assets()
