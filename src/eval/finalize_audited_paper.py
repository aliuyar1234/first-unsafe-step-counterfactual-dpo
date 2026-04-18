"""Resolve reviewed audit queues, merge labels, and rebuild the audited paper layer."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.aggregate_metrics import aggregate_metrics
from src.eval.manual_audit import merge_manual_audit_resolutions, resolve_manual_audit_queue
from src.figures.build_paper_assets import build_paper_assets
from src.utils.io import repo_root


def _finalize_tag(tag: str, config_path: Path) -> None:
    root = repo_root()
    audits_root = root / "results" / "audits"
    metrics_root = root / "results" / "metrics"

    queue_csv = audits_root / f"manual_audit_queue_{tag}.csv"
    key_json = audits_root / f"manual_audit_key_{tag}.json"
    final_csv = audits_root / f"manual_audit_{tag}.csv"
    manifest_json = audits_root / f"manual_audit_manifest_{tag}.json"
    test_runs_csv = metrics_root / f"test_runs_{tag}.csv"

    resolve_manual_audit_queue(queue_csv, key_json, final_csv, manifest_json=manifest_json)
    merge_manual_audit_resolutions(test_runs_csv, final_csv, out_csv=test_runs_csv)
    aggregate_metrics(config_path)


def finalize_audited_paper() -> None:
    root = repo_root()
    _finalize_tag("prompt_only", root / "configs" / "eval_prompt_only.yaml")
    _finalize_tag("sparse_pilots", root / "configs" / "eval_sparse_pilots.yaml")
    build_paper_assets()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-build-paper-assets",
        action="store_true",
        help="Resolve and merge both audit layers but do not rebuild the canonical paper assets.",
    )
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    if args.skip_build_paper_assets:
        root = repo_root()
        _finalize_tag("prompt_only", root / "configs" / "eval_prompt_only.yaml")
        _finalize_tag("sparse_pilots", root / "configs" / "eval_sparse_pilots.yaml")
    else:
        finalize_audited_paper()
