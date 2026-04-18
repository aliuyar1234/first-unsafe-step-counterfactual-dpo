"""Freeze the repaired real train corpus into versioned summary artifacts.

CLI shape:
    python -m src.eval.freeze_train_corpus
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.judging.judge_trajectory import JUDGE_BACKEND_VERSION, JUDGE_PROMPT_VERSION
from src.utils.io import load_json, read_jsonl, repo_root, utc_now_iso
from src.utils.manifests import external_repo_manifest, sha256_file, write_manifest

DEFAULT_PRE_REPAIR_SNAPSHOT: dict[str, Any] = {
    "snapshot_label": "pre_final_judge_repairs",
    "source_note": (
        "Authoritative pre-repair snapshot logged on 2026-03-12 after the real 384-run train sweep "
        "completed but before the final bounded executed-action judge repairs were frozen."
    ),
    "total_runs": 384,
    "severity_0_runs": 382,
    "severity_3_runs": 0,
    "severity_4_runs": 2,
    "severity_5_runs": 0,
    "runs_severity_ge_3": 2,
    "runs_localized": 2,
    "runs_with_kept_rewrites": 2,
    "accepted_train_pairs": 4,
    "contributing_train_stems": 2,
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _current_snapshot(
    *,
    split: str,
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = repo_root()
    metrics_root = root / "results" / "metrics"
    attrition_summary = load_json(metrics_root / f"{split}_pair_attrition_summary.json")
    attrition_rows = {
        (str(row["scenario_stem"]), str(row["variant"])): row
        for row in _read_csv_rows(metrics_root / f"{split}_pair_attrition_by_group.csv")
    }
    pair_stats_rows = [
        row
        for row in _read_csv_rows(metrics_root / "pair_stats.csv")
        if str(row["split"]) == split
    ]
    train_pairs = read_jsonl(root / "data" / "processed" / "pairs" / f"{split}.jsonl")

    yield_rows: list[dict[str, Any]] = []
    for row in pair_stats_rows:
        key = (str(row["scenario_stem"]), str(row["variant"]))
        attrition_row = attrition_rows.get(key)
        if attrition_row is None:
            raise RuntimeError(f"Missing attrition row for {key} while freezing train corpus.")
        yield_rows.append(
            {
                "split": split,
                "scenario_stem": row["scenario_stem"],
                "variant": row["variant"],
                "num_runs_sampled": int(row["num_runs_sampled"]),
                "num_runs_severity_ge_3": int(row["num_runs_severity_ge_3"]),
                "num_localized": int(attrition_row["num_localized"]),
                "num_kept_rewrites": int(row["num_kept_rewrites"]),
                "num_pairs_kept": int(row["num_pairs_kept"]),
                "refusal_only_kept_fraction": float(row["refusal_only_kept_fraction"]),
                "num_low_severity_sensitive_edits": int(attrition_row["num_low_severity_sensitive_edits"]),
                "num_low_severity_completion_overrides": int(attrition_row["num_low_severity_completion_overrides"]),
                "num_low_severity_review_evidence_writes": int(attrition_row["num_low_severity_review_evidence_writes"]),
            }
        )
    yield_rows.sort(key=lambda row: (-int(row["num_pairs_kept"]), str(row["scenario_stem"]), str(row["variant"])))

    accepted_pairs = sum(int(row["num_pairs_kept"]) for row in yield_rows)
    if accepted_pairs != len(train_pairs):
        raise RuntimeError(
            f"Train pair count mismatch while freezing corpus: pair_stats={accepted_pairs}, train.jsonl={len(train_pairs)}"
        )

    contributing_groups = [row for row in yield_rows if int(row["num_pairs_kept"]) > 0]
    contributing_stems = sorted({str(row["scenario_stem"]) for row in contributing_groups})
    severity_distribution = {str(key): int(value) for key, value in dict(attrition_summary["severity_distribution"]).items()}

    snapshot = {
        "run_split": split,
        "method": method,
        "total_runs": int(attrition_summary["total_runs"]),
        "runs_severity_ge_3": int(attrition_summary["runs_severity_ge_3"]),
        "runs_localized": int(attrition_summary["runs_localized"]),
        "runs_with_kept_rewrites": int(attrition_summary["runs_with_kept_rewrites"]),
        "accepted_train_pairs": accepted_pairs,
        "contributing_train_stems": len(contributing_stems),
        "contributing_stem_variant_groups": len(contributing_groups),
        "suspicious_low_severity_runs": int(attrition_summary["suspicious_low_severity_runs"]),
        "severity_distribution": severity_distribution,
        "contributing_stem_names": contributing_stems,
    }
    return snapshot, yield_rows


def _summary_row(
    current_snapshot: dict[str, Any],
    *,
    split_manifest_sha256: str,
    train_pair_sha256: str,
    pair_stats_sha256: str,
    attrition_summary_sha256: str,
    attrition_by_group_sha256: str,
) -> dict[str, Any]:
    severity_distribution = dict(current_snapshot["severity_distribution"])
    return {
        "generated_at_utc": utc_now_iso(),
        "run_split": current_snapshot["run_split"],
        "method": current_snapshot["method"],
        "total_runs": current_snapshot["total_runs"],
        "severity_0_runs": severity_distribution.get("0", 0),
        "severity_3_runs": severity_distribution.get("3", 0),
        "severity_4_runs": severity_distribution.get("4", 0),
        "severity_5_runs": severity_distribution.get("5", 0),
        "runs_severity_ge_3": current_snapshot["runs_severity_ge_3"],
        "runs_localized": current_snapshot["runs_localized"],
        "runs_with_kept_rewrites": current_snapshot["runs_with_kept_rewrites"],
        "accepted_train_pairs": current_snapshot["accepted_train_pairs"],
        "contributing_train_stems": current_snapshot["contributing_train_stems"],
        "contributing_stem_variant_groups": current_snapshot["contributing_stem_variant_groups"],
        "suspicious_low_severity_runs": current_snapshot["suspicious_low_severity_runs"],
        "judge_backend_version": JUDGE_BACKEND_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "split_manifest_sha256": split_manifest_sha256,
        "train_pair_file": "data/processed/pairs/train.jsonl",
        "train_pair_sha256": train_pair_sha256,
        "pair_stats_file": "results/metrics/pair_stats.csv",
        "pair_stats_sha256": pair_stats_sha256,
        "attrition_summary_file": "results/metrics/train_pair_attrition_summary.json",
        "attrition_summary_sha256": attrition_summary_sha256,
        "attrition_by_group_file": "results/metrics/train_pair_attrition_by_group.csv",
        "attrition_by_group_sha256": attrition_by_group_sha256,
    }


def _delta_rows(before_snapshot: dict[str, Any], current_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    severity_distribution = dict(current_snapshot["severity_distribution"])
    metrics = [
        ("total_runs", int(before_snapshot["total_runs"]), int(current_snapshot["total_runs"]), "count"),
        ("severity_0_runs", int(before_snapshot["severity_0_runs"]), int(severity_distribution.get("0", 0)), "count"),
        ("severity_3_runs", int(before_snapshot["severity_3_runs"]), int(severity_distribution.get("3", 0)), "count"),
        ("severity_4_runs", int(before_snapshot["severity_4_runs"]), int(severity_distribution.get("4", 0)), "count"),
        ("severity_5_runs", int(before_snapshot["severity_5_runs"]), int(severity_distribution.get("5", 0)), "count"),
        ("runs_severity_ge_3", int(before_snapshot["runs_severity_ge_3"]), int(current_snapshot["runs_severity_ge_3"]), "count"),
        ("runs_localized", int(before_snapshot["runs_localized"]), int(current_snapshot["runs_localized"]), "count"),
        (
            "runs_with_kept_rewrites",
            int(before_snapshot["runs_with_kept_rewrites"]),
            int(current_snapshot["runs_with_kept_rewrites"]),
            "count",
        ),
        (
            "accepted_train_pairs",
            int(before_snapshot["accepted_train_pairs"]),
            int(current_snapshot["accepted_train_pairs"]),
            "count",
        ),
        (
            "contributing_train_stems",
            int(before_snapshot["contributing_train_stems"]),
            int(current_snapshot["contributing_train_stems"]),
            "count",
        ),
    ]
    return [
        {
            "metric": metric,
            "before_repair": before_value,
            "after_repair": after_value,
            "delta": after_value - before_value,
            "unit": unit,
            "before_snapshot_label": str(before_snapshot["snapshot_label"]),
            "before_snapshot_source": str(before_snapshot["source_note"]),
        }
        for metric, before_value, after_value, unit in metrics
    ]


def freeze_train_corpus(
    *,
    split: str = "train",
    method: str = "base",
    before_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = repo_root()
    metrics_root = root / "results" / "metrics"
    manifests_root = root / "manifests"
    before_snapshot = before_snapshot or DEFAULT_PRE_REPAIR_SNAPSHOT

    current_snapshot, yield_rows = _current_snapshot(split=split, method=method)

    split_manifest_path = root / "data" / "processed" / "splits" / "split_v1.yaml"
    train_pair_path = root / "data" / "processed" / "pairs" / "train.jsonl"
    pair_stats_path = metrics_root / "pair_stats.csv"
    attrition_summary_path = metrics_root / "train_pair_attrition_summary.json"
    attrition_by_group_path = metrics_root / "train_pair_attrition_by_group.csv"

    summary_csv_path = metrics_root / "train_corpus_summary.csv"
    yield_csv_path = metrics_root / "train_pair_yield_by_stem_variant.csv"
    delta_csv_path = metrics_root / "judge_repair_delta.csv"

    _write_csv(
        summary_csv_path,
        [
            "generated_at_utc",
            "run_split",
            "method",
            "total_runs",
            "severity_0_runs",
            "severity_3_runs",
            "severity_4_runs",
            "severity_5_runs",
            "runs_severity_ge_3",
            "runs_localized",
            "runs_with_kept_rewrites",
            "accepted_train_pairs",
            "contributing_train_stems",
            "contributing_stem_variant_groups",
            "suspicious_low_severity_runs",
            "judge_backend_version",
            "judge_prompt_version",
            "split_manifest_sha256",
            "train_pair_file",
            "train_pair_sha256",
            "pair_stats_file",
            "pair_stats_sha256",
            "attrition_summary_file",
            "attrition_summary_sha256",
            "attrition_by_group_file",
            "attrition_by_group_sha256",
        ],
        [
            _summary_row(
                current_snapshot,
                split_manifest_sha256=sha256_file(split_manifest_path),
                train_pair_sha256=sha256_file(train_pair_path),
                pair_stats_sha256=sha256_file(pair_stats_path),
                attrition_summary_sha256=sha256_file(attrition_summary_path),
                attrition_by_group_sha256=sha256_file(attrition_by_group_path),
            )
        ],
    )

    _write_csv(
        yield_csv_path,
        [
            "split",
            "scenario_stem",
            "variant",
            "num_runs_sampled",
            "num_runs_severity_ge_3",
            "num_localized",
            "num_kept_rewrites",
            "num_pairs_kept",
            "refusal_only_kept_fraction",
            "num_low_severity_sensitive_edits",
            "num_low_severity_completion_overrides",
            "num_low_severity_review_evidence_writes",
        ],
        yield_rows,
    )

    _write_csv(
        delta_csv_path,
        [
            "metric",
            "before_repair",
            "after_repair",
            "delta",
            "unit",
            "before_snapshot_label",
            "before_snapshot_source",
        ],
        _delta_rows(before_snapshot, current_snapshot),
    )

    manifest_payload = {
        "generated_at_utc": utc_now_iso(),
        "manifest_version": "train_corpus_repaired_v1",
        "project": "odcv_fus_dpo_v1",
        "freeze_reason": "Sparse-data pivot after exhausting bounded judge repairs on the real 384-run train sweep.",
        "run_split": split,
        "method": method,
        "judge_backend_version": JUDGE_BACKEND_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "external_repo": external_repo_manifest(root / "external" / "ODCV-Bench"),
        "split_manifest": {
            "path": "data/processed/splits/split_v1.yaml",
            "sha256": sha256_file(split_manifest_path),
        },
        "train_pairs": {
            "path": "data/processed/pairs/train.jsonl",
            "sha256": sha256_file(train_pair_path),
            "count": current_snapshot["accepted_train_pairs"],
        },
        "metrics": {
            "train_corpus_summary_csv": {
                "path": "results/metrics/train_corpus_summary.csv",
                "sha256": sha256_file(summary_csv_path),
            },
            "train_pair_yield_by_stem_variant_csv": {
                "path": "results/metrics/train_pair_yield_by_stem_variant.csv",
                "sha256": sha256_file(yield_csv_path),
            },
            "judge_repair_delta_csv": {
                "path": "results/metrics/judge_repair_delta.csv",
                "sha256": sha256_file(delta_csv_path),
            },
            "pair_stats_csv": {
                "path": "results/metrics/pair_stats.csv",
                "sha256": sha256_file(pair_stats_path),
            },
            "attrition_summary_json": {
                "path": "results/metrics/train_pair_attrition_summary.json",
                "sha256": sha256_file(attrition_summary_path),
            },
            "attrition_by_group_csv": {
                "path": "results/metrics/train_pair_attrition_by_group.csv",
                "sha256": sha256_file(attrition_by_group_path),
            },
        },
        "pre_repair_snapshot": before_snapshot,
        "frozen_train_corpus": current_snapshot,
        "pair_gate_status": {
            "target_pairs": 300,
            "target_stems": 18,
            "current_pairs": current_snapshot["accepted_train_pairs"],
            "current_stems": current_snapshot["contributing_train_stems"],
            "met": bool(
                current_snapshot["accepted_train_pairs"] >= 300
                and current_snapshot["contributing_train_stems"] >= 18
            ),
        },
        "paper_path_recommendation": {
            "decision": "pivot_to_evaluation_first_sparse_data_limit_study",
            "rationale": (
                "The repaired train corpus is frozen after exhausting the bounded judge-repair route; "
                "the pair gate remains far below target, so further rescue should move to evaluation-first framing."
            ),
        },
    }
    manifest_path = manifests_root / "train_corpus_repaired_v1.json"
    write_manifest(manifest_path, manifest_payload)
    return manifest_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--method", default="base")
    args = parser.parse_args()
    manifest = freeze_train_corpus(split=str(args.split), method=str(args.method))
    print(manifest["pair_gate_status"])
