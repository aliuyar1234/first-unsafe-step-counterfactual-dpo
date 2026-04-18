from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from src.eval import freeze_train_corpus as freeze_train_corpus_module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_freeze_train_corpus_writes_summary_delta_and_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(freeze_train_corpus_module, "repo_root", lambda: tmp_path)

    split_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        yaml.safe_dump({"stems": [{"scenario_stem": "Stem-A", "split": "train", "domain": "healthcare"}]}),
        encoding="utf-8",
    )

    (tmp_path / "external" / "ODCV-Bench").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed" / "pairs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed" / "pairs" / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"scenario_stem": "Stem-A", "variant": "mandated"}),
                json.dumps({"scenario_stem": "Stem-A", "variant": "mandated"}),
                json.dumps({"scenario_stem": "Stem-A", "variant": "incentivized"}),
                json.dumps({"scenario_stem": "Stem-A", "variant": "incentivized"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_json(
        tmp_path / "results" / "metrics" / "train_pair_attrition_summary.json",
        {
            "run_split": "train",
            "method": "base",
            "split": "train",
            "total_runs": 8,
            "runs_severity_ge_3": 2,
            "runs_localized": 2,
            "runs_with_kept_rewrites": 2,
            "suspicious_low_severity_runs": 0,
            "severity_distribution": {"0": 6, "4": 2},
        },
    )
    _write_csv(
        tmp_path / "results" / "metrics" / "train_pair_attrition_by_group.csv",
        [
            "split",
            "scenario_stem",
            "variant",
            "num_runs",
            "num_severity_ge_3",
            "num_localized",
            "num_kept_rewrites",
            "num_pairs_kept",
            "num_low_severity_sensitive_edits",
            "num_low_severity_completion_overrides",
            "num_low_severity_review_evidence_writes",
        ],
        [
            {
                "split": "train",
                "scenario_stem": "Stem-A",
                "variant": "mandated",
                "num_runs": 4,
                "num_severity_ge_3": 1,
                "num_localized": 1,
                "num_kept_rewrites": 2,
                "num_pairs_kept": 2,
                "num_low_severity_sensitive_edits": 0,
                "num_low_severity_completion_overrides": 0,
                "num_low_severity_review_evidence_writes": 0,
            },
            {
                "split": "train",
                "scenario_stem": "Stem-A",
                "variant": "incentivized",
                "num_runs": 4,
                "num_severity_ge_3": 1,
                "num_localized": 1,
                "num_kept_rewrites": 2,
                "num_pairs_kept": 2,
                "num_low_severity_sensitive_edits": 0,
                "num_low_severity_completion_overrides": 0,
                "num_low_severity_review_evidence_writes": 0,
            },
        ],
    )
    _write_csv(
        tmp_path / "results" / "metrics" / "pair_stats.csv",
        [
            "split",
            "scenario_stem",
            "variant",
            "num_runs_sampled",
            "num_runs_severity_ge_3",
            "num_raw_rewrite_candidates",
            "num_kept_rewrites",
            "num_pairs_kept",
            "refusal_only_kept_fraction",
        ],
        [
            {
                "split": "train",
                "scenario_stem": "Stem-A",
                "variant": "mandated",
                "num_runs_sampled": 4,
                "num_runs_severity_ge_3": 1,
                "num_raw_rewrite_candidates": 3,
                "num_kept_rewrites": 2,
                "num_pairs_kept": 2,
                "refusal_only_kept_fraction": 0.0,
            },
            {
                "split": "train",
                "scenario_stem": "Stem-A",
                "variant": "incentivized",
                "num_runs_sampled": 4,
                "num_runs_severity_ge_3": 1,
                "num_raw_rewrite_candidates": 3,
                "num_kept_rewrites": 2,
                "num_pairs_kept": 2,
                "refusal_only_kept_fraction": 0.0,
            },
        ],
    )

    manifest = freeze_train_corpus_module.freeze_train_corpus(
        before_snapshot={
            "snapshot_label": "unit_test_before",
            "source_note": "unit test snapshot",
            "total_runs": 8,
            "severity_0_runs": 8,
            "severity_3_runs": 0,
            "severity_4_runs": 0,
            "severity_5_runs": 0,
            "runs_severity_ge_3": 0,
            "runs_localized": 0,
            "runs_with_kept_rewrites": 0,
            "accepted_train_pairs": 0,
            "contributing_train_stems": 0,
        }
    )

    with (tmp_path / "results" / "metrics" / "train_corpus_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 1
    assert summary_rows[0]["accepted_train_pairs"] == "4"
    assert summary_rows[0]["contributing_train_stems"] == "1"
    assert summary_rows[0]["contributing_stem_variant_groups"] == "2"

    with (tmp_path / "results" / "metrics" / "train_pair_yield_by_stem_variant.csv").open("r", encoding="utf-8", newline="") as handle:
        yield_rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in yield_rows] == ["incentivized", "mandated"]
    assert {row["num_localized"] for row in yield_rows} == {"1"}

    with (tmp_path / "results" / "metrics" / "judge_repair_delta.csv").open("r", encoding="utf-8", newline="") as handle:
        delta_rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert delta_rows["accepted_train_pairs"]["after_repair"] == "4"
    assert delta_rows["accepted_train_pairs"]["delta"] == "4"

    assert manifest["pair_gate_status"]["current_pairs"] == 4
    assert manifest["pair_gate_status"]["current_stems"] == 1
    assert manifest["frozen_train_corpus"]["contributing_stem_variant_groups"] == 2
    frozen_manifest = json.loads((tmp_path / "manifests" / "train_corpus_repaired_v1.json").read_text(encoding="utf-8"))
    assert frozen_manifest["metrics"]["judge_repair_delta_csv"]["path"] == "results/metrics/judge_repair_delta.csv"
    assert frozen_manifest["paper_path_recommendation"]["decision"] == "pivot_to_evaluation_first_sparse_data_limit_study"
