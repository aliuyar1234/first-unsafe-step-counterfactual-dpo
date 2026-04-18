from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import src.eval.aggregate_metrics as aggregate_metrics_module
import src.eval.run_eval as run_eval_module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runs_csv_path_is_split_specific_off_test() -> None:
    assert run_eval_module._runs_csv_path("test").as_posix().endswith("results/metrics/test_runs.csv")
    assert run_eval_module._runs_csv_path("train_pilot").as_posix().endswith("results/metrics/train_pilot_runs.csv")


def test_validate_complete_real_test_outputs_rejects_non_real_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_eval_module, "repo_root", lambda: tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "eval_main.yaml").write_text(
        yaml.safe_dump({"eval": {"methods": ["base"], "eval_seeds": [0]}}),
        encoding="utf-8",
    )
    split_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        yaml.safe_dump({"stems": [{"scenario_stem": "Stem-A", "domain": "healthcare", "split": "test"}]}),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "results" / "runs" / "test" / "base" / "Stem-A" / "mandated" / "seed0" / "raw_request.json",
        {"backend": "simulated"},
    )
    frame = pd.DataFrame(
        [
            {
                "run_id": "base_run",
                "scenario_stem": "Stem-A",
                "domain": "healthcare",
                "variant": "mandated",
                "split": "test",
                "method": "base",
                "seed": 0,
                "max_severity": 4,
                "primary_violation_type": "data_falsification",
                "first_unsafe_step": 3,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
                "audited": False,
                "final_label_source": "judge",
            }
        ]
    )

    with pytest.raises(ValueError, match="not benchmark-backed"):
        run_eval_module._validate_complete_real_test_outputs(frame)


def test_aggregate_metrics_uses_bootstrap_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(aggregate_metrics_module, "repo_root", lambda: tmp_path)
    captured: dict[str, int] = {}

    def _fake_bootstrap(summary_csv: Path, test_runs_csv: Path, out_csv: Path, *, resamples: int, bootstrap_seed: int) -> None:
        captured["resamples"] = resamples
        captured["bootstrap_seed"] = bootstrap_seed
        pd.DataFrame(
            [
                {
                    "method": "base",
                    "metric": "hs_mr",
                    "ci_low": 0.1,
                    "ci_high": 0.2,
                    "resamples": resamples,
                    "bootstrap_seed": bootstrap_seed,
                }
            ]
        ).to_csv(out_csv, index=False)

    monkeypatch.setattr(aggregate_metrics_module, "compute_bootstrap_ci", _fake_bootstrap)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "eval_main.yaml").write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "methods": ["base"],
                    "eval_seeds": [0],
                    "bootstrap": {"enabled": True, "resamples": 321, "seed": 77},
                }
            }
        ),
        encoding="utf-8",
    )
    split_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        yaml.safe_dump({"stems": [{"scenario_stem": "Stem-A", "domain": "healthcare", "split": "test"}]}),
        encoding="utf-8",
    )
    runs_dir = tmp_path / "results" / "runs" / "test" / "base" / "Stem-A" / "mandated" / "seed0"
    _write_json(runs_dir / "raw_request.json", {"backend": "odcv_benchmark"})
    _write_json(
        tmp_path / "results" / "runs" / "test" / "base" / "Stem-A" / "incentivized" / "seed0" / "raw_request.json",
        {"backend": "odcv_benchmark"},
    )
    metrics_root = tmp_path / "results" / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": "base_run",
                "scenario_stem": "Stem-A",
                "domain": "healthcare",
                "variant": "mandated",
                "split": "test",
                "method": "base",
                "seed": 0,
                "max_severity": 4,
                "primary_violation_type": "data_falsification",
                "first_unsafe_step": 3,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
                "audited": False,
                "final_label_source": "judge",
            },
            {
                "run_id": "base_run_incentivized",
                "scenario_stem": "Stem-A",
                "domain": "healthcare",
                "variant": "incentivized",
                "split": "test",
                "method": "base",
                "seed": 0,
                "max_severity": 2,
                "primary_violation_type": "metric_gaming",
                "first_unsafe_step": 2,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
                "audited": False,
                "final_label_source": "judge",
            }
        ]
    ).to_csv(metrics_root / "test_runs.csv", index=False)

    aggregate_metrics_module.aggregate_metrics(tmp_path / "configs" / "eval_main.yaml")

    assert captured == {"resamples": 321, "bootstrap_seed": 77}
