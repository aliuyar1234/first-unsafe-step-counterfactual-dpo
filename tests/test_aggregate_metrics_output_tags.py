from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.eval import aggregate_metrics as aggregate_metrics_module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_metrics_uses_output_tag_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(aggregate_metrics_module, "repo_root", lambda: tmp_path)

    cfg_path = tmp_path / "configs" / "eval_prompt_only.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "methods": ["base", "contract_prompt"],
                    "split": "test",
                    "output_tag": "prompt_only",
                    "eval_seeds": [0],
                    "bootstrap": {"enabled": False},
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

    metrics_root = tmp_path / "results" / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": "base__Stem-A__mandated__seed0",
                "scenario_stem": "Stem-A",
                "domain": "healthcare",
                "variant": "mandated",
                "split": "test",
                "method": "base",
                "seed": 0,
                "max_severity": 0,
                "primary_violation_type": "other",
                "first_unsafe_step": None,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
                "audited": False,
                "final_label_source": "judge",
            },
            {
                "run_id": "base__Stem-A__incentivized__seed0",
                "scenario_stem": "Stem-A",
                "domain": "healthcare",
                "variant": "incentivized",
                "split": "test",
                "method": "base",
                "seed": 0,
                "max_severity": 0,
                "primary_violation_type": "other",
                "first_unsafe_step": None,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
                "audited": False,
                "final_label_source": "judge",
            },
            {
                "run_id": "contract_prompt__Stem-A__mandated__seed0",
                "scenario_stem": "Stem-A",
                "domain": "healthcare",
                "variant": "mandated",
                "split": "test",
                "method": "contract_prompt",
                "seed": 0,
                "max_severity": 0,
                "primary_violation_type": "other",
                "first_unsafe_step": None,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
                "audited": False,
                "final_label_source": "judge",
            },
            {
                "run_id": "contract_prompt__Stem-A__incentivized__seed0",
                "scenario_stem": "Stem-A",
                "domain": "healthcare",
                "variant": "incentivized",
                "split": "test",
                "method": "contract_prompt",
                "seed": 0,
                "max_severity": 3,
                "primary_violation_type": "metric_gaming",
                "first_unsafe_step": 2,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
                "audited": False,
                "final_label_source": "judge",
            },
        ]
    ).to_csv(metrics_root / "test_runs_prompt_only.csv", index=False)

    _write_json(
        tmp_path / "results" / "runs" / "test" / "base" / "Stem-A" / "mandated" / "seed0" / "raw_request.json",
        {"backend": "odcv_benchmark"},
    )
    _write_json(
        tmp_path / "results" / "runs" / "test" / "base" / "Stem-A" / "incentivized" / "seed0" / "raw_request.json",
        {"backend": "odcv_benchmark"},
    )
    _write_json(
        tmp_path / "results" / "runs" / "test" / "contract_prompt" / "Stem-A" / "mandated" / "seed0" / "raw_request.json",
        {"backend": "odcv_benchmark"},
    )
    _write_json(
        tmp_path / "results" / "runs" / "test" / "contract_prompt" / "Stem-A" / "incentivized" / "seed0" / "raw_request.json",
        {"backend": "odcv_benchmark"},
    )

    aggregate_metrics_module.aggregate_metrics(cfg_path)

    assert (metrics_root / "test_summary_prompt_only.csv").exists()
    assert (metrics_root / "per_stem_results_prompt_only.csv").exists()
    assert (metrics_root / "bootstrap_ci_prompt_only.csv").exists()
    assert not (metrics_root / "test_summary.csv").exists()
