from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import yaml

import src.eval.run_eval as run_eval_module
from src.eval.run_eval import _load_existing_benchmark_run_row, _load_preserved_test_rows, _runs_csv_path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_existing_train_run(
    root: Path,
    *,
    stem: str,
    variant: str,
    seed: int,
    max_severity: int,
    with_derived: bool,
) -> Path:
    run_dir = root / "results" / "runs" / "train" / "base" / stem / variant / f"seed{seed}"
    _write_json(run_dir / "messages.json", {"messages": [], "scenario_source_files": []})
    _write_json(run_dir / "raw_request.json", {"backend": "odcv_benchmark"})
    _write_json(run_dir / "raw_response.json", {"status": "task_complete"})
    _write_jsonl(
        run_dir / "trajectory.jsonl",
        [
            {
                "run_id": f"base__{stem}__{variant}__seed{seed}",
                "step_idx": 1,
                "assistant_text": "Thought: done\nAction: task_complete\nAction Input: done",
                "parsed_action_type": "task_complete",
                "parsed_action": "done",
                "observation_text": "done",
                "is_terminal": True,
            }
        ],
    )
    _write_json(
        run_dir / "run_summary.json",
        {
            "run_id": f"base__{stem}__{variant}__seed{seed}",
            "scenario_stem": stem,
            "variant": variant,
            "split": "train",
            "stage": "data_gen",
            "method": "base",
            "seed": seed,
            "raw_artifact_paths": [],
            "derived_artifact_paths": [],
        },
    )
    if with_derived:
        _write_json(
            run_dir / "judge_output.json",
            {
                "run_id": f"base__{stem}__{variant}__seed{seed}",
                "max_severity": max_severity,
                "primary_violation_type": "data_falsification" if max_severity >= 3 else "other",
                "first_unsafe_step": 1 if max_severity >= 3 else None,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
            },
        )
        _write_json(
            run_dir / "localization_record.json",
            {
                "run_id": f"base__{stem}__{variant}__seed{seed}",
                "first_unsafe_step_candidate": 1 if max_severity >= 3 else None,
                "first_unsafe_step_final": 1 if max_severity >= 3 else None,
                "max_severity": max_severity,
            },
        )
        if max_severity >= 3:
            _write_json(
                run_dir / "rewrites" / "minimal_safe_progress.json",
                {
                    "run_id": f"base__{stem}__{variant}__seed{seed}",
                    "source_step": 1,
                    "rewrite_mode": "minimal_safe_progress",
                    "rejected_output": "unsafe",
                    "chosen_output": "safe",
                    "chosen_audit_label": "SAFE",
                    "chosen_utility_score": 1.0,
                    "filter_keep": True,
                    "filter_reason": "passes_safety_and_format_checks",
                },
            )
    return run_dir


def _write_existing_test_run(
    root: Path,
    *,
    method: str,
    stem: str,
    variant: str,
    seed: int,
    max_severity: int,
) -> Path:
    run_dir = root / "results" / "runs" / "test" / method / stem / variant / f"seed{seed}"
    _write_json(run_dir / "messages.json", {"messages": [], "scenario_source_files": []})
    _write_json(run_dir / "raw_request.json", {"backend": "odcv_benchmark"})
    _write_json(run_dir / "raw_response.json", {"status": "task_complete"})
    _write_jsonl(
        run_dir / "trajectory.jsonl",
        [
            {
                "run_id": f"{method}__{stem}__{variant}__seed{seed}",
                "step_idx": 1,
                "assistant_text": "Thought: done\nAction: task_complete\nAction Input: done",
                "parsed_action_type": "task_complete",
                "parsed_action": "done",
                "observation_text": "done",
                "is_terminal": True,
            }
        ],
    )
    _write_json(
        run_dir / "run_summary.json",
        {
            "run_id": f"{method}__{stem}__{variant}__seed{seed}",
            "scenario_stem": stem,
            "variant": variant,
            "split": "test",
            "stage": "eval",
            "method": method,
            "seed": seed,
            "raw_artifact_paths": [],
            "derived_artifact_paths": [],
        },
    )
    _write_json(
        run_dir / "judge_output.json",
        {
            "run_id": f"{method}__{stem}__{variant}__seed{seed}",
            "max_severity": max_severity,
            "primary_violation_type": "data_falsification" if max_severity >= 3 else "other",
            "first_unsafe_step": 1 if max_severity >= 3 else None,
            "kpi_success": 1,
            "refusal": 0,
            "ambiguous": 0,
        },
    )
    return run_dir


def test_load_preserved_test_rows_keeps_non_rerun_methods(tmp_path: Path) -> None:
    split_manifest_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.write_text(
        yaml.safe_dump(
            {
                "stems": [
                    {"scenario_stem": "Stem-A", "domain": "healthcare", "split": "test"},
                    {"scenario_stem": "Stem-B", "domain": "finance", "split": "test"},
                ]
            }
        ),
        encoding="utf-8",
    )

    base_run = tmp_path / "results" / "runs" / "test" / "base" / "Stem-A" / "mandated" / "seed0"
    base_run.mkdir(parents=True, exist_ok=True)
    (base_run / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": "base_run",
                "scenario_stem": "Stem-A",
                "variant": "mandated",
                "split": "test",
                "method": "base",
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    (base_run / "judge_output.json").write_text(
        json.dumps(
            {
                "max_severity": 4,
                "primary_violation_type": "data_falsification",
                "first_unsafe_step": 3,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
            }
        ),
        encoding="utf-8",
    )

    rerun_run = tmp_path / "results" / "runs" / "test" / "contract_prompt" / "Stem-B" / "mandated" / "seed0"
    rerun_run.mkdir(parents=True, exist_ok=True)
    (rerun_run / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": "rerun_run",
                "scenario_stem": "Stem-B",
                "variant": "mandated",
                "split": "test",
                "method": "contract_prompt",
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    (rerun_run / "judge_output.json").write_text(
        json.dumps(
            {
                "max_severity": 0,
                "primary_violation_type": "other",
                "first_unsafe_step": None,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
            }
        ),
        encoding="utf-8",
    )

    rows = _load_preserved_test_rows(["contract_prompt"], root=tmp_path)

    assert rows == [
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


def test_run_eval_train_resume_reuses_existing_real_run(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "configs" / "data_gen_train.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump({"eval": {"methods": ["base"], "split": "train", "resume_existing": True, "eval_seeds": [0]}}),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "defaults.yaml").write_text(
        yaml.safe_dump({"model": {"main_model_id": "Qwen/Qwen2.5-7B-Instruct"}, "seeds": {"bootstrap_seed": 12345}}),
        encoding="utf-8",
    )
    split_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        yaml.safe_dump({"stems": [{"scenario_stem": "Stem-A", "domain": "healthcare", "split": "train"}]}),
        encoding="utf-8",
    )
    _write_existing_train_run(tmp_path, stem="Stem-A", variant="mandated", seed=0, max_severity=0, with_derived=True)
    _write_existing_train_run(tmp_path, stem="Stem-A", variant="incentivized", seed=0, max_severity=0, with_derived=True)

    monkeypatch.setattr(run_eval_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(run_eval_module, "selected_adapter_path", lambda method: None)

    @contextmanager
    def _fake_server(**_: object):
        yield

    monkeypatch.setattr(run_eval_module, "managed_server_for_method", _fake_server)

    def _unexpected_run_one(*args: object, **kwargs: object) -> Path:
        raise AssertionError("run_one should not be called for a complete existing real train run")

    monkeypatch.setattr(run_eval_module, "run_one", _unexpected_run_one)

    run_eval_module.run_eval(cfg_path)

    rows = pd.read_csv(tmp_path / "results" / "metrics" / "train_runs.csv")
    assert len(rows) == 2
    assert set(rows["variant"]) == {"mandated", "incentivized"}
    mandated = rows[rows["variant"] == "mandated"].iloc[0]
    assert mandated["scenario_stem"] == "Stem-A"
    assert mandated["seed"] == 0


def test_run_eval_train_resume_backfills_missing_derived_artifacts(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "configs" / "data_gen_train.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump({"eval": {"methods": ["base"], "split": "train", "resume_existing": True, "eval_seeds": [0]}}),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "defaults.yaml").write_text(
        yaml.safe_dump({"model": {"main_model_id": "Qwen/Qwen2.5-7B-Instruct"}, "seeds": {"bootstrap_seed": 12345}}),
        encoding="utf-8",
    )
    split_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        yaml.safe_dump({"stems": [{"scenario_stem": "Stem-A", "domain": "healthcare", "split": "train"}]}),
        encoding="utf-8",
    )
    run_dir = _write_existing_train_run(tmp_path, stem="Stem-A", variant="mandated", seed=0, max_severity=4, with_derived=False)
    _write_existing_train_run(tmp_path, stem="Stem-A", variant="incentivized", seed=0, max_severity=0, with_derived=True)

    monkeypatch.setattr(run_eval_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(run_eval_module, "selected_adapter_path", lambda method: None)

    @contextmanager
    def _fake_server(**_: object):
        yield

    monkeypatch.setattr(run_eval_module, "managed_server_for_method", _fake_server)
    monkeypatch.setattr(run_eval_module, "run_one", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_one should not rerun")))

    calls = {"judge": 0, "localize": 0, "rewrite": 0}

    def _fake_judge(run_path: Path) -> Path:
        calls["judge"] += 1
        _write_json(
            run_path / "judge_output.json",
            {
                "run_id": "base__Stem-A__mandated__seed0",
                "max_severity": 4,
                "primary_violation_type": "data_falsification",
                "first_unsafe_step": 1,
                "kpi_success": 1,
                "refusal": 0,
                "ambiguous": 0,
            },
        )
        return run_path / "judge_output.json"

    def _fake_localize(run_path: Path) -> Path:
        calls["localize"] += 1
        _write_json(
            run_path / "localization_record.json",
            {
                "run_id": "base__Stem-A__mandated__seed0",
                "first_unsafe_step_candidate": 1,
                "first_unsafe_step_final": 1,
                "max_severity": 4,
            },
        )
        return run_path / "localization_record.json"

    def _fake_rewrite(run_path: Path) -> int:
        calls["rewrite"] += 1
        _write_json(
            run_path / "rewrites" / "minimal_safe_progress.json",
            {
                "run_id": "base__Stem-A__mandated__seed0",
                "source_step": 1,
                "rewrite_mode": "minimal_safe_progress",
                "rejected_output": "unsafe",
                "chosen_output": "safe",
                "chosen_audit_label": "SAFE",
                "chosen_utility_score": 1.0,
                "filter_keep": True,
                "filter_reason": "passes_safety_and_format_checks",
            },
        )
        return 1

    monkeypatch.setattr(run_eval_module, "judge_run", _fake_judge)
    monkeypatch.setattr(run_eval_module, "localize_first_unsafe_step", _fake_localize)
    monkeypatch.setattr(run_eval_module, "rewrite_run", _fake_rewrite)

    run_eval_module.run_eval(cfg_path)

    assert calls == {"judge": 1, "localize": 1, "rewrite": 1}
    assert (run_dir / "judge_output.json").exists()
    assert (run_dir / "localization_record.json").exists()
    assert (run_dir / "rewrites" / "minimal_safe_progress.json").exists()


def test_load_existing_benchmark_run_row_rewrites_incomplete_rewrite_sets(tmp_path: Path, monkeypatch) -> None:
    run_dir = _write_existing_train_run(
        tmp_path,
        stem="Stem-A",
        variant="mandated",
        seed=0,
        max_severity=4,
        with_derived=True,
    )

    rewrite_calls: list[Path] = []

    def _fake_rewrite(run_path: Path) -> int:
        rewrite_calls.append(run_path)
        for name in ("minimal_safe_progress", "safe_alternative_plan", "explicit_infeasibility_or_escalation"):
            _write_json(
                run_path / "rewrites" / f"{name}.json",
                {
                    "run_id": "base__Stem-A__mandated__seed0",
                    "source_step": 1,
                    "rewrite_mode": name,
                    "rejected_output": "unsafe",
                    "chosen_output": "safe",
                    "chosen_audit_label": "SAFE",
                    "chosen_utility_score": 1.0,
                    "filter_keep": True,
                    "filter_reason": "passes_safety_and_format_checks",
                },
            )
        return 3

    monkeypatch.setattr(run_eval_module, "rewrite_run", _fake_rewrite)

    row = _load_existing_benchmark_run_row(
        run_split="train",
        method="base",
        stem="Stem-A",
        variant="mandated",
        seed=0,
        target_split="train",
        domain="healthcare",
        root=tmp_path,
    )

    assert row is not None
    assert rewrite_calls == [run_dir]
    assert len(sorted((run_dir / "rewrites").glob("*.json"))) == 3


def test_run_eval_train_resume_removes_stale_nonreal_run_before_rerun(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "configs" / "data_gen_train.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump({"eval": {"methods": ["base"], "split": "train", "resume_existing": True, "eval_seeds": [0]}}),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "defaults.yaml").write_text(
        yaml.safe_dump({"model": {"main_model_id": "Qwen/Qwen2.5-7B-Instruct"}, "seeds": {"bootstrap_seed": 12345}}),
        encoding="utf-8",
    )
    split_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        yaml.safe_dump({"stems": [{"scenario_stem": "Stem-A", "domain": "healthcare", "split": "train"}]}),
        encoding="utf-8",
    )
    run_dir = _write_existing_train_run(tmp_path, stem="Stem-A", variant="mandated", seed=0, max_severity=0, with_derived=False)
    _write_json(run_dir / "raw_request.json", {"backend": "simulated"})
    _write_existing_train_run(tmp_path, stem="Stem-A", variant="incentivized", seed=0, max_severity=0, with_derived=True)

    monkeypatch.setattr(run_eval_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(run_eval_module, "selected_adapter_path", lambda method: None)

    @contextmanager
    def _fake_server(**_: object):
        yield

    monkeypatch.setattr(run_eval_module, "managed_server_for_method", _fake_server)

    executed_variants: list[str] = []

    def _fake_execute_run(
        *,
        stem: str,
        variant: str,
        seed: int,
        method: str,
        target_split: str,
        domain: str,
        real_run_max_attempts: int = 1,
    ) -> dict[str, object]:
        if variant == "mandated":
            assert not run_dir.exists()
        executed_variants.append(variant)
        return {
            "run_id": f"{method}__{stem}__{variant}__seed{seed}",
            "scenario_stem": stem,
            "domain": domain,
            "variant": variant,
            "split": target_split,
            "method": method,
            "seed": seed,
            "max_severity": 0,
            "primary_violation_type": "other",
            "first_unsafe_step": None,
            "kpi_success": 1,
            "refusal": 0,
            "ambiguous": 0,
            "audited": False,
            "final_label_source": "judge",
        }

    monkeypatch.setattr(run_eval_module, "_execute_run", _fake_execute_run)

    run_eval_module.run_eval(cfg_path)

    assert executed_variants == ["mandated"]


def test_run_eval_prompt_only_test_uses_config_scoped_validation_and_output_tag(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "configs" / "eval_prompt_only.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "methods": ["base", "contract_prompt", "refusal_prompt"],
                    "split": "test",
                    "output_tag": "prompt_only",
                    "eval_seeds": [0],
                    "bootstrap": {"enabled": False},
                    "manual_audit": {
                        "audit_severity_threshold_band": [2, 3],
                        "audit_threshold_crossings": True,
                        "audit_random_fraction": 0.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "defaults.yaml").write_text(
        yaml.safe_dump({"model": {"main_model_id": "Qwen/Qwen2.5-7B-Instruct"}, "seeds": {"bootstrap_seed": 12345}}),
        encoding="utf-8",
    )
    split_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        yaml.safe_dump({"stems": [{"scenario_stem": "Stem-A", "domain": "healthcare", "split": "test"}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_eval_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(run_eval_module, "selected_adapter_path", lambda method: None)

    @contextmanager
    def _fake_server(**_: object):
        yield

    monkeypatch.setattr(run_eval_module, "managed_server_for_method", _fake_server)

    def _fake_execute_run(
        *,
        stem: str,
        variant: str,
        seed: int,
        method: str,
        target_split: str,
        domain: str,
        real_run_max_attempts: int = 1,
    ) -> dict[str, object]:
        run_dir = tmp_path / "results" / "runs" / "test" / method / stem / variant / f"seed{seed}"
        _write_json(run_dir / "raw_request.json", {"backend": "odcv_benchmark"})
        return {
            "run_id": f"{method}__{stem}__{variant}__seed{seed}",
            "scenario_stem": stem,
            "domain": domain,
            "variant": variant,
            "split": target_split,
            "method": method,
            "seed": seed,
            "max_severity": 0,
            "primary_violation_type": "other",
            "first_unsafe_step": None,
            "kpi_success": 1,
            "refusal": 0,
            "ambiguous": 0,
            "audited": False,
            "final_label_source": "judge",
        }

    monkeypatch.setattr(run_eval_module, "_execute_run", _fake_execute_run)

    aggregated: dict[str, Path] = {}

    def _fake_aggregate_metrics(config_path: Path) -> None:
        aggregated["config_path"] = config_path

    monkeypatch.setattr(run_eval_module, "aggregate_metrics", _fake_aggregate_metrics)

    run_eval_module.run_eval(cfg_path)

    assert (tmp_path / "results" / "metrics" / "test_runs_prompt_only.csv").exists()
    assert not (tmp_path / "results" / "metrics" / "test_runs.csv").exists()
    assert (tmp_path / "results" / "audits" / "manual_audit_queue_prompt_only.csv").exists()
    assert (tmp_path / "results" / "audits" / "manual_audit_manifest_prompt_only.json").exists()
    assert aggregated["config_path"] == cfg_path
    assert _runs_csv_path("test", "prompt_only") == tmp_path / "results" / "metrics" / "test_runs_prompt_only.csv"


def test_run_eval_tagged_test_resume_reuses_existing_rows_without_preserving_other_methods(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "configs" / "eval_sparse_pilots.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "methods": ["sft_chosen", "fus_dpo"],
                    "split": "test",
                    "output_tag": "sparse_pilots",
                    "resume_existing": True,
                    "eval_seeds": [0],
                    "bootstrap": {"enabled": False},
                    "manual_audit": {
                        "audit_severity_threshold_band": [2, 3],
                        "audit_threshold_crossings": True,
                        "audit_random_fraction": 0.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "defaults.yaml").write_text(
        yaml.safe_dump({"model": {"main_model_id": "Qwen/Qwen2.5-7B-Instruct"}, "seeds": {"bootstrap_seed": 12345}}),
        encoding="utf-8",
    )
    split_path = tmp_path / "data" / "processed" / "splits" / "split_v1.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        yaml.safe_dump({"stems": [{"scenario_stem": "Stem-A", "domain": "healthcare", "split": "test"}]}),
        encoding="utf-8",
    )

    _write_existing_test_run(tmp_path, method="base", stem="Stem-A", variant="mandated", seed=0, max_severity=4)
    _write_existing_test_run(tmp_path, method="sft_chosen", stem="Stem-A", variant="mandated", seed=0, max_severity=0)
    _write_existing_test_run(
        tmp_path, method="sft_chosen", stem="Stem-A", variant="incentivized", seed=0, max_severity=0
    )
    _write_existing_test_run(tmp_path, method="fus_dpo", stem="Stem-A", variant="mandated", seed=0, max_severity=0)
    _write_existing_test_run(
        tmp_path, method="fus_dpo", stem="Stem-A", variant="incentivized", seed=0, max_severity=0
    )

    monkeypatch.setattr(run_eval_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(run_eval_module, "selected_adapter_path", lambda method: None)

    @contextmanager
    def _fake_server(**_: object):
        yield

    monkeypatch.setattr(run_eval_module, "managed_server_for_method", _fake_server)
    monkeypatch.setattr(
        run_eval_module,
        "run_one",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_one should not rerun existing tagged test runs")),
    )

    aggregated: dict[str, Path] = {}

    def _fake_aggregate_metrics(config_path: Path) -> None:
        aggregated["config_path"] = config_path

    monkeypatch.setattr(run_eval_module, "aggregate_metrics", _fake_aggregate_metrics)

    run_eval_module.run_eval(cfg_path)

    out_csv = tmp_path / "results" / "metrics" / "test_runs_sparse_pilots.csv"
    assert out_csv.exists()
    rows = pd.read_csv(out_csv)
    assert set(rows["method"]) == {"sft_chosen", "fus_dpo"}
    assert "base" not in set(rows["method"])
    assert len(rows) == 4
    assert aggregated["config_path"] == cfg_path
