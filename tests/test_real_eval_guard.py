from __future__ import annotations

from pathlib import Path

import pytest

import src.agents.run_odcv_local as run_odcv_local_module
from src.agents.run_odcv_local import _guard_against_unexpected_fallback


def test_guard_against_unexpected_fallback_raises_when_real_mode_required() -> None:
    with pytest.raises(RuntimeError, match="Real benchmark execution is required"):
        _guard_against_unexpected_fallback(
            use_real_benchmark=False,
            allow_fallback=False,
            split="test",
            method="fus_dpo",
        )


def test_guard_against_unexpected_fallback_allows_enabled_real_mode() -> None:
    _guard_against_unexpected_fallback(
        use_real_benchmark=True,
        allow_fallback=False,
        split="test",
        method="fus_dpo",
    )


def test_guard_against_unexpected_fallback_allows_explicit_fallback() -> None:
    _guard_against_unexpected_fallback(
        use_real_benchmark=False,
        allow_fallback=True,
        split="train",
        method="base",
    )


def test_run_one_retries_real_benchmark_for_train_runs(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "defaults.yaml").write_text(
        "paths:\n  odcv_root: external/ODCV-Bench\nmodel:\n  main_model_id: Qwen/Qwen2.5-7B-Instruct\n  local_base_url: http://localhost:8000/v1\n  local_api_key: local\nprompts: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "model_qwen25_7b.yaml").write_text("{}", encoding="utf-8")
    (tmp_path / "data" / "processed" / "contracts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "external" / "ODCV-Bench").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_odcv_local_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(run_odcv_local_module, "load_scenario_context", lambda *args, **kwargs: type("Scenario", (), {"source_files": []})())
    monkeypatch.setattr(run_odcv_local_module, "load_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_odcv_local_module, "build_method_messages", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_odcv_local_module, "_real_mode_enabled", lambda: True)
    monkeypatch.setattr(run_odcv_local_module, "_docker_available", lambda: True)
    monkeypatch.setattr(run_odcv_local_module, "_real_mode_supported", lambda method: True)
    monkeypatch.setattr(run_odcv_local_module, "_resolved_model_id", lambda cfg: "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setattr(run_odcv_local_module, "_resolved_base_url", lambda cfg: "http://localhost:8000/v1")
    monkeypatch.setattr(run_odcv_local_module, "_resolved_adapter_id", lambda method: None)
    monkeypatch.setattr(run_odcv_local_module, "_resolved_prompt_version", lambda method: "base_v1")
    monkeypatch.setattr(run_odcv_local_module, "_split_manifest_sha256", lambda: "hash")
    monkeypatch.setattr(run_odcv_local_module, "maybe_git_commit", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_odcv_local_module, "validate_jsonl_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_odcv_local_module, "validate_payload", lambda *args, **kwargs: None)

    attempts = {"count": 0}

    def _fake_run_real_benchmark(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient benchmark failure")
        return (
            [
                {
                    "step_idx": 1,
                    "assistant_text": "done",
                    "parsed_action_type": "task_complete",
                    "parsed_action": "done",
                    "observation_text": "done",
                    "is_terminal": True,
                }
            ],
            {
                "run_id": "run_001",
                "scenario_stem": "Stem-A",
                "variant": "mandated",
                "split": "train",
                "stage": "data_gen",
                "method": "base",
                "seed": 0,
                "benchmark_commit": None,
                "repo_commit": None,
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "adapter_id": None,
                "prompt_version": "base_v1",
                "contract_version": None,
                "judge_version": None,
                "judge_prompt_version": None,
                "split_manifest_sha256": "hash",
                "start_timestamp_utc": "start",
                "end_timestamp_utc": "end",
                "total_steps": 1,
                "terminated_reason": "task_complete",
                "terminal_output": "done",
                "raw_request": {"backend": "odcv_benchmark", "returncode": 0, "stdout": "", "stderr": ""},
                "raw_response": {"status": "task_complete"},
            },
            {"backend": "odcv_benchmark", "returncode": 0, "stdout": "", "stderr": ""},
        )

    monkeypatch.setattr(run_odcv_local_module, "_run_real_benchmark", _fake_run_real_benchmark)

    run_dir = run_odcv_local_module.run_one(
        "Stem-A",
        "mandated",
        0,
        "base",
        split="train",
        stage="data_gen",
        config_path=tmp_path / "configs" / "defaults.yaml",
        allow_fallback=False,
        real_run_max_attempts=3,
    )

    assert attempts["count"] == 3
    assert (run_dir / "run_summary.json").exists()
