"""Repair existing real-benchmark run artifacts after parser/judge fixes.

CLI shape:
    python -m src.eval.repair_real_artifacts --split test --methods base contract_prompt refusal_prompt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.run_odcv_local import _parse_real_messages_record, _resolved_prompt_version, _resolved_raw_response, _split_manifest_sha256
from src.counterfactuals.rewrite_first_unsafe import rewrite_run
from src.judging.judge_trajectory import judge_run
from src.judging.localize_first_unsafe import localize_first_unsafe_step
from src.utils.io import dump_json, load_json, repo_root, write_jsonl
from src.utils.schemas import validate_jsonl_rows, validate_payload


def _experiment_dir(run_summary: dict[str, object]) -> Path:
    return (
        repo_root()
        / "external"
        / "ODCV-Bench"
        / "experiments"
        / f"{run_summary['scenario_stem']}__{run_summary['variant']}__{run_summary['method']}__seed{int(run_summary['seed'])}"
    )


def repair_run_dir(run_dir: Path) -> bool:
    run_summary_path = run_dir / "run_summary.json"
    raw_request_path = run_dir / "raw_request.json"
    if not (run_summary_path.exists() and raw_request_path.exists()):
        return False

    raw_request = load_json(raw_request_path)
    if raw_request.get("backend") != "odcv_benchmark":
        return False

    summary = load_json(run_summary_path)
    raw_response_path = run_dir / "raw_response.json"
    messages_path = _experiment_dir(summary) / "messages_record.txt"
    if not messages_path.exists():
        raise FileNotFoundError(f"Missing benchmark messages_record for {run_dir}: {messages_path}")

    steps, terminal_output = _parse_real_messages_record(
        messages_path,
        run_id=str(summary["run_id"]),
        stem=str(summary["scenario_stem"]),
        variant=str(summary["variant"]),
        split=str(summary["split"]),
        stage=str(summary["stage"]),
        method=str(summary["method"]),
        seed=int(summary["seed"]),
        model_id=str(summary["model_id"]),
        contract_version=None if summary.get("contract_version") is None else str(summary["contract_version"]),
    )

    trajectory_path = write_jsonl(run_dir / "trajectory.jsonl", steps)
    validate_jsonl_rows(steps, "trajectory_step.schema.json")

    if raw_response_path.exists():
        raw_response = load_json(raw_response_path)
        dump_json(raw_response_path, _resolved_raw_response(raw_response, steps))

    summary["total_steps"] = len(steps)
    summary["terminal_output"] = terminal_output
    summary["prompt_version"] = _resolved_prompt_version(str(summary["method"]))
    summary.setdefault("judge_prompt_version", None)
    summary["split_manifest_sha256"] = summary.get("split_manifest_sha256") or _split_manifest_sha256()
    summary["parsing_errors"] = [step.get("parsed_action_type") for step in steps if step.get("parsed_action_type") == "unknown"]
    dump_json(run_summary_path, summary)
    validate_payload(summary, "run_summary.schema.json")

    judge_path = judge_run(run_dir)
    localized_step = localize_first_unsafe_step(run_dir)
    if str(summary.get("split")) in {"train", "val"}:
        judge_output = load_json(judge_path)
        if int(judge_output.get("max_severity", 0)) >= 3 and localized_step is not None:
            rewrite_run(run_dir)
    return trajectory_path.exists()


def repair_split(split: str, methods: list[str]) -> dict[str, int]:
    results_root = repo_root() / "results" / "runs" / split
    repaired: dict[str, int] = {}
    for method in methods:
        repaired_count = 0
        for run_dir in sorted((results_root / method).glob("*/*/seed*")):
            repaired_count += int(repair_run_dir(run_dir))
        repaired[method] = repaired_count
    return repaired


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    repaired = repair_split(args.split, args.methods)
    for method, count in repaired.items():
        print(f"{method}: {count}")
