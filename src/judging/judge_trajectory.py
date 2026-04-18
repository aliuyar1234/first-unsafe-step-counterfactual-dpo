"""Judge full ODCV trajectories and write strict JSON outputs.

Behavior requirements:
- load numbered trajectory from JSONL
- load matching frozen contract
- call the selected frozen judge backend (currently deterministic executed-action judging)
- validate output against `schemas/judge_output.schema.json`
- trigger manual-audit routing for threshold-critical ambiguity
- preserve raw judge responses

CLI shape:
    python -m src.judging.judge_trajectory --run-dir <run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.io import dump_json, read_jsonl, relpath, repo_root
from src.utils.odcv import load_contract
from src.utils.schemas import validate_payload
from src.utils.scoring import judge_trajectory_records

JUDGE_BACKEND_VERSION = "deterministic_executed_action_judge_v5"
JUDGE_PROMPT_VERSION = "trajectory_judge_v1"


def judge_run(run_dir: Path) -> Path:
    """Judge one run and write `judge_output.json`.

    Returns:
        Path to the written judge output file.
    """
    run_dir = run_dir if run_dir.is_absolute() else repo_root() / run_dir
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    trajectory = read_jsonl(run_dir / "trajectory.jsonl")
    contract = load_contract(repo_root() / "data" / "processed" / "contracts", summary["scenario_stem"]) or {}
    judge_output = judge_trajectory_records(summary["run_id"], trajectory)
    validate_payload(judge_output, "judge_output.schema.json")
    output_path = dump_json(run_dir / "judge_output.json", judge_output)
    dump_json(
        run_dir / "raw_judge_response.json",
        {
            "backend": JUDGE_BACKEND_VERSION,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "prompt_path": relpath(repo_root() / "prompts" / "trajectory_judge.txt"),
            "contract": contract,
            "run_summary": summary,
            "trajectory": trajectory,
            "judge_output": judge_output,
        },
    )

    summary["judge_version"] = judge_output["judge_version"]
    summary["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    derived = set(summary.get("derived_artifact_paths", []))
    derived.update({relpath(output_path), relpath(run_dir / "raw_judge_response.json")})
    summary["derived_artifact_paths"] = sorted(derived)
    dump_json(run_dir / "run_summary.json", summary)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(judge_run(Path(args.run_dir)))
