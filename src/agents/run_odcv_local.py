"""Run one ODCV scenario locally against the configured backend.

Behavior requirements:
- execute exactly one stem/variant/seed per invocation
- preserve raw benchmark prompts and responses
- write trajectory steps and run summary artifacts
- avoid invasive edits inside the benchmark repo

CLI shape:
    python -m src.agents.run_odcv_local --stem <stem> --variant mandated --seed 0 --method base
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from src.utils.io import dump_json, ensure_dir, load_json, load_yaml, maybe_git_commit, relpath, repo_root, utc_now_iso, write_jsonl
from src.utils.manifests import sha256_file
from src.utils.odcv import METHOD_PROMPT_VERSIONS, build_method_messages, load_contract, load_scenario_context, simulate_trajectory
from src.utils.schemas import validate_jsonl_rows, validate_payload


def _resolve_split(stem: str) -> str:
    split_path = repo_root() / "data" / "processed" / "splits" / "split_v1.yaml"
    if not split_path.exists():
        return "adhoc"
    split_payload = load_yaml(split_path)
    for row in split_payload.get("stems", []):
        if row.get("scenario_stem") == stem:
            return row["split"]
    return "adhoc"


def _docker_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return proc.returncode == 0


def _real_mode_supported(method: str) -> bool:
    return bool(method)


def _real_mode_enabled() -> bool:
    return os.environ.get("ODCV_USE_REAL_BENCHMARK", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _resolved_base_url(cfg: dict[str, object]) -> str:
    override = os.environ.get("ODCV_LOCAL_BASE_URL", "").strip()
    if override:
        return override
    return str(cfg["model"]["local_base_url"])


def _resolved_model_id(cfg: dict[str, object]) -> str:
    override = os.environ.get("ODCV_LOCAL_MODEL_ID", "").strip()
    if override:
        return override
    return str(cfg["model"]["main_model_id"])


def _resolved_adapter_id(method: str) -> str | None:
    override = os.environ.get("ODCV_LOCAL_ADAPTER_ID", "").strip()
    if override:
        return override
    return None if method in {"base", "contract_prompt", "refusal_prompt"} else method


def _resolved_prompt_version(method: str) -> str:
    return METHOD_PROMPT_VERSIONS.get(method, f"{method}_v1")


def _split_manifest_sha256() -> str | None:
    split_path = repo_root() / "data" / "processed" / "splits" / "split_v1.yaml"
    return sha256_file(split_path) if split_path.exists() else None


def _guard_against_unexpected_fallback(*, use_real_benchmark: bool, allow_fallback: bool, split: str, method: str) -> None:
    if use_real_benchmark or allow_fallback:
        return
    raise RuntimeError(
        "Real benchmark execution is required here, but it is not enabled. "
        f"Refusing to silently simulate `{method}` on split `{split}`. "
        "Set ODCV_USE_REAL_BENCHMARK=1 and provide a local model path via ODCV_BASE_MODEL_PATH."
    )


def _reset_orchestrator(odcv_root: Path) -> Path:
    orchestrator_dir = odcv_root / "orchestrator_api"
    if orchestrator_dir.exists():
        shutil.rmtree(orchestrator_dir, ignore_errors=True)
    with zipfile.ZipFile(odcv_root / "orchestrator_api.zip", "r") as archive:
        archive.extractall(odcv_root)
    _normalize_orchestrator_files(orchestrator_dir)
    return orchestrator_dir


def _normalize_orchestrator_files(orchestrator_dir: Path) -> None:
    dockerfile_path = orchestrator_dir / "Dockerfile"
    if dockerfile_path.exists():
        dockerfile_text = dockerfile_path.read_text(encoding="utf-8", errors="replace")
        dockerfile_text = dockerfile_text.replace("apt-get install -y bc jq", "apt-get install -y bc jq bash")
        if "sed -i 's/\\r$//' /usr/local/bin/*.sh" not in dockerfile_text:
            dockerfile_text = dockerfile_text.replace(
                "RUN chmod +x /usr/local/bin/*.sh",
                "RUN sed -i 's/\\r$//' /usr/local/bin/*.sh && chmod +x /usr/local/bin/*.sh",
        )
        dockerfile_path.write_text(dockerfile_text, encoding="utf-8", newline="\n")
    task_executor_path = orchestrator_dir / "task_executor_api.py"
    if task_executor_path.exists():
        task_executor_text = task_executor_path.read_text(encoding="utf-8", errors="replace")
        legacy_call = "result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd=cwd)"
        repaired_call = (
            "result = subprocess.run(\n"
            "            [\"/bin/bash\", \"-lc\", command],\n"
            "            shell=False,\n"
            "            capture_output=True,\n"
            "            text=True,\n"
            "            timeout=30,\n"
            "            cwd=cwd,\n"
            "        )"
        )
        if legacy_call in task_executor_text:
            task_executor_text = task_executor_text.replace(legacy_call, repaired_call)
            task_executor_path.write_text(task_executor_text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    for script_path in orchestrator_dir.rglob("*.sh"):
        script_text = script_path.read_text(encoding="utf-8", errors="replace")
        script_path.write_text(script_text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def _copy_scenario_into_orchestrator(scenario_dir: Path, orchestrator_dir: Path) -> None:
    for item in scenario_dir.iterdir():
        target = orchestrator_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    _normalize_orchestrator_files(orchestrator_dir)


def _prepare_real_prompt(orchestrator_dir: Path, method: str, contract: dict[str, object] | None, prompt_paths: dict[str, str]) -> None:
    prompt_path = orchestrator_dir / "prompt.json"
    if not prompt_path.exists():
        return
    prompt_payload = load_json(prompt_path)
    original_system = str(prompt_payload.get("system_prompt", "")).strip()
    if method == "contract_prompt":
        template = (repo_root() / prompt_paths["contract_prompt"]).read_text(encoding="utf-8")
        prompt_payload["system_prompt"] = f"{original_system}\n\n{template.format(contract_json=json.dumps(contract or {}, indent=2))}".strip()
    elif method == "refusal_prompt":
        template = (repo_root() / prompt_paths["refusal_prompt"]).read_text(encoding="utf-8")
        prompt_payload["system_prompt"] = f"{original_system}\n\n{template.format(contract_json=json.dumps(contract or {}, indent=2))}".strip()
    dump_json(prompt_path, prompt_payload)


def _parse_block_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current_key: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if line.startswith("== Step"):
            continue
        for key in ("role", "content", "reason", "call"):
            marker = f"{key}:"
            if line.startswith(marker):
                current_key = key
                fields[key] = line[len(marker) :].lstrip()
                break
        else:
            if current_key:
                fields[current_key] = (fields.get(current_key, "") + "\n" + line).strip("\n")
    return fields


def _parse_tool_content(content: str) -> str:
    stripped = content.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(payload, dict):
        if "stdout" in payload or "stderr" in payload:
            return "\n".join(part for part in (payload.get("stdout", ""), payload.get("stderr", "")) if part).strip()
        if "reason" in payload:
            return str(payload["reason"])
    return stripped


def _assistant_actions_from_fields(fields: dict[str, str]) -> list[tuple[str, str, str]]:
    reason = fields.get("reason", "").strip()
    content = fields.get("content", "").strip()
    call_text = fields.get("call", "").strip()
    if not call_text:
        assistant_text = content or reason or "Action: final\nAction Input: No content recorded."
        return [(assistant_text, "final", assistant_text)]
    try:
        calls = ast.literal_eval(call_text)
    except (SyntaxError, ValueError):
        calls = []
    if not isinstance(calls, list):
        calls = [calls] if calls else []
    if not calls:
        assistant_text = content or reason or "Action: final\nAction Input: No content recorded."
        return [(assistant_text, "final", assistant_text)]

    thought = reason or content or "Proceeding with the next benchmark step."
    actions: list[tuple[str, str, str]] = []
    for raw_call in calls:
        function = raw_call.get("function", {}) if isinstance(raw_call, dict) else {}
        action_type = str(function.get("name", "final"))
        arguments_raw = function.get("arguments", "{}")
        try:
            arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
        except json.JSONDecodeError:
            arguments = {"raw": arguments_raw}
        if not isinstance(arguments, dict):
            arguments = {"raw": arguments}
        action_input = str(arguments.get("command") or arguments.get("reason") or arguments.get("raw") or json.dumps(arguments, sort_keys=True))
        assistant_text = f"Thought: {thought}\nAction: {action_type}\nAction Input: {action_input}"
        actions.append((assistant_text, action_type, action_input))
    return actions


def _resolved_raw_response(raw_response: dict[str, object], steps: list[dict[str, object]]) -> dict[str, object]:
    status = str(raw_response.get("status", ""))
    if status not in {"BUILD_OR_RUN_FAILURE", "missing_result_json"}:
        return raw_response
    if not steps:
        return raw_response
    terminal_step = steps[-1]
    if str(terminal_step.get("parsed_action_type")) != "task_complete":
        return raw_response
    reason = str(terminal_step.get("observation_text") or terminal_step.get("parsed_action") or terminal_step.get("assistant_text") or "").strip()
    return {
        "status": "task_complete_from_messages_record",
        "action": "task_complete",
        "reason": reason,
        "source": "messages_record",
        "benchmark_result_json": raw_response,
    }


def _parse_real_messages_record(
    messages_path: Path,
    *,
    run_id: str,
    stem: str,
    variant: str,
    split: str,
    stage: str,
    method: str,
    seed: int,
    model_id: str,
    contract_version: str | None,
) -> tuple[list[dict[str, object]], str]:
    text = messages_path.read_text(encoding="utf-8", errors="replace")
    raw_blocks = [block for block in text.split("== Step ") if block.strip()]
    blocks = [_parse_block_fields("== Step " + block) for block in raw_blocks]
    steps: list[dict[str, object]] = []
    terminal_output = ""
    step_index = 0
    idx = 0
    while idx < len(blocks):
        fields = blocks[idx]
        if fields.get("role") != "assistant":
            idx += 1
            continue
        actions = _assistant_actions_from_fields(fields)
        next_idx = idx + 1
        tool_observations: list[str] = []
        while next_idx < len(blocks) and blocks[next_idx].get("role") == "tool":
            tool_observations.append(_parse_tool_content(blocks[next_idx].get("content", "")))
            next_idx += 1

        observations_by_action = [""] * len(actions)
        for obs_idx, observation in enumerate(tool_observations):
            action_idx = min(obs_idx, max(len(actions) - 1, 0))
            existing = observations_by_action[action_idx]
            observations_by_action[action_idx] = "\n\n".join(part for part in (existing, observation) if part).strip()

        for action_idx, (assistant_text, action_type, action_input) in enumerate(actions):
            observation = observations_by_action[action_idx] if action_idx < len(observations_by_action) else ""
            step_index += 1
            is_terminal = action_type == "task_complete"
            if is_terminal:
                terminal_output = observation or assistant_text
            steps.append(
                {
                    "run_id": run_id,
                    "scenario_stem": stem,
                    "variant": variant,
                    "split": split,
                    "stage": stage,
                    "method": method,
                    "seed": seed,
                    "benchmark_commit": None,
                    "repo_commit": maybe_git_commit(repo_root()),
                    "model_id": model_id,
                    "adapter_id": _resolved_adapter_id(method),
                    "prompt_version": _resolved_prompt_version(method),
                    "contract_version": contract_version,
                    "judge_version": None,
                    "decoding": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512},
                    "step_idx": step_index,
                    "assistant_text": assistant_text,
                    "parsed_action_type": action_type,
                    "parsed_action": action_input,
                    "observation_text": observation,
                    "files_touched": [],
                    "is_terminal": is_terminal,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "wallclock_ms": None,
                    "timestamp_utc": None,
                }
            )
        idx = next_idx
    if steps and not terminal_output:
        terminal_output = str(steps[-1]["observation_text"] or steps[-1]["assistant_text"])
        steps[-1]["is_terminal"] = True
    return steps, terminal_output


def _run_real_benchmark(
    *,
    odcv_root: Path,
    stem: str,
    variant: str,
    seed: int,
    method: str,
    model_id: str,
    base_url: str,
    api_key: str,
    contract: dict[str, object] | None,
    prompts_cfg: dict[str, str],
    split: str,
    stage: str,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    scenario_dir = odcv_root / f"{variant}_scenarios" / stem
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Real scenario directory not found: {scenario_dir}")
    orchestrator_dir = _reset_orchestrator(odcv_root)
    _copy_scenario_into_orchestrator(scenario_dir, orchestrator_dir)
    _prepare_real_prompt(orchestrator_dir, method, contract, prompts_cfg)

    docker_base_url = (
        base_url.replace("http://localhost", "http://host.docker.internal").replace(
            "http://127.0.0.1",
            "http://host.docker.internal",
        )
    )
    run_id = f"{stem}__{variant}__{method}__seed{seed}"
    experiment_dir = odcv_root / "experiments" / run_id
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir, ignore_errors=True)
    cmd = [
        sys.executable,
        str(odcv_root / "run.py"),
        "-f",
        run_id,
        "-t",
        "--openai-base-url",
        docker_base_url,
        "--openai-api-key",
        api_key,
        "--openai-model",
        model_id,
        "--openai-temperature",
        "0.0" if split == "test" else "0.6",
    ]
    started_at = utc_now_iso()
    proc = subprocess.run(
        cmd,
        cwd=odcv_root,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "NO_TIMESTAMP": "true"},
    )
    finished_at = utc_now_iso()
    messages_path = experiment_dir / "messages_record.txt"
    result_path = experiment_dir / "result.json"
    if not messages_path.exists():
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "Benchmark run failed.")

    parsed_run_id = simulate_trajectory(model_id, stem, variant, split, stage, method, seed, contract)[1]["run_id"]
    steps, terminal_output = _parse_real_messages_record(
        messages_path,
        run_id=parsed_run_id,
        stem=stem,
        variant=variant,
        split=split,
        stage=stage,
        method=method,
        seed=seed,
        model_id=model_id,
        contract_version=None if contract is None else str(contract.get("contract_version")),
    )
    raw_request = {
        "backend": "odcv_benchmark",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    raw_response = load_json(result_path) if result_path.exists() else {"status": "missing_result_json"}
    raw_response = _resolved_raw_response(raw_response, steps)
    run_meta = {
        "run_id": parsed_run_id,
        "scenario_stem": stem,
        "variant": variant,
        "split": split,
        "stage": stage,
        "method": method,
        "seed": seed,
        "benchmark_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=odcv_root, capture_output=True, text=True, check=False).stdout.strip() or None,
        "repo_commit": maybe_git_commit(repo_root()),
        "model_id": model_id,
        "adapter_id": _resolved_adapter_id(method),
        "prompt_version": _resolved_prompt_version(method),
        "contract_version": None if contract is None else contract.get("contract_version"),
        "judge_version": None,
        "judge_prompt_version": None,
        "split_manifest_sha256": _split_manifest_sha256(),
        "start_timestamp_utc": started_at,
        "end_timestamp_utc": finished_at,
        "total_steps": len(steps),
        "terminated_reason": "task_complete" if steps and steps[-1]["parsed_action_type"] == "task_complete" else "benchmark_run",
        "terminal_output": terminal_output,
        "raw_request": raw_request,
        "raw_response": raw_response,
    }
    return steps, run_meta, raw_request


def run_one(
    stem: str,
    variant: str,
    seed: int,
    method: str,
    *,
    split: str | None = None,
    stage: str | None = None,
    config_path: Path | None = None,
    allow_fallback: bool = True,
    real_run_max_attempts: int = 1,
) -> Path:
    """Run one ODCV scenario and return the run directory path."""
    cfg = load_yaml(config_path or repo_root() / "configs" / "defaults.yaml")
    model_cfg = load_yaml(repo_root() / "configs" / "model_qwen25_7b.yaml")
    odcv_root = repo_root() / cfg["paths"]["odcv_root"]
    contract_root = repo_root() / "data" / "processed" / "contracts"
    resolved_split = split or _resolve_split(stem)
    resolved_stage = stage or ("eval" if resolved_split in {"test", "val"} else "data_gen")
    scenario = load_scenario_context(odcv_root, stem, variant)
    contract = load_contract(contract_root, stem)
    messages = build_method_messages(method, scenario, contract, cfg["prompts"])
    model_id = _resolved_model_id(cfg)
    base_url = _resolved_base_url(cfg)
    use_real_benchmark = _real_mode_enabled() and _docker_available() and _real_mode_supported(method)
    _guard_against_unexpected_fallback(
        use_real_benchmark=use_real_benchmark,
        allow_fallback=allow_fallback,
        split=resolved_split,
        method=method,
    )
    if use_real_benchmark:
        last_real_exc: Exception | None = None
        attempts = max(1, int(real_run_max_attempts))
        for attempt in range(1, attempts + 1):
            try:
                steps, run_meta, raw_request = _run_real_benchmark(
                    odcv_root=odcv_root,
                    stem=stem,
                    variant=variant,
                    seed=seed,
                    method=method,
                    model_id=model_id,
                    base_url=base_url,
                    api_key=cfg["model"]["local_api_key"],
                    contract=contract,
                    prompts_cfg=cfg["prompts"],
                    split=resolved_split,
                    stage=resolved_stage,
                )
                break
            except Exception as exc:
                last_real_exc = exc
                if attempt == attempts:
                    if not allow_fallback:
                        raise
                    steps, run_meta, raw_request = simulate_trajectory(
                        model_id=model_id,
                        stem=stem,
                        variant=variant,
                        split=resolved_split,
                        stage=resolved_stage,
                        method=method,
                        seed=seed,
                        contract=contract,
                    )
                continue
    else:
        steps, run_meta, raw_request = simulate_trajectory(
            model_id=model_id,
            stem=stem,
            variant=variant,
            split=resolved_split,
            stage=resolved_stage,
            method=method,
            seed=seed,
            contract=contract,
        )
    run_id = run_meta["run_id"]
    run_dir = ensure_dir(repo_root() / "results" / "runs" / resolved_split / method / stem / variant / f"seed{seed}")
    dump_json(run_dir / "messages.json", {"messages": messages, "scenario_source_files": scenario.source_files})
    dump_json(run_dir / "raw_request.json", raw_request)
    dump_json(run_dir / "raw_response.json", run_meta["raw_response"])
    trajectory_path = write_jsonl(run_dir / "trajectory.jsonl", steps)
    validate_jsonl_rows(steps, "trajectory_step.schema.json")
    run_summary = {
        key: value
        for key, value in run_meta.items()
        if key not in {"raw_request", "raw_response"}
    }
    run_summary["raw_artifact_paths"] = [
        relpath(run_dir / "messages.json"),
        relpath(run_dir / "raw_request.json"),
        relpath(run_dir / "raw_response.json"),
        relpath(trajectory_path),
    ]
    run_summary["derived_artifact_paths"] = []
    run_summary["truncation_warning"] = False
    run_summary["parsing_errors"] = [step.get("parsed_action_type") for step in steps if step.get("parsed_action_type") == "unknown"]
    summary_path = dump_json(run_dir / "run_summary.json", run_summary)
    validate_payload(run_summary, "run_summary.schema.json")
    return summary_path.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", required=True)
    parser.add_argument("--variant", required=True, choices=["mandated", "incentivized"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split")
    parser.add_argument("--stage")
    parser.add_argument("--config", dest="config_path", default=str(repo_root() / "configs" / "defaults.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_path = run_one(
        args.stem,
        args.variant,
        args.seed,
        args.method,
        split=args.split,
        stage=args.stage,
        config_path=Path(args.config_path),
    )
    print(run_path)
