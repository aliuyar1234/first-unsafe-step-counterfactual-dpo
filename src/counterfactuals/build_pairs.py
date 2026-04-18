"""Build DPO preference pairs from filtered rewrites."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, load_yaml, read_jsonl, repo_root, write_jsonl
from src.utils.manifests import sha256_file
from src.utils.odcv import build_pair_prompt, load_contract, load_scenario_context, load_split_manifest
from src.utils.schemas import validate_jsonl_rows
from src.utils.scoring import looks_like_refusal


def _history_prefix(trajectory: list[dict[str, object]], source_step: int) -> str:
    lines: list[str] = []
    for step in trajectory:
        idx = int(step["step_idx"])
        if idx >= source_step:
            break
        lines.append(f"[{idx}] {step['assistant_text']}")
        observation = str(step.get("observation_text", "")).strip()
        if observation:
            lines.append(f"Observation: {observation}")
    return "\n".join(lines).strip()


def _pair_generation_policy(cfg: dict[str, Any]) -> tuple[set[str], set[str], dict[str, str]]:
    pair_cfg = cfg.get("pair_generation", {})
    methods = {str(item) for item in pair_cfg.get("allowed_source_methods", ["base"])}
    stages = {str(item) for item in pair_cfg.get("allowed_source_stages", ["data_gen"])}
    required_backend_by_split = {
        str(split): str(backend)
        for split, backend in dict(pair_cfg.get("required_backend_by_split", {"train": "odcv_benchmark", "val": "odcv_benchmark"})).items()
    }
    return methods, stages, required_backend_by_split


def build_pairs(config_path: Path) -> None:
    """Build train/val preference pair datasets."""
    cfg = load_yaml(config_path)
    split_manifest_path = repo_root() / "data" / "processed" / "splits" / "split_v1.yaml"
    split_manifest = load_split_manifest(split_manifest_path)
    split_manifest_hash = sha256_file(split_manifest_path)
    split_lookup = {row["scenario_stem"]: row["split"] for row in split_manifest["stems"]}
    odcv_root = repo_root() / cfg["paths"]["odcv_root"]
    contract_root = repo_root() / "data" / "processed" / "contracts"
    pairs_root = ensure_dir(repo_root() / "data" / "processed" / "pairs")
    metrics_root = ensure_dir(repo_root() / "results" / "metrics")
    allowed_methods, allowed_stages, required_backend_by_split = _pair_generation_policy(cfg)

    pairs_by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    refusal_only_kept = {"train": 0, "val": 0}
    stats_by_group: dict[tuple[str, str, str], dict[str, object]] = {}
    run_dirs = sorted(path.parent for path in (repo_root() / "results" / "runs").rglob("run_summary.json"))

    for run_dir in run_dirs:
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        stem = summary["scenario_stem"]
        split = split_lookup.get(stem)
        if split not in {"train", "val"}:
            continue
        if str(summary.get("split")) != split:
            continue
        if str(summary.get("method")) not in allowed_methods:
            continue
        if str(summary.get("stage")) not in allowed_stages:
            continue
        raw_request_path = run_dir / "raw_request.json"
        if not raw_request_path.exists():
            raise FileNotFoundError(f"Missing raw_request.json for pair-generation source run: {run_dir}")
        raw_request = json.loads(raw_request_path.read_text(encoding="utf-8"))
        required_backend = required_backend_by_split.get(split)
        if required_backend and str(raw_request.get("backend")) != required_backend:
            raise RuntimeError(
                f"Pair-generation source run must use backend `{required_backend}` but found `{raw_request.get('backend')}`: {run_dir}"
            )
        judge_path = run_dir / "judge_output.json"
        localization_path = run_dir / "localization_record.json"
        if not judge_path.exists() or not localization_path.exists():
            raise FileNotFoundError(f"Missing judge/localization artifacts for pair-generation source run: {run_dir}")
        judge_output = json.loads(judge_path.read_text(encoding="utf-8"))
        stats_key = (split, stem, str(summary["variant"]))
        stats_row = stats_by_group.setdefault(
            stats_key,
            {
                "split": split,
                "scenario_stem": stem,
                "variant": summary["variant"],
                "num_runs_sampled": 0,
                "num_runs_severity_ge_3": 0,
                "num_raw_rewrite_candidates": 0,
                "num_kept_rewrites": 0,
                "num_pairs_kept": 0,
                "num_refusal_pairs_kept": 0,
            },
        )
        stats_row["num_runs_sampled"] = int(stats_row["num_runs_sampled"]) + 1
        if judge_output["max_severity"] < 3:
            continue
        stats_row["num_runs_severity_ge_3"] = int(stats_row["num_runs_severity_ge_3"]) + 1
        localization = json.loads(localization_path.read_text(encoding="utf-8"))
        source_step = localization["first_unsafe_step_final"]
        if source_step is None:
            continue
        rewrite_paths = sorted((run_dir / "rewrites").glob("*.json"))
        if not rewrite_paths:
            raise FileNotFoundError(f"Missing rewrites for pair-generation source run: {run_dir}")
        rewrites = [json.loads(path.read_text(encoding="utf-8")) for path in rewrite_paths]
        kept = [rewrite for rewrite in rewrites if rewrite["filter_keep"]]
        kept.sort(key=lambda row: (-float(row["chosen_utility_score"]), str(row["rewrite_mode"])))
        stats_row["num_raw_rewrite_candidates"] = int(stats_row["num_raw_rewrite_candidates"]) + len(rewrites)
        stats_row["num_kept_rewrites"] = int(stats_row["num_kept_rewrites"]) + len(kept)
        trajectory = read_jsonl(run_dir / "trajectory.jsonl")
        contract = load_contract(contract_root, stem)
        if contract is None:
            continue
        scenario = load_scenario_context(odcv_root, stem, summary["variant"])
        prompt = build_pair_prompt(scenario, contract, _history_prefix(trajectory, int(source_step)))
        run_pairs = 0
        for rewrite in kept[:2]:
            is_refusal = looks_like_refusal(str(rewrite["chosen_output"]))
            current_total = max(1, len(pairs_by_split[split]))
            if is_refusal and refusal_only_kept[split] / current_total > 0.30:
                continue
            pair = {
                "pair_id": f"pair__{stem}__{summary['variant']}__{summary['run_id']}__step{source_step}__mode_{rewrite['rewrite_mode']}",
                "run_id": summary["run_id"],
                "scenario_stem": stem,
                "variant": summary["variant"],
                "split": split,
                "source_step": int(source_step),
                "source_method": str(summary["method"]),
                "source_stage": str(summary["stage"]),
                "source_backend": str(raw_request.get("backend")),
                "judge_version": summary.get("judge_version"),
                "contract_version": summary.get("contract_version"),
                "split_manifest_sha256": split_manifest_hash,
                "source_artifact_paths": list(summary.get("raw_artifact_paths", [])) + list(summary.get("derived_artifact_paths", [])),
                "prompt": prompt,
                "chosen": rewrite["chosen_output"],
                "rejected": rewrite["rejected_output"],
                "chosen_mode": rewrite["rewrite_mode"],
                "pair_version": "pair_v1",
            }
            pairs_by_split[split].append(pair)
            if is_refusal:
                refusal_only_kept[split] += 1
                stats_row["num_refusal_pairs_kept"] = int(stats_row["num_refusal_pairs_kept"]) + 1
            run_pairs += 1
        stats_row["num_pairs_kept"] = int(stats_row["num_pairs_kept"]) + run_pairs

    for split, rows in pairs_by_split.items():
        write_jsonl(pairs_root / f"{split}.jsonl", rows)
        validate_jsonl_rows(rows, "preference_pair.schema.json")

    stats_rows: list[dict[str, object]] = []
    for stats_row in stats_by_group.values():
        pair_count = max(1, int(stats_row["num_pairs_kept"]))
        stats_rows.append(
            {
                "split": stats_row["split"],
                "scenario_stem": stats_row["scenario_stem"],
                "variant": stats_row["variant"],
                "num_runs_sampled": stats_row["num_runs_sampled"],
                "num_runs_severity_ge_3": stats_row["num_runs_severity_ge_3"],
                "num_raw_rewrite_candidates": stats_row["num_raw_rewrite_candidates"],
                "num_kept_rewrites": stats_row["num_kept_rewrites"],
                "num_pairs_kept": stats_row["num_pairs_kept"],
                "refusal_only_kept_fraction": round(int(stats_row["num_refusal_pairs_kept"]) / pair_count, 4) if int(stats_row["num_pairs_kept"]) else 0.0,
            }
        )
    stats_rows.sort(key=lambda row: (str(row["split"]), str(row["scenario_stem"]), str(row["variant"])))

    with (metrics_root / "pair_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
        )
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_pairs(Path(args.config))
