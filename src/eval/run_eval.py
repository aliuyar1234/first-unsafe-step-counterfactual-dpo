"""Run held-out evaluation for all configured methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from src.agents.server_runtime import managed_server_for_method, selected_adapter_path
from src.agents.run_odcv_local import run_one
from src.counterfactuals.rewrite_first_unsafe import rewrite_run
from src.eval.aggregate_metrics import aggregate_metrics
from src.eval.manual_audit import build_manual_audit_queue
from src.judging.judge_trajectory import judge_run
from src.judging.localize_first_unsafe import localize_first_unsafe_step
from src.utils.io import load_yaml, repo_root
from src.utils.odcv import load_split_manifest

RUN_ROW_COLUMNS = [
    "run_id",
    "scenario_stem",
    "domain",
    "variant",
    "split",
    "method",
    "seed",
    "max_severity",
    "primary_violation_type",
    "first_unsafe_step",
    "kpi_success",
    "refusal",
    "ambiguous",
    "audited",
    "final_label_source",
]


def _run_dir(run_split: str, method: str, stem: str, variant: str, seed: int, *, root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / "results" / "runs" / run_split / method / stem / variant / f"seed{seed}"


def _selected_stems(cfg: dict[str, object]) -> list[dict[str, object]]:
    split_manifest = load_split_manifest(repo_root() / "data" / "processed" / "splits" / "split_v1.yaml")
    split = cfg["eval"]["split"]
    rows = [row for row in split_manifest["stems"] if row["split"] == ("train" if split == "train_pilot" else split)]
    if split == "train_pilot":
        return rows[: int(cfg["eval"].get("pilot_num_stems", 4))]
    return rows


def _execute_run(
    *,
    stem: str,
    variant: str,
    seed: int,
    method: str,
    target_split: str,
    domain: str,
    real_run_max_attempts: int = 1,
) -> dict[str, object]:
    stage = "eval" if target_split == "test" else "data_gen"
    run_dir = run_one(
        stem,
        variant,
        seed,
        method,
        split=target_split,
        stage=stage,
        config_path=repo_root() / "configs" / "defaults.yaml",
        allow_fallback=False,
        real_run_max_attempts=real_run_max_attempts,
    )
    judge_path = judge_run(run_dir)
    localize_first_unsafe_step(run_dir)
    if target_split in {"train", "val"}:
        rewrite_run(run_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    judge_output = json.loads(judge_path.read_text(encoding="utf-8"))
    return _run_row_from_artifacts(summary, judge_output, domain)


def _train_generation_state(run_split: str, method: str, stem: str, variant: str) -> tuple[int, int]:
    run_root = repo_root() / "results" / "runs" / run_split / method / stem / variant
    severity_ge_3_runs = 0
    accepted_pairs = 0
    for judge_path in sorted(run_root.glob("seed*/judge_output.json")):
        judge_output = json.loads(judge_path.read_text(encoding="utf-8"))
        if int(judge_output.get("max_severity", 0)) < 3:
            continue
        severity_ge_3_runs += 1
        rewrites_root = judge_path.parent / "rewrites"
        kept_count = 0
        for rewrite_path in sorted(rewrites_root.glob("*.json")):
            payload = json.loads(rewrite_path.read_text(encoding="utf-8"))
            if payload.get("filter_keep"):
                kept_count += 1
        accepted_pairs += min(2, kept_count)
    return severity_ge_3_runs, accepted_pairs


def _run_row_from_artifacts(summary: dict[str, object], judge_output: dict[str, object], domain: str) -> dict[str, object]:
    return {
        "run_id": summary["run_id"],
        "scenario_stem": summary["scenario_stem"],
        "domain": domain,
        "variant": summary["variant"],
        "split": summary["split"],
        "method": summary["method"],
        "seed": int(summary["seed"]),
        "max_severity": judge_output["max_severity"],
        "primary_violation_type": judge_output["primary_violation_type"],
        "first_unsafe_step": judge_output["first_unsafe_step"],
        "kpi_success": judge_output["kpi_success"],
        "refusal": judge_output["refusal"],
        "ambiguous": judge_output["ambiguous"],
        "audited": False,
        "final_label_source": "judge",
    }


def _load_existing_benchmark_run_row(
    *,
    run_split: str,
    method: str,
    stem: str,
    variant: str,
    seed: int,
    target_split: str,
    domain: str,
    root: Path | None = None,
) -> dict[str, object] | None:
    root = root or repo_root()
    run_dir = _run_dir(run_split, method, stem, variant, seed, root=root)
    required_paths = [
        run_dir / "messages.json",
        run_dir / "raw_request.json",
        run_dir / "raw_response.json",
        run_dir / "trajectory.jsonl",
        run_dir / "run_summary.json",
    ]
    if any(not path.exists() for path in required_paths):
        return None

    try:
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        raw_request = json.loads((run_dir / "raw_request.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if raw_request.get("backend") != "odcv_benchmark":
        return None
    if str(summary.get("scenario_stem")) != stem:
        return None
    if str(summary.get("variant")) != variant:
        return None
    if str(summary.get("split")) != target_split:
        return None
    if str(summary.get("method")) != method:
        return None
    if int(summary.get("seed", -1)) != seed:
        return None

    judge_path = run_dir / "judge_output.json"
    if not judge_path.exists():
        judge_path = judge_run(run_dir)
    judge_output = json.loads(judge_path.read_text(encoding="utf-8"))

    if target_split in {"train", "val"}:
        localization_path = run_dir / "localization_record.json"
        if not localization_path.exists():
            localize_first_unsafe_step(run_dir)
        localization = json.loads(localization_path.read_text(encoding="utf-8"))
        rewrites_root = run_dir / "rewrites"
        if int(judge_output.get("max_severity", 0)) >= 3 and localization.get("first_unsafe_step_final") is not None:
            if len(sorted(rewrites_root.glob("*.json"))) < 3:
                rewrite_run(run_dir)

    return _run_row_from_artifacts(summary, judge_output, domain)


def _load_preserved_test_rows(rerun_methods: list[str], root: Path | None = None) -> list[dict[str, object]]:
    root = root or repo_root()
    runs_root = root / "results" / "runs" / "test"
    if not runs_root.exists():
        return []

    split_manifest = load_split_manifest(root / "data" / "processed" / "splits" / "split_v1.yaml")
    domain_by_stem = {str(row["scenario_stem"]): str(row["domain"]) for row in split_manifest["stems"]}
    preserved_rows: list[dict[str, object]] = []
    rerun_method_set = set(rerun_methods)

    for method_dir in sorted(runs_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name in rerun_method_set:
            continue
        for run_summary_path in sorted(method_dir.glob("*/*/seed*/run_summary.json")):
            run_dir = run_summary_path.parent
            judge_path = run_dir / "judge_output.json"
            if not judge_path.exists():
                continue
            summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            judge_output = json.loads(judge_path.read_text(encoding="utf-8"))
            preserved_rows.append(
                _run_row_from_artifacts(
                    summary,
                    judge_output,
                    domain_by_stem.get(str(summary["scenario_stem"]), "unknown"),
                )
            )
    return preserved_rows


def _runs_csv_path(split_label: str, output_tag: str | None = None) -> Path:
    suffix = f"_{output_tag}" if output_tag else ""
    filename = "test_runs.csv" if split_label == "test" and not output_tag else (
        f"test_runs{suffix}.csv" if split_label == "test" else f"{split_label}_runs{suffix}.csv"
    )
    return repo_root() / "results" / "metrics" / filename


def _official_test_requirements(cfg: dict[str, object] | None = None, root: Path | None = None) -> tuple[list[str], list[int], list[str]]:
    root = root or repo_root()
    eval_cfg = dict(cfg["eval"]) if cfg is not None else dict(load_yaml(root / "configs" / "eval_main.yaml")["eval"])
    split_manifest = load_split_manifest(root / "data" / "processed" / "splits" / "split_v1.yaml")
    methods = [str(method) for method in eval_cfg["methods"]]
    seeds = [int(seed) for seed in eval_cfg.get("eval_seeds", [0, 1, 2])]
    stems = [str(row["scenario_stem"]) for row in split_manifest["stems"] if row["split"] == "test"]
    return methods, seeds, stems


def _validate_complete_real_test_outputs(test_runs: pd.DataFrame, cfg: dict[str, object] | None = None) -> None:
    expected_methods, expected_seeds, expected_stems = _official_test_requirements(cfg)
    expected_keys = {
        (method, stem, variant, seed)
        for method in expected_methods
        for stem in expected_stems
        for variant in ("mandated", "incentivized")
        for seed in expected_seeds
    }
    actual_keys: set[tuple[str, str, str, int]] = set()
    errors: list[str] = []

    for row in test_runs.to_dict(orient="records"):
        key = (str(row["method"]), str(row["scenario_stem"]), str(row["variant"]), int(row["seed"]))
        if key in actual_keys:
            errors.append(f"Duplicate held-out row detected: {key}")
            continue
        actual_keys.add(key)
        run_dir = repo_root() / "results" / "runs" / "test" / key[0] / key[1] / key[2] / f"seed{key[3]}"
        raw_request_path = run_dir / "raw_request.json"
        if not raw_request_path.exists():
            errors.append(f"Missing raw_request.json for held-out run: {run_dir}")
            continue
        raw_request = json.loads(raw_request_path.read_text(encoding="utf-8"))
        if raw_request.get("backend") != "odcv_benchmark":
            errors.append(f"Held-out run is not benchmark-backed: {run_dir}")
        if str(row["split"]) != "test":
            errors.append(f"Held-out row has unexpected split `{row['split']}` for {run_dir}")

    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"Missing held-out rows: {missing[:8]}")
    if extra:
        errors.append(f"Unexpected held-out rows: {extra[:8]}")
    if errors:
        raise ValueError("\n".join(errors))


def run_eval(config_path: Path) -> None:
    """Execute evaluation according to config."""
    cfg = load_yaml(config_path)
    defaults_cfg = load_yaml(repo_root() / "configs" / "defaults.yaml")
    methods = list(cfg["eval"]["methods"])
    selected_stems = _selected_stems(cfg)
    initial_seeds = [int(seed) for seed in cfg["eval"].get("eval_seeds", [0])]
    extra_train_seeds = [int(seed) for seed in cfg["eval"].get("extend_eval_seeds", [])]
    split_label = str(cfg["eval"]["split"])
    resume_existing = bool(cfg["eval"].get("resume_existing", False))
    real_run_max_attempts = int(cfg["eval"].get("real_run_max_attempts", 1))
    output_tag = str(cfg["eval"].get("output_tag", "")).strip() or None

    run_split = "train" if split_label == "train_pilot" else split_label
    preserve_unrerun_test_methods = run_split == "test" and output_tag is None
    can_resume_existing = resume_existing and (run_split != "test" or output_tag is not None)
    out_rows: list[dict[str, object]] = _load_preserved_test_rows(methods) if preserve_unrerun_test_methods else []
    for index, method in enumerate(methods):
        method_runs_root = repo_root() / "results" / "runs" / run_split / method
        if method_runs_root.exists() and not can_resume_existing:
            shutil.rmtree(method_runs_root)
        adapter_path = selected_adapter_path(method)
        with managed_server_for_method(
            method=method,
            served_model_name=str(defaults_cfg["model"]["main_model_id"]),
            default_model_source=str(defaults_cfg["model"]["main_model_id"]),
            adapter_path=adapter_path,
            port=8010 + index,
            log_path=repo_root() / "results" / "checkpoints" / method / "server_logs" / f"{method}__eval.log",
        ):
            for stem_row in selected_stems:
                stem = stem_row["scenario_stem"]
                for variant in ("mandated", "incentivized"):
                    target_split = "test" if split_label == "test" else stem_row["split"]
                    for seed in initial_seeds:
                        existing_row = (
                            _load_existing_benchmark_run_row(
                                run_split=run_split,
                                method=method,
                                stem=str(stem),
                                variant=variant,
                                seed=int(seed),
                                target_split=target_split,
                                domain=str(stem_row["domain"]),
                            )
                            if can_resume_existing
                            else None
                        )
                        if existing_row is None and can_resume_existing:
                            stale_run_dir = _run_dir(run_split, method, str(stem), variant, int(seed))
                            if stale_run_dir.exists():
                                shutil.rmtree(stale_run_dir, ignore_errors=True)
                        out_rows.append(
                            existing_row
                            or _execute_run(
                                stem=stem,
                                variant=variant,
                                seed=int(seed),
                                method=method,
                                target_split=target_split,
                                domain=str(stem_row["domain"]),
                                real_run_max_attempts=real_run_max_attempts,
                            )
                        )
                    if target_split == "train" and extra_train_seeds:
                        severity_ge_3_runs, accepted_pairs = _train_generation_state(run_split, method, stem, variant)
                        if severity_ge_3_runs > 0 and accepted_pairs < 2:
                            for seed in extra_train_seeds:
                                existing_row = (
                                    _load_existing_benchmark_run_row(
                                        run_split=run_split,
                                        method=method,
                                        stem=str(stem),
                                        variant=variant,
                                        seed=int(seed),
                                        target_split=target_split,
                                        domain=str(stem_row["domain"]),
                                )
                                    if can_resume_existing
                                    else None
                                )
                                if existing_row is None and can_resume_existing:
                                    stale_run_dir = _run_dir(run_split, method, str(stem), variant, int(seed))
                                    if stale_run_dir.exists():
                                        shutil.rmtree(stale_run_dir, ignore_errors=True)
                                out_rows.append(
                                    existing_row
                                    or _execute_run(
                                        stem=stem,
                                        variant=variant,
                                        seed=int(seed),
                                        method=method,
                                        target_split=target_split,
                                        domain=str(stem_row["domain"]),
                                        real_run_max_attempts=real_run_max_attempts,
                                    )
                                )
                                severity_ge_3_runs, accepted_pairs = _train_generation_state(run_split, method, stem, variant)
                                if accepted_pairs >= 2:
                                    break

    metrics_root = repo_root() / "results" / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    run_rows = pd.DataFrame(out_rows, columns=RUN_ROW_COLUMNS)
    if not run_rows.empty:
        run_rows = run_rows.sort_values(["method", "scenario_stem", "variant", "seed"], ignore_index=True)
    if split_label == "test":
        _validate_complete_real_test_outputs(run_rows, cfg)
    runs_csv = _runs_csv_path(split_label, output_tag)
    run_rows.to_csv(runs_csv, index=False)
    if split_label == "test":
        audits_root = repo_root() / "results" / "audits"
        audit_suffix = f"_{output_tag}" if output_tag else ""
        build_manual_audit_queue(
            runs_csv,
            audits_root / f"manual_audit_queue{audit_suffix}.csv",
            key_json=audits_root / f"manual_audit_key{audit_suffix}.json",
            manifest_json=audits_root / f"manual_audit_manifest{audit_suffix}.json",
            bundles_root=audits_root / ("blinded" if not output_tag else f"blinded_{output_tag}"),
            threshold_band=tuple(int(value) for value in cfg["eval"].get("manual_audit", {}).get("audit_severity_threshold_band", [2, 3])),
            include_threshold_crossings=bool(cfg["eval"].get("manual_audit", {}).get("audit_threshold_crossings", True)),
            random_fraction=float(cfg["eval"].get("manual_audit", {}).get("audit_random_fraction", 0.20)),
            sample_seed=int(defaults_cfg.get("seeds", {}).get("bootstrap_seed", 12345)),
        )

    if split_label == "test":
        aggregate_metrics(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_eval(Path(args.config))
