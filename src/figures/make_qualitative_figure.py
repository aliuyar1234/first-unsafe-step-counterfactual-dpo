"""Generate the qualitative before/after trajectory figure from run artifacts and audit notes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eval.manual_audit import manual_audit_manifest_is_resolved
from src.figures.common import ensure_parent


def _load_steps(run_dir: Path) -> list[dict]:
    path = run_dir / "trajectory.jsonl"
    steps: list[dict] = []
    if not path.exists():
        return steps
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            steps.append(json.loads(line))
    return steps


def _pick_run_dir(runs_root: Path, run_id: str) -> Path | None:
    for path in runs_root.rglob("run_summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("run_id") == run_id:
            return path.parent
    return None


def _render_steps(steps: list[dict], max_lines: int = 3) -> str:
    rendered = []
    for step in steps[:max_lines]:
        rendered.append(f"[{step['step_idx']}] {step['assistant_text']}")
    return "\n\n".join(rendered) if rendered else "No trajectory found."


def main(runs_root: Path, audit_csv: Path, out_path: Path) -> None:
    manifest_path = audit_csv.with_name("manual_audit_manifest.json")
    if not manual_audit_manifest_is_resolved(manifest_path):
        raise RuntimeError(
            f"Manual audit is not finalized yet. Resolve the blinded queue before generating qualitative figures: {manifest_path}"
        )
    audit = pd.read_csv(audit_csv)
    crossing = audit[audit["threshold_crossing_case"] == True]  # noqa: E712
    if crossing.empty:
        candidate = audit.iloc[0] if not audit.empty else None
    else:
        candidate = crossing.iloc[0]

    before_text = "No audit rows available."
    after_text = "No audit rows available."
    title = "Qualitative Trajectory Snapshot"
    if candidate is not None:
        stem = candidate["scenario_stem"]
        variant = candidate["variant"]
        title = f"Qualitative Trajectory: {stem} ({variant})"
        base_rows = audit[(audit["scenario_stem"] == stem) & (audit["variant"] == variant) & (audit["resolved_method"] == "base")]
        dpo_rows = audit[(audit["scenario_stem"] == stem) & (audit["variant"] == variant) & (audit["resolved_method"] == "fus_dpo")]
        if not base_rows.empty:
            base_dir = _pick_run_dir(runs_root, str(base_rows.iloc[0]["run_id"]))
            if base_dir is not None:
                before_text = _render_steps(_load_steps(base_dir))
        if not dpo_rows.empty:
            dpo_dir = _pick_run_dir(runs_root, str(dpo_rows.iloc[0]["run_id"]))
            if dpo_dir is not None:
                after_text = _render_steps(_load_steps(dpo_dir))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)
    axes[0].axis("off")
    axes[0].set_title("Base")
    axes[0].text(0.0, 1.0, before_text, va="top", ha="left", wrap=True, fontsize=9, family="monospace")
    axes[1].axis("off")
    axes[1].set_title("FUS-DPO")
    axes[1].text(0.0, 1.0, after_text, va="top", ha="left", wrap=True, fontsize=9, family="monospace")
    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
