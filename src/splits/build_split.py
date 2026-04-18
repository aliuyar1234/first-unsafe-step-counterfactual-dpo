"""Build a frozen stem-level train/val/test split for ODCV-Bench.

Behavior requirements:
- read actual stem names from `external/ODCV-Bench/mandated_scenarios`
- require a finalized domain map file
- keep mandated/incentivized variants of the same stem together
- produce exactly 24 train / 8 val / 8 test stems by default
- deterministic with seed=17
- write YAML plus a manifest with hashes

CLI shape:
    python -m src.splits.build_split --config configs/split_policy.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import dump_json, dump_yaml, ensure_dir, load_yaml, repo_root
from src.utils.odcv import list_scenario_stems
from src.utils.schemas import validate_payload


def _allocate_group_counts(
    domain_size: int,
    remaining_targets: dict[str, int],
    remaining_total: int,
) -> dict[str, int]:
    allocations: dict[str, int] = {split: 0 for split in remaining_targets}
    fractional: list[tuple[float, str]] = []
    for split, target in remaining_targets.items():
        if remaining_total == 0:
            raw = 0.0
        else:
            raw = domain_size * target / remaining_total
        floor = min(target, int(raw))
        allocations[split] = floor
        fractional.append((raw - floor, split))
    leftover = domain_size - sum(allocations.values())
    for _, split in sorted(fractional, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        if allocations[split] < remaining_targets[split]:
            allocations[split] += 1
            leftover -= 1
    if leftover:
        for split in sorted(remaining_targets):
            while leftover > 0 and allocations[split] < remaining_targets[split]:
                allocations[split] += 1
                leftover -= 1
    return allocations


def build_split(config_path: Path) -> Path:
    """Build and write the frozen split manifest.

    Returns:
        Path to the written split YAML.
    """
    cfg = load_yaml(config_path)
    split_cfg = cfg["split"]
    source_dir = repo_root() / split_cfg["source_dir"]
    domain_map_path = repo_root() / split_cfg["domain_map_file"]
    output_path = repo_root() / split_cfg["output_file"]
    manifest_path = repo_root() / "manifests" / "split_manifest.json"
    stems = list_scenario_stems(source_dir)

    domain_map = load_yaml(domain_map_path) or {}
    stem_to_domain: dict[str, str] = domain_map.get("stem_to_domain", {})
    if not stem_to_domain:
        raise ValueError(f"{domain_map_path} has an empty stem_to_domain map.")
    missing = [stem for stem in stems if stem not in stem_to_domain]
    extra = [stem for stem in stem_to_domain if stem not in stems]
    if missing or extra:
        raise ValueError(f"Domain map mismatch. Missing={missing} Extra={extra}")

    counts: dict[str, int] = split_cfg["counts"]
    if sum(counts.values()) != len(stems):
        raise ValueError(f"Split counts {counts} do not match discovered stem count {len(stems)}.")

    grouped: dict[str, list[str]] = {}
    for stem in stems:
        grouped.setdefault(stem_to_domain[stem], []).append(stem)

    remaining_targets = dict(counts)
    remaining_total = len(stems)
    assignments: dict[str, str] = {}
    for domain in sorted(grouped):
        group_stems = sorted(grouped[domain])
        allocations = _allocate_group_counts(len(group_stems), remaining_targets, remaining_total)
        cursor = 0
        for split in ("train", "val", "test"):
            count = allocations[split]
            for stem in group_stems[cursor : cursor + count]:
                assignments[stem] = split
            cursor += count
            remaining_targets[split] -= count
        remaining_total -= len(group_stems)

    stems_payload = [
        {
            "scenario_stem": stem,
            "domain": stem_to_domain[stem],
            "split": assignments[stem],
        }
        for stem in stems
    ]
    payload: dict[str, Any] = {
        "split_version": "split_v1",
        "seed": split_cfg["seed"],
        "source_dir": split_cfg["source_dir"],
        "counts": counts,
        "stems": stems_payload,
    }
    validate_payload(payload, "split_manifest.schema.json")
    ensure_dir(output_path.parent)
    ensure_dir(manifest_path.parent)
    dump_yaml(output_path, payload)
    dump_json(manifest_path, payload)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(build_split(Path(args.config)))
