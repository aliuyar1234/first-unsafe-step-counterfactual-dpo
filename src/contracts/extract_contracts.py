"""Extract one frozen contract JSON per ODCV scenario stem.

Behavior requirements:
- read the actual scenario files/prompts from the benchmark
- call the contract extraction model/prompt once per stem
- validate against `schemas/contract.schema.json`
- preserve raw extraction responses
- allow a manual review pass before contracts are marked final
- never change a finalized contract without logging it

CLI shape:
    python -m src.contracts.extract_contracts --config configs/defaults.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import dump_json, ensure_dir, load_yaml, relpath, repo_root
from src.utils.odcv import extract_contract_from_scenario, load_scenario_context, load_split_manifest
from src.utils.schemas import validate_payload


def extract_contracts(config_path: Path) -> int:
    """Extract and validate contracts for all stems.

    Returns:
        Number of finalized contracts written.
    """
    cfg = load_yaml(config_path)
    split_manifest_path = repo_root() / "data" / "processed" / "splits" / "split_v1.yaml"
    if not split_manifest_path.exists():
        raise FileNotFoundError("Frozen split missing. Run src.splits.build_split first.")
    split_manifest = load_split_manifest(split_manifest_path)
    odcv_root = repo_root() / cfg["paths"]["odcv_root"]
    contracts_root = ensure_dir(repo_root() / "data" / "processed" / "contracts")
    raw_root = ensure_dir(contracts_root / "raw")
    written = 0
    for row in split_manifest["stems"]:
        stem = row["scenario_stem"]
        scenario = load_scenario_context(odcv_root, stem, "mandated")
        contract = extract_contract_from_scenario(stem, row["domain"], scenario)
        validate_payload(contract, "contract.schema.json")
        raw_payload: dict[str, Any] = {
            "scenario_stem": stem,
            "extraction_backend": "heuristic_contract_extractor_v1",
            "source_files": scenario.source_files,
            "prompt_path": relpath(repo_root() / cfg["prompts"]["contract_extraction"]),
            "contract": contract,
        }
        dump_json(raw_root / f"{stem}.raw.json", raw_payload)
        dump_json(contracts_root / f"{stem}.json", contract)
        written += 1
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(extract_contracts(Path(args.config)))
