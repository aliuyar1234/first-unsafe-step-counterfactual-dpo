from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_contract_example_valid():
    jsonschema.validate(_load_json("examples/contract_example.json"), _load_json("schemas/contract.schema.json"))


def test_trajectory_example_valid():
    jsonschema.validate(_load_json("examples/trajectory_step_example.json"), _load_json("schemas/trajectory_step.schema.json"))


def test_judge_example_valid():
    jsonschema.validate(_load_json("examples/judge_output_example.json"), _load_json("schemas/judge_output.schema.json"))


def test_rewrite_example_valid():
    jsonschema.validate(_load_json("examples/counterfactual_rewrite_example.json"), _load_json("schemas/counterfactual_rewrite.schema.json"))


def test_preference_pair_example_valid():
    jsonschema.validate(_load_json("examples/preference_pair_example.json"), _load_json("schemas/preference_pair.schema.json"))


def test_evaluation_result_example_valid():
    jsonschema.validate(_load_json("examples/evaluation_result_example.json"), _load_json("schemas/evaluation_result.schema.json"))


def test_new_schema_examples_validate() -> None:
    jsonschema.validate(_load_json("examples/split_manifest_example.json"), _load_json("schemas/split_manifest.schema.json"))
    jsonschema.validate(_load_json("examples/run_summary_example.json"), _load_json("schemas/run_summary.schema.json"))
    jsonschema.validate(_load_json("examples/localization_record_example.json"), _load_json("schemas/localization_record.schema.json"))


def test_claims_checklist_bibliography_keys_exist() -> None:
    claims = yaml.safe_load((ROOT / "paper" / "claims_checklist.yaml").read_text(encoding="utf-8"))
    bib_text = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
    bib_keys = set()
    for line in bib_text.splitlines():
        line = line.strip()
        if line.startswith("@") and "{" in line and line.endswith(","):
            bib_keys.add(line.split("{", 1)[1][:-1])
    for claim in claims["claims"]:
        for item in claim.get("evidence_required", []):
            if isinstance(item, dict) and "bibliography" in item:
                assert item["bibliography"] in bib_keys
