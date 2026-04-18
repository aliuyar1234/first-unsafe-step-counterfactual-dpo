"""Canonical artifact field names and helper types.

Implementers:
- keep these names aligned with `schemas/*.json`
- prefer explicit dataclasses or pydantic models
- validate immediately before writing files
"""

from __future__ import annotations

import jsonschema
from pathlib import Path
from typing import Any, Dict

from src.utils.io import load_json, resolve_repo_path


SCHEMA_DIR = resolve_repo_path("schemas")


def load_json_schema(path: Path) -> Dict[str, Any]:
    """Load one JSON Schema file from disk.

    Requirements:
    - raise a helpful error if missing
    - use UTF-8
    """
    resolved = path if path.is_absolute() else SCHEMA_DIR / path
    if not resolved.exists():
        raise FileNotFoundError(f"Schema file not found: {resolved}")
    payload = load_json(resolved)
    if not isinstance(payload, dict):
        raise TypeError(f"Schema file did not contain a JSON object: {resolved}")
    return payload


def validate_payload(payload: Dict[str, Any], schema_path: Path | str) -> None:
    schema = load_json_schema(Path(schema_path))
    jsonschema.validate(payload, schema)


def validate_jsonl_rows(rows: list[Dict[str, Any]], schema_path: Path | str) -> None:
    schema = load_json_schema(Path(schema_path))
    for row in rows:
        jsonschema.validate(row, schema)
