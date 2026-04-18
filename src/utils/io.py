from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import orjson
import yaml


ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return ROOT


def resolve_repo_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def ensure_dir(path: Path | str) -> Path:
    resolved = resolve_repo_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_text(path: Path | str) -> str:
    return resolve_repo_path(path).read_text(encoding="utf-8")


def write_text(path: Path | str, text: str) -> Path:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def load_json(path: Path | str) -> Any:
    return json.loads(load_text(path))


def dump_json(path: Path | str, payload: Any) -> Path:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    return resolved


def load_yaml(path: Path | str) -> Any:
    return yaml.safe_load(load_text(path))


def dump_yaml(path: Path | str, payload: Any) -> Path:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return resolved


def read_jsonl(path: Path | str) -> list[Any]:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        return []
    rows: list[Any] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[Any]) -> Path:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS).decode("utf-8"))
            handle.write("\n")
    return resolved


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def relpath(path: Path | str) -> str:
    resolved = resolve_repo_path(path)
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def maybe_git_commit(path: Path | str) -> str | None:
    resolved = resolve_repo_path(path)
    workdir = resolved if resolved.is_dir() else resolved.parent
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None

