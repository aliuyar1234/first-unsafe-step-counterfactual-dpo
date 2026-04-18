"""Helpers for reproducibility manifests.

Requirements:
- compute SHA256 digests
- snapshot configs and prompt hashes
- record external repo commit hashes and model IDs
- write deterministic JSON
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any, Dict

import orjson

from src.utils.io import maybe_git_commit, resolve_repo_path

def sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest for one file."""
    resolved = resolve_repo_path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Write deterministic JSON manifest to disk."""
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def collect_prompt_hashes(prompt_paths: Dict[str, str]) -> Dict[str, str]:
    return {name: sha256_file(Path(prompt_path)) for name, prompt_path in sorted(prompt_paths.items())}


def external_repo_manifest(repo_path: Path | str) -> Dict[str, Any]:
    resolved = resolve_repo_path(repo_path)
    commit = maybe_git_commit(resolved)
    remote_url = None
    if resolved.exists():
        try:
            proc = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=resolved,
                check=True,
                capture_output=True,
                text=True,
            )
            remote_url = proc.stdout.strip() or None
        except (FileNotFoundError, subprocess.CalledProcessError):
            remote_url = None
    return {
        "path": str(resolved),
        "commit": commit,
        "remote_url": remote_url,
    }
