from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess


def _load_run_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "external" / "ODCV-Bench" / "run.py"
    spec = importlib.util.spec_from_file_location("odcv_bench_run", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_text_command_uses_utf8_replace(monkeypatch) -> None:
    module = _load_run_module()
    captured: dict[str, object] = {}

    def _fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    result = module._run_text_command(["docker", "compose", "up"])

    assert result.stdout == "ok"
    assert captured["text"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["stderr"] == subprocess.STDOUT
