from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def _load_task_executor_module():
    module_path = ROOT / "external" / "ODCV-Bench" / "orchestrator_api" / "task_executor_api.py"
    spec = importlib.util.spec_from_file_location("odcv_task_executor_api_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(
        "flask",
        types.SimpleNamespace(
            Flask=lambda *args, **kwargs: types.SimpleNamespace(
                route=lambda *route_args, **route_kwargs: (lambda fn: fn)
            ),
            render_template=lambda *args, **kwargs: "",
            request=types.SimpleNamespace(json={}, get_json=lambda silent=True: {}),
            jsonify=lambda payload=None, **kwargs: payload if payload is not None else kwargs,
            redirect=lambda *args, **kwargs: None,
            url_for=lambda *args, **kwargs: "",
        ),
    )
    spec.loader.exec_module(module)
    return module


def test_execute_command_uses_bash_lc(monkeypatch, tmp_path: Path) -> None:
    module = _load_task_executor_module()
    captured: dict[str, object] = {}

    def _fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    result = module.execute_command("if [[ 1 -eq 1 ]]; then echo ok; fi", str(tmp_path))

    assert result["returncode"] == 0
    assert captured["args"] == ["/bin/bash", "-lc", "if [[ 1 -eq 1 ]]; then echo ok; fi"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"] == str(tmp_path)
