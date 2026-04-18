from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def _load_agent_module():
    module_path = ROOT / "external" / "ODCV-Bench" / "mission_executor" / "agent_main.py"
    spec = importlib.util.spec_from_file_location("odcv_agent_main_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
    spec.loader.exec_module(module)
    return module


def test_task_complete_verifier_skips_when_script_missing(monkeypatch) -> None:
    module = _load_agent_module()

    monkeypatch.setattr(
        module,
        "_dispatch_shell",
        lambda *args, **kwargs: {"stdout": "", "stderr": "missing", "returncode": 1, "role": "tool"},
    )

    assert module._maybe_run_task_complete_verifier() is None


def test_task_complete_verifier_reports_pass(monkeypatch) -> None:
    module = _load_agent_module()
    calls: list[str] = []

    def _fake_dispatch_shell(command: str, **kwargs):
        calls.append(command)
        if command.startswith("test -x"):
            return {"stdout": "", "stderr": "", "returncode": 0, "role": "tool"}
        return {"stdout": "PASS: validated 5 complete allocation records\n", "stderr": "", "returncode": 0, "role": "tool"}

    monkeypatch.setattr(module, "_dispatch_shell", _fake_dispatch_shell)

    verification = module._maybe_run_task_complete_verifier()

    assert verification is not None
    assert verification["passed"] is True
    assert "PASS:" in verification["stdout"]
    assert calls[0].startswith("test -x ")


def test_task_complete_verifier_reports_failure(monkeypatch) -> None:
    module = _load_agent_module()

    def _fake_dispatch_shell(command: str, **kwargs):
        if command.startswith("test -x"):
            return {"stdout": "", "stderr": "", "returncode": 0, "role": "tool"}
        return {"stdout": "FAIL: unexpected patient_id: disease\n", "stderr": "", "returncode": 1, "role": "tool"}

    monkeypatch.setattr(module, "_dispatch_shell", _fake_dispatch_shell)

    verification = module._maybe_run_task_complete_verifier()

    assert verification is not None
    assert verification["passed"] is False
    assert "unexpected patient_id" in verification["stdout"]
