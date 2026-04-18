from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _load_odcv_run_module():
    module_path = ROOT / "external" / "ODCV-Bench" / "run.py"
    spec = importlib.util.spec_from_file_location("odcv_bench_run_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_text_command_uses_utf8_replace(monkeypatch) -> None:
    module = _load_odcv_run_module()
    captured: dict[str, object] = {}

    def _fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    result = module._run_text_command(["docker", "compose", "down", "-v"])

    assert result.returncode == 0
    assert captured["args"] == ["docker", "compose", "down", "-v"]
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
    assert captured["kwargs"]["text"] is True


def test_run_docker_compose_uses_detached_orchestrator_and_wait(monkeypatch, tmp_path: Path) -> None:
    module = _load_odcv_run_module()
    commands: list[list[str]] = []

    def _fake_run_text_command(args: list[str]):
        commands.append(args)
        if args[:3] == ["docker", "compose", "build"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="build ok")
        if args[:6] == ["docker", "compose", "up", "-d", "--force-recreate", "orchestrator"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="orchestrator up")
        if args[:6] == ["docker", "compose", "up", "-d", "--force-recreate", "executor"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="executor up")
        if args[:3] == ["docker", "compose", "down"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="down ok")
        if args[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"logs for {args[-1]}")
        if args[:2] == ["docker", "wait"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="0")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(module, "_run_text_command", _fake_run_text_command)
    monkeypatch.setattr(module, "_wait_for_container_health", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "_compose_images_available", lambda: False)

    log_file = tmp_path / "run.log"
    exit_code = module.run_docker_compose(log_file)

    assert exit_code == 0
    assert ["docker", "compose", "build"] in commands
    assert ["docker", "compose", "up", "-d", "--force-recreate", "orchestrator"] in commands
    assert ["docker", "compose", "up", "-d", "--force-recreate", "executor"] in commands
    assert ["docker", "wait", "mission_executor"] in commands
    assert (tmp_path / "docker_output.log").exists()


def test_run_docker_compose_skips_build_when_cached_images_exist(monkeypatch, tmp_path: Path) -> None:
    module = _load_odcv_run_module()
    commands: list[list[str]] = []

    def _fake_run_text_command(args: list[str]):
        commands.append(args)
        if args[:3] == ["docker", "compose", "up"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="up ok")
        if args[:3] == ["docker", "compose", "down"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="down ok")
        if args[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"logs for {args[-1]}")
        if args[:2] == ["docker", "wait"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="0")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(module, "_run_text_command", _fake_run_text_command)
    monkeypatch.setattr(module, "_wait_for_container_health", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "_compose_images_available", lambda: True)

    log_file = tmp_path / "run.log"
    exit_code = module.run_docker_compose(log_file)

    assert exit_code == 0
    assert ["docker", "compose", "build"] not in commands
    assert ["docker", "compose", "up", "-d", "--force-recreate", "orchestrator"] in commands
    assert ["docker", "compose", "up", "-d", "--force-recreate", "executor"] in commands
    assert "Skipping docker compose build because cached local orchestrator/executor images already exist." in log_file.read_text(encoding="utf-8")


def test_main_writes_missing_result_placeholder_when_messages_are_recovered(monkeypatch, tmp_path: Path) -> None:
    module = _load_odcv_run_module()
    artifact_dir = tmp_path / "experiments" / "job"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            job_label="job",
            clean_after=False,
            no_timestamp=True,
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="local",
            openai_model="Qwen/Qwen2.5-7B-Instruct",
            openai_temperature="0.6",
        ),
    )
    monkeypatch.setattr(module, "ensure_working_directory_is_script_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "build_artifact_dir", lambda base_label, no_timestamp=False: artifact_dir)
    monkeypatch.setattr(module, "run_docker_compose", lambda log_file: 0)

    def _fake_docker_cp(container_path: str, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if container_path.endswith("messages_record.txt"):
            dest.write_text("== Step 1 ==\nrole: assistant\ncontent: done\n", encoding="utf-8")
            return True
        return False

    monkeypatch.setattr(module, "docker_cp", _fake_docker_cp)
    monkeypatch.setattr(
        module,
        "_run_text_command",
        lambda args: subprocess.CompletedProcess(args=args, returncode=0, stdout="cleanup ok"),
    )

    exit_code = module.main()

    assert exit_code == 0
    payload = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "missing_result_json"
    assert (artifact_dir / "messages_record.txt").exists()
