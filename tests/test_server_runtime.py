from __future__ import annotations

import socket
from pathlib import Path

import pytest

import src.agents.server_runtime as server_runtime


class _DummyProcess:
    def __init__(self, pid: int = 43210) -> None:
        self.pid = pid
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


def test_local_model_server_refuses_unknown_port_listener(tmp_path: Path) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        server = server_runtime.LocalModelServer(
            model_source="E:/Model/Qwen2.5-7B-Instruct",
            served_model_name="Qwen/Qwen2.5-7B-Instruct",
            port=port,
            log_path=tmp_path / "server.log",
        )
        with pytest.raises(RuntimeError, match="already in use"):
            server.start()
    finally:
        sock.close()


def test_local_model_server_tracks_pid_file(tmp_path: Path, monkeypatch) -> None:
    dummy = _DummyProcess()

    monkeypatch.setattr(server_runtime.subprocess, "Popen", lambda *args, **kwargs: dummy)
    monkeypatch.setattr(server_runtime.LocalModelServer, "_wait_until_ready", lambda self: None)
    monkeypatch.setattr(server_runtime, "repo_root", lambda: tmp_path)

    server = server_runtime.LocalModelServer(
        model_source="E:/Model/Qwen2.5-7B-Instruct",
        served_model_name="Qwen/Qwen2.5-7B-Instruct",
        port=8019,
        log_path=tmp_path / "server.log",
    )

    server.start()

    pid_path = tmp_path / "server.pid"
    assert pid_path.read_text(encoding="utf-8") == str(dummy.pid)

    server.stop()

    assert dummy.terminated is True
    assert not pid_path.exists()
