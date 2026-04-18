"""Helpers for running local adapter-backed OpenAI-compatible model servers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Iterator
from urllib import error, request

from src.utils.io import load_json, repo_root


def resolve_model_source(default_source: str | None) -> str | None:
    override = os.environ.get("ODCV_BASE_MODEL_PATH", "").strip()
    if override:
        return override
    if default_source:
        return str(default_source)
    return None


def is_local_model_source(model_source: str | None) -> bool:
    return bool(model_source and Path(model_source).exists())


@contextmanager
def temporary_env(overrides: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LocalModelServer:
    """Start and stop the repo-owned local OpenAI-compatible server."""

    def __init__(
        self,
        *,
        model_source: str,
        served_model_name: str,
        adapter_path: Path | None = None,
        port: int = 8010,
        host: str = "127.0.0.1",
        dtype: str = "bfloat16",
        log_path: Path | None = None,
        startup_timeout_seconds: float = 240.0,
    ) -> None:
        self.model_source = model_source
        self.served_model_name = served_model_name
        self.adapter_path = adapter_path
        self.port = int(port)
        self.host = host
        self.dtype = dtype
        self.log_path = log_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def pid_path(self) -> Path | None:
        return None if self.log_path is None else self.log_path.with_suffix(".pid")

    def _cleanup_stale_pid(self) -> None:
        pid_path = self.pid_path
        if pid_path is None or not pid_path.exists():
            return
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid_path.unlink(missing_ok=True)
            return

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        else:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
        pid_path.unlink(missing_ok=True)

    def _port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            return probe.connect_ex((self.host, self.port)) == 0

    def start(self) -> None:
        if self._process is not None:
            return
        self._cleanup_stale_pid()
        if self._port_in_use():
            raise RuntimeError(
                f"Local model server port {self.port} is already in use on {self.host}. "
                "Refusing to attach to an unknown listener."
            )
        command = [
            sys.executable,
            "-m",
            "src.agents.serve_local_openai",
            "--model-path",
            self.model_source,
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
        ]
        if self.adapter_path is not None:
            command.extend(["--adapter-path", str(self.adapter_path)])
        stdout_target = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("w", encoding="utf-8")
            stdout_target = self._log_handle
        self._process = subprocess.Popen(
            command,
            cwd=repo_root(),
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            text=True,
        )
        pid_path = self.pid_path
        if pid_path is not None:
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text(str(self._process.pid), encoding="utf-8")
        self._wait_until_ready()

    def _health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_seconds
        last_error = "server did not become healthy"
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"Local model server exited early with code {self._process.returncode}.")
            try:
                with request.urlopen(self._health_url(), timeout=5) as response:
                    if response.status == 200:
                        return
            except error.URLError as exc:
                last_error = str(exc)
            time.sleep(2)
        self.stop()
        raise RuntimeError(f"Timed out waiting for local model server on {self._health_url()}: {last_error}")

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        finally:
            self._process = None
            pid_path = self.pid_path
            if pid_path is not None:
                pid_path.unlink(missing_ok=True)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def __enter__(self) -> "LocalModelServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def selected_adapter_path(method: str) -> Path | None:
    manifest_path = repo_root() / "manifests" / "training_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    method_entry = manifest.get("methods", {}).get(method, {})
    selected = method_entry.get("selected_checkpoint")
    if not selected:
        return None
    resolved = repo_root() / str(selected)
    return resolved if resolved.exists() else None


@contextmanager
def managed_server_for_method(
    *,
    method: str,
    served_model_name: str,
    default_model_source: str | None,
    adapter_path: Path | None = None,
    port: int = 8010,
    log_path: Path | None = None,
) -> Iterator[dict[str, str] | None]:
    model_source = resolve_model_source(default_model_source)
    if not is_local_model_source(model_source):
        yield None
        return
    server = LocalModelServer(
        model_source=str(model_source),
        served_model_name=served_model_name,
        adapter_path=adapter_path,
        port=port,
        log_path=log_path,
    )
    with server:
        with temporary_env(
            {
                "ODCV_LOCAL_BASE_URL": server.base_url,
                "ODCV_LOCAL_MODEL_ID": served_model_name,
                "ODCV_LOCAL_ADAPTER_ID": None if adapter_path is None else str(adapter_path),
            }
        ):
            yield {"base_url": server.base_url, "model_id": served_model_name, "method": method}
