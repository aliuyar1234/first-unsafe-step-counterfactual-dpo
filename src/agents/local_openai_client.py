"""Thin OpenAI-compatible client used to connect ODCV-Bench to a local vLLM server.

Behavior requirements:
- target `http://localhost:8000/v1` by default
- support chat/completions-style requests needed by the benchmark
- preserve raw request/response payloads for logging
- expose the smallest interface required by ODCV integration
- do not bake benchmark-specific logic into this client
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from urllib import error, request
from typing import Any, Dict


@dataclass
class LocalOpenAIClient:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "local"
    timeout_seconds: float = 30.0

    def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send one chat completion request to the local OpenAI-compatible backend."""
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Local backend returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach local backend at {self.base_url}: {exc.reason}") from exc
