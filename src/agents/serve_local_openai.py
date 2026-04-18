"""Serve a local Hugging Face causal LM behind a tiny OpenAI-compatible API.

This server is intentionally minimal:
- exposes `/v1/models` and `/v1/chat/completions`
- uses the model's native chat template, including Qwen tool-call formatting
- returns the response shape required by the ODCV-Bench runtime
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.agents.parse_agent_output import parse_agent_output

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency
    PeftModel = None


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", flags=re.IGNORECASE)
TRAILING_SPECIAL_PATTERN = re.compile(r"(<\|im_end\|>|<\|endoftext\|>)+\s*$")


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


class LocalHFOpenAIServer:
    def __init__(
        self,
        *,
        model_path: Path,
        served_model_name: str,
        adapter_path: Path | None,
        host: str,
        port: int,
        dtype: str,
        max_new_tokens_default: int,
    ) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.adapter_path = adapter_path
        self.host = host
        self.port = port
        self.max_new_tokens_default = max_new_tokens_default
        self.dtype = self._resolve_dtype(dtype)
        self.lock = Lock()
        self.tokenizer = None
        self.model = None

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        normalized = dtype_name.strip().lower()
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if self.adapter_path is not None:
            if PeftModel is None:
                raise RuntimeError("peft is required to load an adapter checkpoint.")
            model = PeftModel.from_pretrained(model, str(self.adapter_path), is_trainable=False)
        model.eval()
        self.model = model

    def _normalize_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            item: dict[str, Any] = {"role": message.role, "content": self._normalize_content(message.content)}
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = []
                for tool_call in message.tool_calls:
                    function = dict(tool_call.get("function", {}))
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError:
                            function["arguments"] = arguments
                    normalized_call = dict(tool_call)
                    normalized_call["function"] = function
                    item["tool_calls"].append(normalized_call)
            normalized.append(item)
        return normalized

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        if content is None:
            return ""
        return str(content)

    def _render_prompt(self, request: ChatCompletionRequest) -> torch.Tensor:
        assert self.tokenizer is not None
        messages = self._normalize_messages(request.messages)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=request.tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        return encoded["input_ids"].to(self.model.device)

    def _decode_generation(self, generated_ids: torch.Tensor, prompt_length: int) -> str:
        assert self.tokenizer is not None
        completion_ids = generated_ids[0, prompt_length:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        return TRAILING_SPECIAL_PATTERN.sub("", text).strip()

    def _fallback_tool_call(self, text: str) -> tuple[str | None, list[dict[str, Any]]]:
        parsed = parse_agent_output(text)
        action_type = parsed["parsed_action_type"].lower().strip()
        action_input = parsed["parsed_action"].strip()
        if action_type == "bash":
            return "", [self._build_tool_call("bash", {"command": action_input})]
        if action_type in {"task_complete", "final"} and action_input:
            return "", [self._build_tool_call("task_complete", {"reason": action_input})]
        return text, []

    def _extract_tool_calls(self, text: str) -> tuple[str | None, list[dict[str, Any]]]:
        matches = list(TOOL_CALL_PATTERN.finditer(text))
        if not matches:
            return self._fallback_tool_call(text)
        tool_calls: list[dict[str, Any]] = []
        for match in matches:
            raw_json = match.group(1).strip()
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                try:
                    payload = ast.literal_eval(raw_json)
                except (SyntaxError, ValueError):
                    continue
            if not isinstance(payload, dict):
                continue
            name = str(payload.get("name", "")).strip()
            arguments = payload.get("arguments", {})
            if not name:
                continue
            tool_calls.append(self._build_tool_call(name, arguments))
        content = TOOL_CALL_PATTERN.sub("", text).strip()
        if not tool_calls:
            return self._fallback_tool_call(content or text)
        return content or None, tool_calls

    def _build_tool_call(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                parsed_arguments: dict[str, Any] | str = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"raw": arguments}
        else:
            parsed_arguments = arguments
        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(parsed_arguments, ensure_ascii=True),
            },
        }

    def complete(self, request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported by this local server.")
        if self.model is None or self.tokenizer is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")

        max_new_tokens = request.max_completion_tokens or request.max_tokens or self.max_new_tokens_default
        do_sample = request.temperature is not None and request.temperature > 0.0
        input_ids = self._render_prompt(request)
        prompt_length = int(input_ids.shape[1])

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(request.temperature)
            generation_kwargs["top_p"] = float(request.top_p)

        with self.lock, torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)
        completion_text = self._decode_generation(output_ids, prompt_length)
        content, tool_calls = self._extract_tool_calls(completion_text)
        finish_reason = "tool_calls" if tool_calls else "stop"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_length,
                "completion_tokens": int(output_ids.shape[1] - prompt_length),
                "total_tokens": int(output_ids.shape[1]),
            },
        }


def create_app(server: LocalHFOpenAIServer) -> FastAPI:
    app = FastAPI(title="Local HF OpenAI Server")

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": server.served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(payload: ChatCompletionRequest) -> JSONResponse:
        return JSONResponse(server.complete(payload))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": server.served_model_name}

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default=os.environ.get("ODCV_SERVER_DTYPE", "bfloat16"))
    parser.add_argument("--max-new-tokens-default", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    server = LocalHFOpenAIServer(
        model_path=Path(args.model_path),
        served_model_name=args.served_model_name,
        adapter_path=None if not args.adapter_path else Path(args.adapter_path),
        host=args.host,
        port=int(args.port),
        dtype=args.dtype,
        max_new_tokens_default=int(args.max_new_tokens_default),
    )
    server.load()

    import uvicorn

    uvicorn.run(create_app(server), host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
