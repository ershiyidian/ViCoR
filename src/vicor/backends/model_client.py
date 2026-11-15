"""Async OpenAI-compatible and Responses API VLM client."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from httpx_sse import aconnect_sse
from PIL import Image

from .api_slots import api_slot
from .responses_adapter import (
    from_responses_output,
    load_codex_responses_settings,
    to_responses_input,
    to_responses_tools,
)


def tool(
    name: str,
    description: str,
    properties: dict | None = None,
    required: tuple[str, ...] = (),
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
            },
        },
    }


TEXT = {"type": "string"}
NUMBER = {"type": "integer"}

MAIN_GAME_TOOLS = [
    tool("get_screenshot", "Request the next fresh visual observation."),
    tool(
        "press",
        "Press one permitted key for a duration.",
        {"key": TEXT, "duration_ms": NUMBER},
        ("key",),
    ),
    tool(
        "press_keys",
        "Press permitted keys SIMULTANEOUSLY for a duration.",
        {"keys": {"type": "array", "items": TEXT}, "duration_ms": NUMBER},
        ("keys",),
    ),
    tool(
        "press_sequence",
        "Execute permitted keys IN ORDER without a model request between steps. "
        "Each step is released before the next; the main agent has priority over "
        "reflexes. An episode transition interrupts the sequence.",
        {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"key": TEXT, "duration_ms": NUMBER},
                    "required": ["key"],
                },
            }
        },
        ("steps",),
    ),
    tool(
        "click",
        "Click at screenshot coordinates if permitted by this game's controls.",
        {"x": NUMBER, "y": NUMBER},
        ("x", "y"),
    ),
    tool(
        "click_grid",
        "Click a visually chosen position on a 1000 by 1000 coordinate grid over "
        "the WHOLE screenshot. Both coordinates range from 0 to 1000.",
        {"x": NUMBER, "y": NUMBER},
        ("x", "y"),
    ),
    tool("get_action_history", "Read the most recent real executed action records."),
    tool(
        "inspect_success",
        "Load the previous successful episode's screenshot sequence, executed actions "
        "and official outcome into your next observation.",
    ),
    tool(
        "remember",
        "Save a concise observation, plan or reusable lesson across decisions and episodes.",
        {"content": TEXT},
        ("content",),
    ),
    tool(
        "list_rules",
        "Read the installed reflex names, enable state and intervention counts.",
    ),
    tool(
        "recall_rule_memory",
        "Read memories associated with visual event detectors.",
        {"memory_key": TEXT},
    ),
    tool(
        "stop_rule",
        "Immediately stop an installed conditional reflex and release its current input.",
        {"name": TEXT},
        ("name",),
    ),
]
MAIN_TOOLS = list(MAIN_GAME_TOOLS)


def image_part(data: bytes) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()},
    }


def main_visual_view(data: bytes) -> tuple[bytes, tuple[int, int, int, int]]:
    with Image.open(BytesIO(data)) as picture:
        bounds = (0, 0, picture.width, picture.height)
    return data, bounds


def main_image_part(data: bytes) -> dict:
    view, _ = main_visual_view(data)
    return image_part(view)


_LOCAL_MODEL_ENDPOINT_LOCKS: dict[str, asyncio.Lock] = {}


class ModelClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        events,
        directory: Path,
        *,
        backend: str = "openai",
        codex_config: str | Path | None = None,
        codex_auth: str | Path | None = None,
        api_slot_dir: str | Path | None = None,
        api_slot_limit: int = 2,
    ) -> None:
        self.backend = backend
        self.auth_headers: dict[str, str] = {}
        if backend == "codex-responses":
            endpoint, codex_model, key = load_codex_responses_settings(codex_config, codex_auth)
            if model != codex_model:
                raise ValueError("Requested model differs from configured Responses model")
            self.auth_headers = {"Authorization": f"Bearer {key}"}
        elif backend in {"openai", "ninfer"}:
            api_key = os.environ.get("VICOR_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
            if api_key:
                self.auth_headers = {"Authorization": f"Bearer {api_key}"}
        else:
            raise ValueError(f"Unknown model backend: {backend}")
        self.endpoint, self.model, self.events = endpoint, model, events
        self.api_slot_dir = api_slot_dir if backend == "codex-responses" else None
        self.api_slot_limit = api_slot_limit
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=12 if backend == "codex-responses" else 10,
                read=None,
                write=180,
                pool=180,
            )
        )
        self.sequence = 0

    async def ask(
        self,
        messages: list,
        tools: list,
        *,
        role: str,
        max_tokens: int,
        effort: str = "low",
        reasoning_budget_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.sequence += 1
        number = self.sequence
        began = time.monotonic()
        if self.backend == "codex-responses":
            payload: dict[str, Any] = {
                "model": self.model,
                "input": to_responses_input(messages),
                "reasoning": {"effort": effort},
                "max_output_tokens": max(max_tokens, 4096 if role == "main" else 8192),
                "temperature": 0,
                "store": False,
                "stream": True,
            }
            if tools:
                payload.update({"tools": to_responses_tools(tools), "tool_choice": "auto"})
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "reasoning_effort": effort,
                "temperature": 0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {
                    "enable_thinking": effort != "none",
                    "preserve_thinking": role in {"main", "light_learner"} and effort != "none",
                },
            }
            if reasoning_budget_tokens is not None:
                payload["reasoning_budget_tokens"] = reasoning_budget_tokens
            if tools:
                payload.update({"tools": tools, "tool_choice": "auto"})
        self.events.add(
            "model_request",
            role=role,
            request_id=number,
            model=self.model,
            backend=self.backend,
            generation={
                "reasoning_effort": effort,
                "reasoning_budget_tokens": payload.get("reasoning_budget_tokens"),
                "max_output_tokens": payload.get("max_output_tokens", payload.get("max_tokens")),
                "temperature": payload["temperature"],
                **payload.get("chat_template_kwargs", {}),
            },
        )
        if self.backend == "codex-responses":
            async with api_slot(self.api_slot_dir, self.api_slot_limit):
                async with aconnect_sse(
                    self.http,
                    "POST",
                    self.endpoint,
                    json=payload,
                    headers={"X-Agent-ID": role, **self.auth_headers},
                ) as stream:
                    stream.response.raise_for_status()
                    data = None
                    event_count = 0
                    async for event in stream.aiter_sse():
                        if not event.data or event.data == "[DONE]":
                            continue
                        part = event.json()
                        event_count += 1
                        if event_count == 1:
                            self.events.add(
                                "model_stream_started",
                                role=role,
                                request_id=number,
                                elapsed_s=time.monotonic() - began,
                            )
                        if part.get("type") in {"error", "response.failed"}:
                            raise RuntimeError(json.dumps(part, ensure_ascii=False))
                        if part.get("type") in {"response.completed", "response.incomplete"}:
                            data = part["response"]
                            break
                    if data is None:
                        raise RuntimeError("Model response stream ended without a complete response")
            self.events.add("model_stream_finished", role=role, request_id=number, events_received=event_count)
        else:
            lock = _LOCAL_MODEL_ENDPOINT_LOCKS.setdefault(self.endpoint, asyncio.Lock())
            async with lock:
                for _attempt in range(240):
                    response = await self.http.post(
                        self.endpoint,
                        json=payload,
                        headers={"X-Agent-ID": role, **self.auth_headers},
                    )
                    if response.status_code not in (429, 502, 503, 504):
                        break
                    await asyncio.sleep(0.25)
            if response.is_error:
                self.events.add(
                    "model_http_error",
                    role=role,
                    request_id=number,
                    status_code=response.status_code,
                    response_body=response.text[:3000],
                    elapsed_s=time.monotonic() - began,
                )
            response.raise_for_status()
            data = response.json()
        if self.backend == "codex-responses":
            assistant, usage, finish_reason = from_responses_output(data)
        else:
            choice = data["choices"][0]
            assistant, usage, finish_reason = (
                choice["message"],
                data.get("usage"),
                choice.get("finish_reason"),
            )
        record = self.events.add(
            "model_response",
            role=role,
            request_id=number,
            model=self.model,
            backend=self.backend,
            elapsed_s=time.monotonic() - began,
            usage=usage,
            timings=data.get("timings"),
            finish_reason=finish_reason,
            message=assistant,
        )
        path = self.directory / "model_responses"
        path.mkdir(exist_ok=True)
        (path / f"{record['id']:08d}_{role}_{number:05d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.backend == "codex-responses" and finish_reason == "length":
            raise RuntimeError(
                f"Model output reached budget without completing: role={role}, "
                f"output_tokens={usage.get('completion_tokens')}, "
                f"reasoning_tokens={usage.get('completion_tokens_details', {}).get('reasoning_tokens')}"
            )
        return assistant

    async def close(self) -> None:
        await self.http.aclose()
