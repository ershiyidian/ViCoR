"""Stateless OpenAI Responses adapter for a visual, tool-using game agent.

Codex's configured provider is authoritative for its API root and credential.
Only the endpoint in that local configuration receives the credential. This
module has no network or file-system side effects while converting messages.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from urllib.parse import urlsplit


def load_codex_responses_settings(
    config_file: str | Path | None = None,
    auth_file: str | Path | None = None,
) -> tuple[str, str, str]:
    config_path = Path(config_file or Path.home() / ".codex/config.toml").expanduser()
    auth_path = Path(auth_file or Path.home() / ".codex/auth.json").expanduser()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    provider_name = config["model_provider"]
    provider = config["model_providers"][provider_name]
    if provider.get("wire_api") != "responses":
        raise ValueError("Configured Codex provider does not use the Responses API")
    base_url = provider["base_url"].rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Configured Codex API URL must be an HTTPS provider root")
    key = json.loads(auth_path.read_text(encoding="utf-8")).get("OPENAI_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Codex authentication lacks OPENAI_API_KEY")
    endpoint = base_url if base_url.endswith("/responses") else base_url + "/responses"
    return endpoint, config["model"], key


def to_responses_input(messages: list[dict]) -> list[dict]:
    """Convert a complete Chat transcript without editing old multimodal turns."""
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            converted.append({
                "type": "function_call_output",
                "call_id": message["tool_call_id"],
                "output": message.get("content", "") or "",
            })
            continue
        if role not in ("system", "developer", "user", "assistant"):
            raise ValueError(f"Unsupported conversation role: {role}")
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for item in content:
                if item["type"] == "text":
                    blocks.append({
                        "type": "output_text" if role == "assistant" else "input_text",
                        "text": item.get("text", ""),
                    })
                elif item["type"] == "image_url":
                    if role not in ("user", "developer"):
                        raise ValueError("Images may only appear in user/developer turns")
                    blocks.append({
                        "type": "input_image",
                        "image_url": item["image_url"]["url"],
                        "detail": item["image_url"].get("detail", "auto"),
                    })
                else:
                    raise ValueError(f"Unsupported multimodal block: {item['type']}")
            content = blocks
        if content not in (None, "", []):
            converted.append({"role": role, "content": content})
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            converted.append({
                "type": "function_call",
                "call_id": call["id"],
                "name": fn["name"],
                "arguments": fn["arguments"],
            })
    return converted


def to_responses_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "name": item["function"]["name"],
            "description": item["function"].get("description", ""),
            "parameters": item["function"]["parameters"],
        }
        for item in tools
    ]


def from_responses_output(data: dict) -> tuple[dict, dict, str]:
    """Normalize DeepSeek/OpenAI Responses into existing Chat tool-call shape."""
    content: list[str] = []
    calls: list[dict] = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    content.append(part.get("text", ""))
        elif item.get("type") == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if not isinstance(call_id, str):
                raise ValueError("Responses function_call has no call_id")
            args = item.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": item["name"], "arguments": args},
            })
    usage = data.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    normalized_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_tokens_details": {
            "cached_tokens": input_details.get("cached_tokens", 0),
        },
        "completion_tokens_details": usage.get("output_tokens_details") or {},
    }
    status = data.get("status", "completed")
    if status not in ("completed", "incomplete"):
        raise RuntimeError(f"Responses model did not complete: {status}")
    reason = "tool_calls" if calls else ("length" if status == "incomplete" else "stop")
    message = {
        "role": "assistant",
        "content": "\n".join(content) if content else None,
        "tool_calls": calls,
    }
    return message, normalized_usage, reason
