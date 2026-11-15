"""OpenAI-compatible tool schema projection and multi-turn conversation state."""

from __future__ import annotations

import copy
import json


class ChainConversation:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.prompt_tokens: int = 0
        self.checkpoint_pending: bool = False
        self.checkpoint_attempts: int = 0


def image_content(images: list[dict]) -> list[dict]:
    content: list[dict] = []
    for image in images:
        content.extend([
            {"type": "text", "text": image["steer_text"]},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image['mime_type']};base64,{image['data']}"},
            },
        ])
    return content


def model_tools(runtime) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in runtime.tools()
    ]


def decode_arguments(call: dict):
    value = call["function"]["arguments"]
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def project_tool_schemas(tools: list[dict]) -> list[dict]:
    """Project `oneOf` tool parameter branches onto root properties for OpenAI-compatible servers."""
    result = copy.deepcopy(tools)
    for tool in result:
        schema = tool.get("function", {}).get("parameters", {})
        branches = schema.get("oneOf")
        if (
            not isinstance(branches, list)
            or not branches
            or schema.get("additionalProperties", True) is not True
            or "unevaluatedProperties" in schema
            or any(
                not isinstance(branch, dict)
                or branch.get("additionalProperties") is not False
                or branch.get("patternProperties")
                for branch in branches
            )
        ):
            continue
        alternatives: dict[str, dict[str, dict]] = {}
        for branch in branches:
            for name, definition in branch.get("properties", {}).items():
                alternatives.setdefault(name, {})[json.dumps(definition, sort_keys=True)] = definition
        properties = schema.setdefault("properties", {})
        for name, definitions in alternatives.items():
            if name in properties:
                continue
            values = list(definitions.values())
            if len(values) == 1:
                properties[name] = values[0]
            elif all(isinstance(value, dict) and "const" in value for value in values):
                constants = {json.dumps(value["const"], sort_keys=True): value["const"] for value in values}
                properties[name] = {"enum": list(constants.values())}
            else:
                properties[name] = {"anyOf": values}
        common = set.intersection(*(set(branch.get("required", [])) for branch in branches))
        required = set(schema.get("required", [])) | common
        if required:
            schema["required"] = sorted(required)
    return result
