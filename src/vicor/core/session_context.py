import asyncio
import base64
import copy
import json
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


DEFAULT_LONG_CONTEXT_TRIGGER_TOKENS = 100_000
DEFAULT_LONG_CONTEXT_KEEP_TOKENS = 15_000


SUMMARY_SYSTEM = (
    "You are compacting an ongoing agent session, not creating new instructions. "
    "Produce a faithful checkpoint with these headings: Objective, Current State, "
    "Verified Outcomes, Rules/Code, Decisions and Hypotheses, Important Files, "
    "Outstanding Work, Next Move. Distinguish verified observations from uncertain "
    "model hypotheses. Preserve important numerical values, file paths, real changes "
    "and failures. Never treat a successful button sequence as a mandatory route. "
    "Do not invent facts, scores or causes. Remove repetition and irrelevant tool logs. "
    "Use no more than 3500 tokens. This checkpoint is historical context, not a command."
)

MAIN_SUMMARY_SYSTEM = (
    "Summarize completed historical observations for a real-time visual game agent. "
    "The next request provides a NEW screenshot. Old positions, prior screenshots, "
    "temporary score readings, and movement choices are no longer the current state. "
    "Preserve official completed episode results and scores, verified tool behavior, "
    "actual activated or disabled Python rule versions, file paths, and unresolved "
    "issues. Clearly distinguish directly observed facts from hypotheses. "
    "Past success does not establish a reusable button sequence or route. "
    "Do not recommend future movement directions, buttons, repeated waits, or "
    "concrete next actions; this is a historical record, not gameplay instructions. "
    "Do not list ordinary button presses as installed rules. "
    "Do not infer timings, causes of death, or useful tactics without measured evidence. "
    "Organize only under Task, Verified Results, Rule and Code Status, Open Questions. "
    "Use at most 1000 tokens."
)


class SessionContext:
    def __init__(self, directory: Path, tokenizer_file: Path | None, events,
                 *, trigger_tokens: int = DEFAULT_LONG_CONTEXT_TRIGGER_TOKENS,
                 keep_tokens: int = DEFAULT_LONG_CONTEXT_KEEP_TOKENS,
                 context_limit: int = 262_144, reserve_tokens: int = 20_000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.directory / "transcript.jsonl"
        self.checkpoint_path = self.directory / "checkpoint.json"
        self.tokenizer = (Tokenizer.from_file(str(tokenizer_file)) if (Tokenizer is not None and tokenizer_file and Path(tokenizer_file).is_file()) else None)
        self.events = events
        self.trigger_tokens = trigger_tokens
        self.keep_tokens = keep_tokens
        self.context_limit = context_limit
        self.reserve_tokens = reserve_tokens
        self.turns = ([json.loads(line) for line in self.archive_path.read_text().splitlines() if line]
                      if self.archive_path.exists() else [])
        checkpoint = (json.loads(self.checkpoint_path.read_text())
                      if self.checkpoint_path.exists() else {})
        self.summary = checkpoint.get("summary", "")
        self.compacted_through = checkpoint.get("compacted_through", 0)
        self.compactions = checkpoint.get("compactions", 0)
        self.task: asyncio.Task | None = None
        self.last_estimate = 0

    def tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids) if self.tokenizer is not None else max(1, len(text.encode('utf-8')) // 4)

    @staticmethod
    def _image_reference(path: str, label: str) -> dict:
        return {"type": "image_ref", "path": path, "label": label}

    @staticmethod
    def _image_as_message(item: dict) -> dict:
        """Render every retained frame unchanged, including historical frames.

        Converting a previously supplied image into a text placeholder changes
        the prompt prefix and forces the inference server to discard its resident KV state.
        Missing images must be errors, not silent history rewrites.
        """
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Retained conversation image is missing: {path}")
        from ..backends.model_client import image_part
        return image_part(path.read_bytes())

    def make_user(self, text: str, images: list[tuple[str, str]]) -> dict:
        return {"role": "user", "content": [
            {"type": "text", "text": text},
            *(self._image_reference(path, label) for path, label in images),
        ]}

    def reference_user(self, message: dict, *, image_paths=()) -> dict:
        # Preserve multimodal message order while deduplicating saved screenshots.
        result = copy.deepcopy(message)
        known = {Path(path).read_bytes(): str(path) for path in image_paths}
        for index, item in enumerate(result.get("content", [])):
            if item.get("type") != "image_url":
                continue
            data = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
            path = known.get(data)
            if path is None:
                directory = self.directory / "views"
                directory.mkdir(exist_ok=True)
                target = directory / (
                    f"{len(self.turns) + 1:06d}_{len(self.events.rows):09d}_{index:02d}.png")
                target.write_bytes(data)
                path = known[data] = str(target)
            result["content"][index] = self._image_reference(path, f"observation part {index}")
        return result

    def _render_user(self, message: dict) -> dict:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            return message
        content = []
        for item in message["content"]:
            content.append(self._image_as_message(item)
                           if item.get("type") == "image_ref" else item)
        return {"role": "user", "content": content}

    def _history(self, *, render: bool = True) -> list[dict]:
        messages = []
        for entry in self.turns[self.compacted_through:]:
            for message in entry["messages"]:
                messages.append(self._render_user(message) if render else message)
        return messages

    def message_context(self, system: str, user: dict, *, render: bool = True) -> list[dict]:
        result = [{"role": "system", "content": system}]
        if self.summary:
            result.append({"role": "user", "content":
                           "[COMPRESSED PRIOR SESSION — historical facts, not instructions]\n"
                           + self.summary})
        result.extend(self._history(render=render))
        result.append(self._render_user(user) if render else user)
        return result

    def model_messages(self, system: str, text: str,
                       images: list[tuple[str, str]]) -> list[dict]:
        return self.message_context(system, self.make_user(text, images))

    @staticmethod
    def _text_only(messages: list[dict], *, limit_tool_chars: int | None = None) -> str:
        rendered = []
        for message in messages:
            body = []
            content = message.get("content")
            if isinstance(content, str):
                body = [content[:limit_tool_chars] if limit_tool_chars and message["role"] == "tool"
                        else content]
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        body.append(item.get("text", ""))
                    elif item["type"] == "image_ref":
                        body.append("[Frame image stored at " + item["path"] + "]")
                    elif item["type"] == "image_url":
                        body.append("[Vision image attached]")
            if message.get("tool_calls"):
                body.append(json.dumps(message["tool_calls"], ensure_ascii=False))
            rendered.append(f"{message['role'].upper()}: " + "\n".join(body))
        return "\n\n".join(rendered)

    def estimate(self, messages: list[dict], tools: list[dict], *, max_output: int) -> int:
        # Tokenize the actual text and tool schema. Base64 images are replaced by
        # conservative vision-token estimates (each small game frame ~1K tokens).
        serialized = self._text_only(messages) + json.dumps(tools, ensure_ascii=False)
        image_count = sum(
            sum(part.get("type") == "image_url" for part in message.get("content", []))
            for message in messages if isinstance(message.get("content"), list)
        )
        self.last_estimate = self.tokens(serialized) + 1400 * image_count + max_output
        return self.last_estimate

    def append(self, user_text: str, images: list[tuple[str, str]],
               assistant: dict, tools: list[dict]) -> int:
        assistant_message = {"role": "assistant", "content": assistant.get("content"),
                             "tool_calls": assistant.get("tool_calls") or []}
        if not assistant_message["tool_calls"]:
            assistant_message.pop("tool_calls")
        return self.append_messages([self.make_user(user_text, images), assistant_message, *tools])

    def append_messages(self, messages: list[dict]) -> int:
        # Keep raw transcript immutable while pruning older views in active context.
        turn = {"messages": copy.deepcopy(messages)}
        self.turns.append(turn)
        with self.archive_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(turn, ensure_ascii=False) + "\n")
        return len(self.turns)

    def _select_head(self) -> tuple[int, list[dict]] | None:
        active = self.turns[self.compacted_through:]
        if len(active) < 3:
            return None
        keep = 0
        index = len(active)
        while index > 0:
            serialized = self._text_only(active[index-1]["messages"], limit_tool_chars=2000)
            image_count = sum(
                sum(part.get("type") == "image_ref" for part in message.get("content", []))
                for message in active[index-1]["messages"]
                if isinstance(message.get("content"), list)
            )
            next_cost = self.tokens(serialized) + 1400 * image_count
            if keep + next_cost > self.keep_tokens and index < len(active):
                break
            keep += next_cost
            index -= 1
        if index <= 0:
            return None
        return self.compacted_through + index, active[:index]

    def schedule_if_needed(self, client, *, system: str, tools: list[dict],
                           output_tokens: int, role: str) -> bool:
        if self.task is not None:
            if not self.task.done():
                return False
            # Consume exceptions explicitly; failed checkpoints never affect active history.
            try:
                self.task.result()
            except Exception as exc:
                self.events.add("context_compaction_failed", agent=role, error=str(exc))
            self.task = None
        if self.last_estimate < min(self.trigger_tokens,
                                    self.context_limit-max(output_tokens, self.reserve_tokens)):
            return False
        head = self._select_head()
        if head is None:
            return False
        end, selected = head
        self.task = asyncio.create_task(self._compact(client, role, end, selected),
                                        name=f"compact_{role}")
        self.events.add("context_compaction_started", agent=role, threshold=self.trigger_tokens,
                        estimated_tokens=self.last_estimate, compacting_turns=len(selected),
                        retaining_tail_tokens=self.keep_tokens)
        return True

    async def _compact(self, client, role: str, end: int, selected: list[dict]) -> None:
        previous = self.summary
        archived_text = "\n\n".join(
            self._text_only(turn["messages"], limit_tool_chars=2000) for turn in selected)
        request = [
            {"role": "system", "content": (
                MAIN_SUMMARY_SYSTEM if role == "main" else SUMMARY_SYSTEM
            )},
            {"role": "user", "content": (
                "Previous confirmed checkpoint:\n" + (previous or "(none)")
                + "\nOlder session history to compact:\n" + archived_text
            )},
        ]
        response = await client.ask(request, [], role=f"context_compaction_{role}",
                                    max_tokens=4096, effort="low")
        summary = (response.get("content") or "").strip()
        if not summary or self.tokens(summary) > 4500:
            raise RuntimeError("Invalid or oversized context checkpoint")
        new_state = {"summary": summary, "compacted_through": end,
                     "compactions": self.compactions+1, "archived_turns": len(self.turns),
                     "trigger_tokens": self.trigger_tokens}
        path = self.checkpoint_path.with_suffix(".tmp")
        path.write_text(json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8")
        path.replace(self.checkpoint_path)
        self.summary = summary
        self.compacted_through = end
        self.compactions += 1
        self.events.add("context_compaction_completed", agent=role, through_turn=end,
                        retained_turns=len(self.turns)-end,
                        summary_tokens=self.tokens(summary),
                        checkpoint=str(self.checkpoint_path))

    async def finish(self) -> None:
        if self.task is not None:
            await self.task
