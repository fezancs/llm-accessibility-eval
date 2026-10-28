"""Shared types for the provider adapters.

Internal, provider-neutral conversation format used by every experiment:

    {"role": "user", "content": [Part, ...]}
    {"role": "assistant", "content": [...], "provider": "<adapter>", "raw": <adapter-specific turn>}
    {"role": "tool", "results": [{"tool_call_id", "name", "content": str, "is_error": bool}]}

    Part = {"type": "text", "text": str, "cache_breakpoint": bool (optional)}
         | {"type": "image", "path": str, "media_type": "image/jpeg" | ...}

An assistant turn is replayed to the same provider from "raw" unchanged, so thinking
blocks, reasoning items and thought signatures survive tool loops. An assistant turn with
provider "neutral" (no raw; text in "content", calls in "tool_calls" as {id, name,
arguments}) can be sent to any provider: E2's padded control replays a transcript that one
model produced to every model.
"""
from __future__ import annotations

import base64
import json
import pathlib
from dataclasses import dataclass, field


class AttemptError(Exception):
    """One failed attempt. `usage` holds whatever token counts arrived before the failure."""

    def __init__(self, message: str, retryable: bool, status_code: int | None = None,
                 usage: dict | None = None, kind: str = "api_error"):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.usage = usage or {}
        self.kind = kind


@dataclass
class Outcome:
    text: str
    tool_calls: list[dict]
    raw_assistant: object
    stop_reason: str                 # end_turn | tool_use | max_tokens | refusal | <provider value>
    usage: dict
    first_token_t: float | None      # time.monotonic() of the first streamed output delta
    model_returned: str | None
    params_sent: dict
    raw_response: object
    provider_served: str | None = None
    provider_reported_cost: float | None = None
    extra: dict = field(default_factory=dict)


def empty_usage() -> dict:
    return {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_tokens": 0,
            "output_tokens": 0, "reasoning_tokens": None}


def image_b64(part: dict) -> tuple[str, str]:
    data = pathlib.Path(part["path"]).read_bytes()
    return base64.standard_b64encode(data).decode("ascii"), part.get("media_type", "image/jpeg")


def parse_arguments(raw: str | dict | None) -> tuple[dict | None, bool]:
    """Parse a tool call's arguments. Returns (arguments, valid)."""
    if isinstance(raw, dict):
        return raw, True
    if raw is None or raw == "":
        return {}, True
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, False
    return (val, True) if isinstance(val, dict) else (None, False)


def neutral_text(m: dict) -> str:
    return "".join(p.get("text", "") for p in m.get("content", []) if p.get("type") == "text")


def neutral_calls(m: dict) -> list[dict]:
    return [c for c in m.get("tool_calls") or [] if c.get("name")]
