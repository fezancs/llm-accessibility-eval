"""OpenAI Responses API adapter (streaming).

The Responses API is OpenAI's native interface for reasoning models: with store=False and
include=["reasoning.encrypted_content"], reasoning items are returned and replayed across
tool calls, which Chat Completions cannot do. Evaluated calls send no `reasoning`, so the
model's default effort applies; judges pass models.yaml `judge_params`.
"""
from __future__ import annotations

import json
import os
import time

import openai

from .base import AttemptError, Outcome, empty_usage, image_b64, neutral_calls, neutral_text, parse_arguments

_client = None


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0, timeout=900.0)
    return _client


def to_input(messages: list[dict]) -> list[dict]:
    items: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            content = []
            for p in m["content"]:
                if p["type"] == "text":
                    content.append({"type": "input_text", "text": p["text"]})
                else:
                    data, mt = image_b64(p)
                    content.append({"type": "input_image", "image_url": f"data:{mt};base64,{data}"})
            items.append({"role": "user", "content": content})
        elif m["role"] == "assistant" and m.get("provider") == "neutral":
            if neutral_text(m).strip():
                items.append({"role": "assistant", "content": neutral_text(m)})
            items.extend({"type": "function_call", "call_id": c["id"], "name": c["name"],
                          "arguments": json.dumps(c.get("arguments") or {})} for c in neutral_calls(m))
        elif m["role"] == "assistant":
            if m.get("provider") != "openai":
                raise ValueError("assistant turn from another provider cannot be replayed to OpenAI")
            items.extend(m["raw"])
        elif m["role"] == "tool":
            items.extend({"type": "function_call_output", "call_id": r["tool_call_id"],
                          "output": r["content"]} for r in m["results"])
    return items


def _usage(u) -> dict:
    usage = empty_usage()
    if u is None:
        return usage
    cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
    usage["input_tokens"] = (u.input_tokens or 0) - cached
    usage["cached_input_tokens"] = cached
    usage["output_tokens"] = u.output_tokens or 0
    usage["reasoning_tokens"] = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
    return usage


def _classify(exc: Exception) -> tuple[bool, int | None]:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True, None
    if isinstance(exc, openai.APIStatusError):
        code = exc.status_code
        return (code in (408, 409, 429) or code >= 500), code
    return True, None


def call(model_cfg: dict, system: str | None, messages: list[dict], tools: list[dict] | None,
         max_tokens: int, extra_params: dict, cache: str, t0_holder: list) -> Outcome:
    kwargs = {"model": model_cfg["provider_model_id"], "input": to_input(messages),
              "max_output_tokens": max_tokens, "store": False,
              "include": ["reasoning.encrypted_content"], "stream": True}
    if system:
        kwargs["instructions"] = system
    if tools:
        kwargs["tools"] = [{"type": "function", "name": t["name"], "description": t["description"],
                            "parameters": t["input_schema"], "strict": False} for t in tools]
    kwargs.update(extra_params)
    params_sent = {k: v for k, v in kwargs.items() if k not in ("input", "instructions", "tools")}

    first = None
    final = None
    t0_holder.append(time.monotonic())
    try:
        for event in client().responses.create(**kwargs):
            et = event.type
            if first is None and et in ("response.output_text.delta", "response.function_call_arguments.delta",
                                        "response.reasoning_summary_text.delta", "response.reasoning_text.delta",
                                        "response.refusal.delta"):
                first = time.monotonic()
            elif et in ("response.completed", "response.incomplete"):
                final = event.response
            elif et == "response.failed":
                err = event.response.error
                raise AttemptError(f"response.failed: {err}", True, None, _usage(event.response.usage))
            elif et == "error":
                raise AttemptError(f"stream error: {getattr(event, 'message', event)}", True, None)
    except AttemptError:
        raise
    except Exception as exc:        # noqa: BLE001
        retryable, code = _classify(exc)
        raise AttemptError(f"{type(exc).__name__}: {exc}", retryable, code) from exc
    if final is None:
        raise AttemptError("stream ended without response.completed", True, None)

    text, tool_calls, refused = [], [], False
    for item in final.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    text.append(c.text)
                elif c.type == "refusal":
                    refused = True
                    text.append(c.refusal)
        elif item.type == "function_call":
            args, ok = parse_arguments(item.arguments)
            tool_calls.append({"id": item.call_id, "name": item.name, "arguments": args, "valid": ok,
                               "arguments_raw": item.arguments})
    reason = getattr(final.incomplete_details, "reason", None) if final.incomplete_details else None
    if refused or reason == "content_filter":
        stop = "refusal"
    elif reason == "max_output_tokens":
        stop = "max_tokens"
    elif tool_calls:
        stop = "tool_use"
    else:
        stop = "end_turn"
    return Outcome(
        text="".join(text), tool_calls=tool_calls,
        raw_assistant=[i.model_dump(mode="json", exclude_none=True) for i in final.output],
        stop_reason=stop, usage=_usage(final.usage), first_token_t=first,
        model_returned=final.model, params_sent=params_sent,
        raw_response=final.model_dump(mode="json", exclude_none=True))
