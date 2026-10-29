"""Anthropic Messages API adapter (streaming).

Choices that matter for the study:
- Evaluated calls send no `thinking` and no `output_config`, so each Claude model runs at
  its documented default (adaptive thinking; default effort). Judges pass the lowest
  setting from models.yaml `judge_params` (Sonnet 5.5: thinking between_tools, effort low).
- No server-side refusal `fallbacks`: a fallback would answer with a different model and
  contaminate the evaluation. A refusal is recorded as its own outcome.
- `eager_input_streaming` is left off, so the API validates tool-input JSON for Claude as
  the other providers' APIs do; malformed-call rates stay comparable across vendors.
- Prompt caching: a breakpoint after the system prompt, explicit breakpoints on parts
  marked `cache_breakpoint`, and top-level automatic caching when cache == "auto"
  (multi-turn sessions, where the history is a shared prefix for the next call).
"""
from __future__ import annotations

import os
import time

import anthropic

from .base import AttemptError, Outcome, empty_usage, image_b64, neutral_calls, neutral_text

_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Retries are done by harness.llm so every attempt is logged and billed.
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"),
                                      max_retries=0, timeout=900.0)
    return _client


def _user_blocks(parts: list[dict]) -> list[dict]:
    out = []
    for p in parts:
        if p["type"] == "text":
            b = {"type": "text", "text": p["text"]}
        elif p["type"] == "image":
            data, mt = image_b64(p)
            b = {"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}}
        else:
            raise ValueError(f"unsupported part {p['type']}")
        if p.get("cache_breakpoint"):
            b["cache_control"] = {"type": "ephemeral"}
        out.append(b)
    return out


def to_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": _user_blocks(m["content"])})
        elif m["role"] == "assistant" and m.get("provider") == "neutral":
            blocks = ([{"type": "text", "text": neutral_text(m)}] if neutral_text(m).strip() else []) + [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c.get("arguments") or {}}
                for c in neutral_calls(m)]
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "assistant":
            if m.get("provider") != "anthropic":
                raise ValueError("assistant turn from another provider cannot be replayed to Anthropic")
            out.append({"role": "assistant", "content": m["raw"]})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r["tool_call_id"], "content": r["content"],
                 **({"is_error": True} if r.get("is_error") else {})} for r in m["results"]]})
        else:
            raise ValueError(m["role"])
    return out


def _usage(u) -> dict:
    usage = empty_usage()
    if u is None:
        return usage
    usage["input_tokens"] = getattr(u, "input_tokens", 0) or 0
    usage["cached_input_tokens"] = getattr(u, "cache_read_input_tokens", 0) or 0
    usage["cache_write_tokens"] = getattr(u, "cache_creation_input_tokens", 0) or 0
    usage["output_tokens"] = getattr(u, "output_tokens", 0) or 0
    return usage


def _classify(exc: Exception) -> tuple[bool, int | None]:
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True, None
    if isinstance(exc, anthropic.APIStatusError):
        code = exc.status_code
        return (code in (408, 409, 429) or code >= 500), code
    return True, None          # stream interrupted by a transport error


def call(model_cfg: dict, system: str | None, messages: list[dict], tools: list[dict] | None,
         max_tokens: int, extra_params: dict, cache: str, t0_holder: list) -> Outcome:
    kwargs = {"model": model_cfg["provider_model_id"], "max_tokens": max_tokens,
              "messages": to_messages(messages)}
    if system:
        block = {"type": "text", "text": system}
        if cache != "none":
            block["cache_control"] = {"type": "ephemeral"}
        kwargs["system"] = [block]
    if tools:
        kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                            "input_schema": t["input_schema"]} for t in tools]
    if cache == "auto":
        kwargs["cache_control"] = {"type": "ephemeral"}
    kwargs.update(extra_params)          # judge_params only (thinking / output_config)
    params_sent = {k: v for k, v in kwargs.items() if k not in ("messages", "system", "tools")}

    partial = empty_usage()
    first = None
    t0_holder.append(time.monotonic())
    try:
        with client().messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "message_start":
                    partial = _usage(event.message.usage)
                elif event.type == "content_block_delta" and first is None:
                    first = time.monotonic()
                elif event.type == "message_delta" and event.usage is not None:
                    partial["output_tokens"] = event.usage.output_tokens or partial["output_tokens"]
            final = stream.get_final_message()
    except Exception as exc:        # noqa: BLE001 - classified below, re-raised as AttemptError
        retryable, code = _classify(exc)
        raise AttemptError(f"{type(exc).__name__}: {exc}", retryable, code, partial) from exc

    text, tool_calls = [], []
    for b in final.content:
        if b.type == "text":
            text.append(b.text)
        elif b.type == "tool_use":
            tool_calls.append({"id": b.id, "name": b.name, "arguments": b.input
                               if isinstance(b.input, dict) else None,
                               "valid": isinstance(b.input, dict)})
    stop = final.stop_reason or "end_turn"
    return Outcome(
        text="".join(text), tool_calls=tool_calls,
        raw_assistant=[b.model_dump(mode="json", exclude_none=True) for b in final.content],
        stop_reason=stop, usage=_usage(final.usage), first_token_t=first,
        model_returned=final.model, params_sent=params_sent,
        raw_response=final.model_dump(mode="json", exclude_none=True),
        extra={"stop_details": final.stop_details.model_dump(mode="json")
               if getattr(final, "stop_details", None) else None})
