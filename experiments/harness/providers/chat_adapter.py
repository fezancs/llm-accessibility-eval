"""Chat Completions adapter for OpenAI-compatible endpoints: DeepSeek's native API and
OpenRouter (open-weight models).

OpenRouter calls pin one upstream host with no fallbacks (models.yaml `openrouter_provider`)
and log the host that served each call. DeepSeek's `reasoning_content` and OpenRouter's
`reasoning_details` are kept in the assistant turn and replayed within a tool loop.
"""
from __future__ import annotations

import json
import os
import time

import openai

from .base import AttemptError, Outcome, empty_usage, image_b64, neutral_calls, neutral_text, parse_arguments

ENDPOINTS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}
_clients: dict[str, openai.OpenAI] = {}


def client(provider: str) -> openai.OpenAI:
    if provider not in _clients:
        base, env = ENDPOINTS[provider]
        _clients[provider] = openai.OpenAI(base_url=base, api_key=os.environ.get(env),
                                           max_retries=0, timeout=900.0)
    return _clients[provider]


def to_messages(system: str | None, messages: list[dict], provider: str) -> list[dict]:
    out = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m["role"] == "user":
            if all(p["type"] == "text" for p in m["content"]):
                out.append({"role": "user", "content": "".join(p["text"] for p in m["content"])})
            else:
                content = []
                for p in m["content"]:
                    if p["type"] == "text":
                        content.append({"type": "text", "text": p["text"]})
                    else:
                        data, mt = image_b64(p)
                        content.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}})
                out.append({"role": "user", "content": content})
        elif m["role"] == "assistant" and m.get("provider") == "neutral":
            msg = {"role": "assistant", "content": neutral_text(m) or None}
            calls = neutral_calls(m)
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                    "name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}} for c in calls]
            if msg["content"] is not None or calls:
                out.append(msg)
        elif m["role"] == "assistant":
            if m.get("provider") != provider:
                raise ValueError("assistant turn from another provider cannot be replayed here")
            out.append(m["raw"])
        elif m["role"] == "tool":
            out.extend({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]}
                       for r in m["results"])
    return out


def _usage(u) -> dict:
    usage = empty_usage()
    if u is None:
        return usage
    extra = u.model_extra or {}
    prompt = u.prompt_tokens or 0
    if "prompt_cache_hit_tokens" in extra:                       # DeepSeek
        cached = extra.get("prompt_cache_hit_tokens") or 0
    else:
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    usage["input_tokens"] = prompt - cached
    usage["cached_input_tokens"] = cached
    usage["output_tokens"] = u.completion_tokens or 0
    usage["reasoning_tokens"] = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
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
    provider = model_cfg["provider"]
    model_id = model_cfg["openrouter_id"] if provider == "openrouter" else model_cfg["provider_model_id"]
    kwargs = {"model": model_id, "messages": to_messages(system, messages, provider),
              "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        kwargs["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]
    extra_body = {}
    if provider == "openrouter":
        host = model_cfg.get("openrouter_provider")
        if not host:
            raise AttemptError(f"{model_cfg['key']}: openrouter_provider not set in models.yaml "
                               "(the upstream host must be pinned)", retryable=False, kind="config")
        extra_body["provider"] = {"order": [host], "allow_fallbacks": False}
        extra_body["usage"] = {"include": True}
    extra_body.update(extra_params.get("extra_body", {}))
    kwargs.update({k: v for k, v in extra_params.items() if k != "extra_body"})
    if extra_body:
        kwargs["extra_body"] = extra_body
    params_sent = {k: v for k, v in kwargs.items() if k not in ("messages", "tools")}

    content, reasoning, reasoning_details = [], [], []
    calls: dict[int, dict] = {}
    first = None
    finish = None
    usage_obj = None
    model_returned = served = None
    reported_cost = None
    t0_holder.append(time.monotonic())
    try:
        for chunk in client(provider).chat.completions.create(**kwargs):
            model_returned = chunk.model or model_returned
            extra = chunk.model_extra or {}
            served = extra.get("provider") or served
            if chunk.usage is not None:
                usage_obj = chunk.usage
                reported_cost = (chunk.usage.model_extra or {}).get("cost", reported_cost)
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            d = ch.delta
            dx = d.model_extra or {}
            rc = dx.get("reasoning_content") or dx.get("reasoning")
            if first is None and (d.content or rc or d.tool_calls):
                first = time.monotonic()
            if d.content:
                content.append(d.content)
            if rc:
                reasoning.append(rc)
            if dx.get("reasoning_details"):
                reasoning_details.extend(dx["reasoning_details"])
            for tc in d.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
            if ch.finish_reason:
                finish = ch.finish_reason
    except AttemptError:
        raise
    except Exception as exc:        # noqa: BLE001
        retryable, code = _classify(exc)
        raise AttemptError(f"{type(exc).__name__}: {exc}", retryable, code, _usage(usage_obj)) from exc

    tool_calls = []
    raw_calls = []
    for idx in sorted(calls):
        c = calls[idx]
        args, ok = parse_arguments(c["arguments"])
        tool_calls.append({"id": c["id"], "name": c["name"], "arguments": args, "valid": ok,
                           "arguments_raw": c["arguments"]})
        raw_calls.append({"id": c["id"], "type": "function",
                          "function": {"name": c["name"], "arguments": c["arguments"]}})
    raw = {"role": "assistant", "content": "".join(content) or None}
    if raw_calls:
        raw["tool_calls"] = raw_calls
    if reasoning and provider == "deepseek":
        raw["reasoning_content"] = "".join(reasoning)
    if reasoning_details:
        raw["reasoning_details"] = reasoning_details
    stop = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
            "content_filter": "refusal"}.get(finish or "", finish or "unknown")
    if tool_calls and stop == "end_turn":
        stop = "tool_use"
    return Outcome(
        text="".join(content), tool_calls=tool_calls, raw_assistant=raw, stop_reason=stop,
        usage=_usage(usage_obj), first_token_t=first, model_returned=model_returned,
        params_sent=params_sent, provider_served=served, provider_reported_cost=reported_cost,
        raw_response={"message": raw, "reasoning": "".join(reasoning) or None, "finish_reason": finish,
                      "usage": usage_obj.model_dump(mode="json") if usage_obj else None})
