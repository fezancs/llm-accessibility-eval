"""Google Gemini API adapter (google-genai, streaming).

Streamed parts are kept exactly as they arrive (not merged), so thought signatures stay
attached to the parts that carried them when the turn is replayed in a tool loop.
Billed output = candidates + thoughts tokens. Evaluated calls send no thinking_config, so
the model default applies; judges pass models.yaml `judge_params`.
"""
from __future__ import annotations

import os
import pathlib
import time

from google import genai
from google.genai import errors, types

from .base import AttemptError, Outcome, empty_usage, neutral_calls, neutral_text

_client = None
INJECTED_SIGNATURE = b"skip_thought_signature_validator"


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"),
                               http_options=types.HttpOptions(timeout=900_000))
    return _client


def to_contents(messages: list[dict]) -> list[types.Content]:
    out = []
    for m in messages:
        if m["role"] == "user":
            parts = []
            for p in m["content"]:
                if p["type"] == "text":
                    parts.append(types.Part(text=p["text"]))
                else:
                    parts.append(types.Part(inline_data=types.Blob(
                        mime_type=p.get("media_type", "image/jpeg"), data=pathlib.Path(p["path"]).read_bytes())))
            out.append(types.Content(role="user", parts=parts))
        elif m["role"] == "assistant" and m.get("provider") == "neutral":
            parts = [types.Part(text=neutral_text(m))] if neutral_text(m).strip() else []
            # Function calls Gemini did not generate carry no thought signature; Google documents
            # this placeholder for injected history. VERIFY in the pilot (CLAUDE.md Phase 2).
            parts += [types.Part(function_call=types.FunctionCall(id=c["id"], name=c["name"], args=c.get("arguments") or {}),
                                 thought_signature=INJECTED_SIGNATURE) for c in neutral_calls(m)]
            if parts:
                out.append(types.Content(role="model", parts=parts))
        elif m["role"] == "assistant":
            if m.get("provider") != "google":
                raise ValueError("assistant turn from another provider cannot be replayed to Gemini")
            out.append(types.Content.model_validate({"role": "model", "parts": m["raw"]}))
        elif m["role"] == "tool":
            out.append(types.Content(role="user", parts=[
                types.Part(function_response=types.FunctionResponse(
                    id=r["tool_call_id"], name=r["name"],
                    response={"error": r["content"]} if r.get("is_error") else {"output": r["content"]}))
                for r in m["results"]]))
    return out


def _usage(u) -> dict:
    usage = empty_usage()
    if u is None:
        return usage
    prompt = (u.prompt_token_count or 0) + (u.tool_use_prompt_token_count or 0)
    cached = u.cached_content_token_count or 0
    usage["input_tokens"] = prompt - cached
    usage["cached_input_tokens"] = cached
    usage["reasoning_tokens"] = u.thoughts_token_count or 0
    usage["output_tokens"] = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    return usage


def _classify(exc: Exception) -> tuple[bool, int | None]:
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        return (code in (408, 409, 429) or (code or 0) >= 500), code
    return True, None


def call(model_cfg: dict, system: str | None, messages: list[dict], tools: list[dict] | None,
         max_tokens: int, extra_params: dict, cache: str, t0_holder: list) -> Outcome:
    cfg = {"max_output_tokens": max_tokens,
           "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True)}
    if system:
        cfg["system_instruction"] = system
    if tools:
        cfg["tools"] = [types.Tool(function_declarations=[types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"])
            for t in tools])]
    if "thinking_config" in extra_params:
        cfg["thinking_config"] = types.ThinkingConfig(**extra_params["thinking_config"])
    params_sent = {"model": model_cfg["provider_model_id"], "max_output_tokens": max_tokens,
                   **{k: v for k, v in extra_params.items()}}

    parts, first, usage_obj, finish, model_returned = [], None, None, None, None
    block_reason = None
    t0_holder.append(time.monotonic())
    try:
        stream = client().models.generate_content_stream(
            model=model_cfg["provider_model_id"], contents=to_contents(messages),
            config=types.GenerateContentConfig(**cfg))
        for chunk in stream:
            model_returned = chunk.model_version or model_returned
            if chunk.usage_metadata is not None:
                usage_obj = chunk.usage_metadata
            if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                block_reason = str(chunk.prompt_feedback.block_reason)
            for cand in chunk.candidates or []:
                if cand.finish_reason:
                    finish = str(cand.finish_reason.value if hasattr(cand.finish_reason, "value")
                                 else cand.finish_reason)
                for p in (cand.content.parts if cand.content and cand.content.parts else []):
                    if first is None and (p.text or p.function_call):
                        first = time.monotonic()
                    parts.append(p)
                break                   # one candidate requested
    except Exception as exc:        # noqa: BLE001
        retryable, code = _classify(exc)
        raise AttemptError(f"{type(exc).__name__}: {exc}", retryable, code, _usage(usage_obj)) from exc

    text, tool_calls = [], []
    for i, p in enumerate(parts):
        if p.text and not p.thought:
            text.append(p.text)
        if p.function_call:
            fid = p.function_call.id or f"call_{i}"
            tool_calls.append({"id": fid, "name": p.function_call.name,
                               "arguments": dict(p.function_call.args or {}), "valid": True})
    if block_reason or finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
        stop = "refusal"
    elif finish == "MAX_TOKENS":
        stop = "max_tokens"
    elif finish == "MALFORMED_FUNCTION_CALL":
        stop = "malformed_tool_call"
    elif tool_calls:
        stop = "tool_use"
    else:
        stop = "end_turn"
    raw_parts = [p.model_dump(mode="json", exclude_none=True) for p in parts]
    return Outcome(
        text="".join(text), tool_calls=tool_calls, raw_assistant=raw_parts, stop_reason=stop,
        usage=_usage(usage_obj), first_token_t=first, model_returned=model_returned,
        params_sent=params_sent,
        raw_response={"parts": raw_parts, "finish_reason": finish, "block_reason": block_reason,
                      "usage": usage_obj.model_dump(mode="json", exclude_none=True) if usage_obj else None})

