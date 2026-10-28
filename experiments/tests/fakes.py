"""Offline fake LLM provider for end-to-end tests of the experiment scripts.

Responses are scripted by each test (synthetic data). `install()` redirects runs/ and the
response cache to a pytest tmp directory, so nothing reaches runs/ or paper/.

    adapter = fakes.install(monkeypatch, tmp_path, responder, [fakes.fake_model("fake-good")])

`responder(model_key, system, messages, tools)` returns a string (the reply text) or a dict
{"text": str, "tool_calls": [{"name": str, "arguments": dict | None, "valid": bool}],
 "stop_reason": str, "error": "message" (raise a non-retryable API error)}.
"""
from __future__ import annotations

import json
import threading
import time

from harness import cache, config, llm
from harness.providers.base import AttemptError, Outcome, empty_usage


def fake_model(key: str, family: str = "fake", roles=("e1", "e2", "e3_gen", "e3_judge"), access: str = "closed",
               vision: bool = True, tier: str = "core", price=(1.0, 2.0)) -> dict:
    return {"key": key, "display": key.replace("-", " ").title(), "family": family, "access": access, "tier": tier,
            "provider": "fake", "provider_model_id": key, "vision": vision, "context_tokens": 200_000,
            "architecture": "fake", "test_time_compute": "--",
            "price": {"input": price[0], "output": price[1], "cache_read": price[0] / 10}, "roles": list(roles)}


def text_of(message: dict) -> str:
    """All text in an internal message (user parts, assistant text, tool results)."""
    if message["role"] == "tool":
        return "\n".join(r["content"] for r in message["results"])
    return "".join(p.get("text", "") for p in message.get("content", []) if isinstance(p, dict))


class FakeAdapter:
    def __init__(self, responder):
        self.responder = responder
        self.lock = threading.Lock()
        self.calls: list[dict] = []

    def call(self, model_cfg, system, messages, tools, max_tokens, extra, cache_mode, t0_holder) -> Outcome:
        t0_holder.append(time.monotonic())
        for m in messages:                          # the harness must never replay another provider's turn
            if m["role"] == "assistant":
                assert m.get("provider") in ("fake", "neutral"), m.get("provider")
        reply = self.responder(model_cfg["key"], system, messages, tools)
        if isinstance(reply, str):
            reply = {"text": reply}
        if reply.get("error"):
            raise AttemptError(reply["error"], retryable=False, status_code=400)
        text = reply.get("text", "")
        calls = [{"id": c.get("id") or f"call_{len(messages)}_{i}", "name": c["name"],
                  "arguments": c.get("arguments"), "valid": c.get("valid", c.get("arguments") is not None)}
                 for i, c in enumerate(reply.get("tool_calls", []))]
        n_in = (len(system or "") + sum(len(json.dumps(m, default=str)) for m in messages)) // 4
        usage = {**empty_usage(), "input_tokens": n_in, "output_tokens": max(1, len(text) // 4 + 20 * len(calls))}
        with self.lock:
            self.calls.append({"model": model_cfg["key"], "n_messages": len(messages), "tools": bool(tools),
                               "last_user": text_of(messages[-1])[:200] if messages else ""})
        return Outcome(text=text, tool_calls=calls, raw_assistant={"text": text, "tool_calls": calls},
                       stop_reason=reply.get("stop_reason") or ("tool_use" if calls else "end_turn"), usage=usage,
                       first_token_t=time.monotonic(), model_returned=model_cfg["key"] + "-20261028",
                       params_sent={"max_tokens": max_tokens, **extra}, raw_response={"text": text, "tool_calls": calls})


def install(monkeypatch, tmp_path, responder, models: list[dict]) -> FakeAdapter:
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "RUNS", runs)
    monkeypatch.setattr(cache, "CACHE_DIR", runs / "cache")
    original = config.models_config()
    patched = {**original, "models": list(original["models"]) + list(models)}
    monkeypatch.setattr(config, "models_config", lambda: patched)
    adapter = FakeAdapter(responder)
    monkeypatch.setitem(llm.ADAPTERS, "fake", adapter)
    monkeypatch.setitem(llm._SEMAPHORES, "fake", threading.BoundedSemaphore(8))
    return adapter
