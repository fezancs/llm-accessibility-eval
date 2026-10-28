"""One streaming client for every provider (SPEC 2.1-2.4).

    llm = LLM("e1")                                   # logs to runs/e1/calls.jsonl, budget cap "e1"
    r = llm.call("gpt-6-luna", [user("...")], item_id="G01/nudged", sample_idx=0)

Every attempt (including failed ones) is one line in the call log and is billed against the
budget key. A cached success is returned without a request, so scripts are resumable.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import random
import re
import threading
import time
import uuid

from . import budget, cache, config
from .providers import anthropic_adapter, chat_adapter, gemini_adapter, openai_adapter
from .providers.base import AttemptError, Outcome
from .cache import _normalise

ADAPTERS = {"anthropic": anthropic_adapter, "openai": openai_adapter, "google": gemini_adapter,
            "deepseek": chat_adapter, "openrouter": chat_adapter}
CONCURRENCY = {"anthropic": 8, "openai": 8, "google": 8, "deepseek": 6, "openrouter": 4}
_SEMAPHORES = {p: threading.BoundedSemaphore(n) for p, n in CONCURRENCY.items()}
MAX_RETRIES = 5
DEFAULT_MAX_TOKENS = 64_000
CONTEXT_RE = re.compile(r"prompt is too long|context[_ ]length|maximum context|context window|"
                        r"exceeds the maximum number of tokens|too many tokens|input is too long", re.I)


class ModelUnavailable(RuntimeError):
    pass


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int | None, call_ids: list[str], cost_usd: float):
        super().__init__(message)
        self.status_code = status_code
        self.call_ids = call_ids
        self.cost_usd = cost_usd
        self.context_exhausted = bool(CONTEXT_RE.search(message))


@dataclasses.dataclass
class LLMResult:
    text: str
    tool_calls: list[dict]
    assistant: dict                  # internal assistant message to append to the history
    stop_reason: str
    status: str                      # ok | refusal
    usage: dict
    cost_usd: float                  # all attempts of this request
    ttft_s: float | None
    total_s: float                   # successful attempt, request send to final token
    wall_s: float                    # all attempts including backoff
    model_returned: str | None
    provider_served: str | None
    call_ids: list[str]
    request_key: str
    cached: bool = False


def user(*parts) -> dict:
    """Build a user message from strings and part dicts."""
    return {"role": "user", "content": [{"type": "text", "text": p} if isinstance(p, str) else p
                                        for p in parts]}


def tool_message(results: list[dict]) -> dict:
    return {"role": "tool", "results": results}


def _check_available(m: dict) -> None:
    pid = str(m.get("provider_model_id", ""))
    if m.get("available") is False:
        raise ModelUnavailable(f"{m['key']} is marked unavailable in models.yaml")
    if m["provider"] != "openrouter" and (not pid or pid.upper().startswith("VERIFY")):
        raise ModelUnavailable(f"{m['key']}: provider_model_id not verified in models.yaml")
    if m["provider"] == "openrouter" and str(m.get("openrouter_id", "")).upper().startswith("VERIFY"):
        raise ModelUnavailable(f"{m['key']}: openrouter_id not verified in models.yaml")
    # Refuse BEFORE spending: a call whose cost cannot be computed must never be made.
    for card in ("price", "price_over_100k", "price_offpeak"):
        if card in m and (not m[card] or m[card].get("input") is None or m[card].get("output") is None):
            raise budget.PriceMissing(f"{m['key']}: {card} missing in models.yaml; fill it in before calling")


class LLM:
    def __init__(self, experiment: str, budget_key: str | None = None):
        self.experiment = experiment
        self.budget_key = budget_key or experiment
        self.dir = config.RUNS / experiment
        self.log_path = self.dir / "calls.jsonl"
        self.tracker = budget.BudgetTracker(self.budget_key, sorted(config.RUNS.glob("*/calls.jsonl"))
                                            + [self.log_path])

    # ------------------------------------------------------------------
    def call(self, model_key: str, messages: list[dict], system: str | None = None,
             tools: list[dict] | None = None, *, item_id: str, sample_idx: int = 0,
             purpose: str = "eval", max_tokens: int | None = None, cache_mode: str = "none",
             log_request: bool = True) -> LLMResult:
        m = config.model(model_key)
        _check_available(m)
        extra = dict(m.get("judge_params") or {}) if purpose == "judge" else dict(m.get("eval_params") or {})
        max_tokens = max_tokens or min(DEFAULT_MAX_TOKENS, m.get("max_output_tokens") or DEFAULT_MAX_TOKENS)
        key_params = {"provider": m["provider"], "model_id": m.get("provider_model_id"),
                      "openrouter_id": m.get("openrouter_id") if m["provider"] == "openrouter" else None,
                      "max_tokens": max_tokens, "extra": extra, "cache_mode": cache_mode}
        key = cache.request_key(model_key, system, messages, tools, key_params, sample_idx)
        hit = cache.get(key)
        if hit is not None:
            hit["cached"] = True
            return LLMResult(**hit)

        adapter = ADAPTERS[m["provider"]]
        call_ids: list[str] = []
        spent = 0.0
        wall0 = time.monotonic()
        for attempt in range(MAX_RETRIES + 1):
            self.tracker.check()
            call_id = uuid.uuid4().hex
            call_ids.append(call_id)
            started = dt.datetime.now(dt.timezone.utc)
            t0_holder: list[float] = []
            err: AttemptError | None = None
            out: Outcome | None = None
            with _SEMAPHORES[m["provider"]]:
                try:
                    out = adapter.call(m, system, messages, tools, max_tokens, extra, cache_mode, t0_holder)
                except AttemptError as e:
                    err = e
            t_end = time.monotonic()
            t0 = t0_holder[0] if t0_holder else t_end
            usage = out.usage if out else err.usage
            cost, card = budget.cost_usd(m, usage, started)
            spent += cost
            rec = {
                "call_id": call_id, "experiment": self.experiment, "budget_key": self.budget_key,
                "item_id": item_id, "model_key": model_key, "provider": m["provider"],
                "model_returned": out.model_returned if out else None,
                "provider_served": out.provider_served if out else None,
                "request_hash": key, "sample_idx": sample_idx, "attempt": attempt, "purpose": purpose,
                "started_at": started.isoformat(),
                "ttft_s": (out.first_token_t - t0) if out and out.first_token_t else None,
                "total_s": t_end - t0,
                **{k: usage.get(k) for k in ("input_tokens", "cached_input_tokens", "cache_write_tokens",
                                             "output_tokens", "reasoning_tokens")},
                "cost_usd": cost, "price_card": card,
                "provider_reported_cost": out.provider_reported_cost if out else None,
                "price_snapshot_date": config.models_config().get("snapshot_date"),
                "params": out.params_sent if out else {"max_tokens": max_tokens, **extra},
                "stop_reason": out.stop_reason if out else None,
                "status": ("refusal" if out.stop_reason == "refusal" else "ok") if out else "error",
                "error": str(err) if err else None, "response_path": None,
            }
            if out is not None:
                rp = self.dir / "raw" / model_key / f"{call_id}.json"
                rp.parent.mkdir(parents=True, exist_ok=True)
                req = ({"system": system, "messages": _normalise(messages),
                        "tools": [t["name"] for t in tools or []]} if log_request else {"omitted": True})
                rp.write_text(json.dumps({"request": req, "response": out.raw_response},
                                         ensure_ascii=False), encoding="utf-8")
                rec["response_path"] = str(rp.relative_to(config.RUNS))
            budget.append_jsonl(self.log_path, rec)

            if out is not None:
                result = LLMResult(
                    text=out.text, tool_calls=out.tool_calls,
                    assistant={"role": "assistant", "provider": m["provider"],
                               "content": [{"type": "text", "text": out.text}] if out.text else [],
                               "tool_calls": out.tool_calls, "raw": out.raw_assistant},
                    stop_reason=out.stop_reason, status=rec["status"], usage=usage, cost_usd=spent,
                    ttft_s=rec["ttft_s"], total_s=rec["total_s"], wall_s=time.monotonic() - wall0,
                    model_returned=out.model_returned, provider_served=out.provider_served,
                    call_ids=call_ids, request_key=key)
                cache.put(key, dataclasses.asdict(result))
                return result
            if not err.retryable or attempt == MAX_RETRIES:
                raise LLMError(str(err), err.status_code, call_ids, spent)
            time.sleep(min(60.0, 2.0 * 2 ** attempt) + random.uniform(0, 1))
        raise AssertionError("unreachable")

