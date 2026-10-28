"""Per-call cost from token counts and models.yaml prices, and spend caps (SPEC 2.2, budget.yaml).

Normalised usage (every provider adapter returns this shape):
    input_tokens         prompt tokens billed at the full input price (no cache read, no cache write)
    cached_input_tokens  prompt tokens billed at the cache-read price
    cache_write_tokens   prompt tokens billed at the cache-write price (Anthropic only)
    output_tokens        all billed output tokens, reasoning/thinking included
    reasoning_tokens     the reasoning share of output_tokens (informational, not billed twice)
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import pathlib
import threading

from . import config

HAIKU_TIER_THRESHOLD = 100_000   # prompts over 100k tokens use price_over_100k


class BudgetExceeded(RuntimeError):
    pass


class PriceMissing(RuntimeError):
    pass


def prompt_tokens(usage: dict) -> int:
    return int(usage.get("input_tokens", 0) + usage.get("cached_input_tokens", 0)
               + usage.get("cache_write_tokens", 0))


def is_deepseek_peak(when: dt.datetime, windows: list[list[int]], weekdays_only: bool = True) -> bool:
    """True if `when` (timezone-aware) falls in one of the [start, end) UTC hour windows."""
    t = when.astimezone(dt.timezone.utc)
    if weekdays_only and t.weekday() >= 5:
        return False
    return any(start <= t.hour < end for start, end in windows)


def price_table(m: dict, usage: dict, when: dt.datetime | None) -> tuple[dict, str]:
    """Pick the price card that applies to one request. Returns (prices, card_name)."""
    card, name = m.get("price") or {}, "standard"
    if "price_over_100k" in m and prompt_tokens(usage) > HAIKU_TIER_THRESHOLD:
        card, name = m["price_over_100k"], "over_100k"
    if "price_offpeak" in m:
        if when is None:
            raise ValueError(f"{m['key']} has peak/off-peak prices; the call time is required")
        windows = m.get("peak_windows_utc", [[1, 4], [6, 10]])
        if not is_deepseek_peak(when, windows, m.get("peak_weekdays_only", True)):
            card, name = m["price_offpeak"], "offpeak"
        else:
            name = "peak"
    if card.get("input") is None or card.get("output") is None:
        raise PriceMissing(f"no price for {m['key']} in models.yaml; fill it in before calling")
    return card, name


def cost_usd(model_key_or_cfg, usage: dict, when: dt.datetime | None = None) -> tuple[float, str]:
    """Billed USD for one request, and the name of the price card used."""
    m = config.model(model_key_or_cfg) if isinstance(model_key_or_cfg, str) else model_key_or_cfg
    card, name = price_table(m, usage, when)
    p_in = card["input"]
    p_read = card.get("cache_read", p_in)
    p_write = card.get("cache_write_5m", p_in)
    total = (usage.get("input_tokens", 0) * p_in
             + usage.get("cached_input_tokens", 0) * p_read
             + usage.get("cache_write_tokens", 0) * p_write
             + usage.get("output_tokens", 0) * card["output"]) / 1e6
    return total, name


class BudgetTracker:
    """Cumulative spend per budget key, read from the append-only call logs.

    Several processes may write the same log, so the tracker re-reads new lines on
    every check instead of trusting an in-memory counter.
    """

    def __init__(self, budget_key: str, log_paths: list[pathlib.Path]):
        caps = config.budget_config()["caps"]
        if budget_key not in caps:
            raise KeyError(f"no cap for budget key {budget_key!r} in budget.yaml")
        self.key = budget_key
        self.cap = float(caps[budget_key])
        self.paths = [pathlib.Path(p) for p in log_paths]
        self._offsets = {p: 0 for p in self.paths}
        self._spent = 0.0
        self._lock = threading.Lock()

    def spent(self) -> float:
        with self._lock:
            for p in self.paths:
                if not p.exists():
                    continue
                with open(p, encoding="utf-8") as f:
                    f.seek(self._offsets[p])
                    while True:
                        line = f.readline()
                        if not line or not line.endswith("\n"):
                            break                      # partial line: re-read next time
                        self._offsets[p] = f.tell()
                        rec = json.loads(line)
                        if rec.get("budget_key", self.key) == self.key:
                            self._spent += float(rec.get("cost_usd") or 0.0)
            return self._spent

    def check(self) -> None:
        s = self.spent()
        if s >= self.cap:
            raise BudgetExceeded(
                f"budget '{self.key}' reached: ${s:.2f} spent of ${self.cap:.2f} cap. "
                "Stop and ask the author how to proceed (CLAUDE.md hard rule 3).")


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    """Append one JSON line under an exclusive file lock (safe across processes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
