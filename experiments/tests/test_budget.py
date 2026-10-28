"""Cost computation against hand-computed examples (CLAUDE.md Phase 1, step 1)."""
import datetime as dt
import json

import pytest

from harness import budget

UTC = dt.timezone.utc
MON_0230 = dt.datetime(2026, 11, 2, 2, 30, tzinfo=UTC)     # Monday, inside 01-04 UTC
MON_0500 = dt.datetime(2026, 11, 2, 5, 0, tzinfo=UTC)      # Monday, between the two windows
MON_0930 = dt.datetime(2026, 11, 2, 9, 30, tzinfo=UTC)     # Monday, inside 06-10 UTC
SAT_0230 = dt.datetime(2026, 11, 7, 2, 30, tzinfo=UTC)     # Saturday


def u(i=0, c=0, w=0, o=0):
    return {"input_tokens": i, "cached_input_tokens": c, "cache_write_tokens": w, "output_tokens": o}


def test_weekday_fixture():
    assert MON_0230.weekday() == 0 and SAT_0230.weekday() == 5


def test_opus_all_token_kinds():
    # 1000*4 + 2000*0.20 + 500*5 + 3000*20 = 66,900 micro-dollars
    cost, card = budget.cost_usd("claude-opus-5-5", u(1000, 2000, 500, 3000))
    assert card == "standard"
    assert cost == pytest.approx(0.0669)


def test_haiku_standard_tier():
    cost, card = budget.cost_usd("claude-haiku-5-5", u(50_000, o=1000))
    assert card == "standard"
    assert cost == pytest.approx((50_000 * 0.10 + 1000 * 0.50) / 1e6)


def test_haiku_exactly_100k_is_standard():
    _, card = budget.cost_usd("claude-haiku-5-5", u(60_000, 40_000, 0, 10))
    assert card == "standard"


def test_haiku_over_100k_tier_counts_cached_tokens():
    # 90k uncached + 20k cached = 110k prompt -> whole request on the >100k card:
    # 90000*0.50 + 20000*0.05 + 2000*2.50 = 51,000 micro-dollars
    cost, card = budget.cost_usd("claude-haiku-5-5", u(90_000, 20_000, 0, 2000))
    assert card == "over_100k"
    assert cost == pytest.approx(0.051)


def test_deepseek_peak_and_offpeak():
    tokens = u(1000, 1000, 0, 1000)
    peak, card = budget.cost_usd("deepseek-v4-pro", tokens, MON_0230)
    assert card == "peak" and peak == pytest.approx((1000 * 1.32 + 1000 * 0.044 + 1000 * 3.96) / 1e6)
    assert budget.cost_usd("deepseek-v4-pro", tokens, MON_0930)[1] == "peak"
    off, card = budget.cost_usd("deepseek-v4-pro", tokens, MON_0500)
    assert card == "offpeak" and off == pytest.approx((1000 * 0.66 + 1000 * 0.022 + 1000 * 1.98) / 1e6)
    assert budget.cost_usd("deepseek-v4-pro", tokens, SAT_0230)[1] == "offpeak"


def test_deepseek_requires_call_time():
    with pytest.raises(ValueError):
        budget.cost_usd("deepseek-v4-pro", u(1, o=1))


def test_missing_price_refuses():
    with pytest.raises(budget.PriceMissing):
        budget.cost_usd("gemma-4-31b", u(1, o=1))


def test_openai_cached_tokens_use_cache_price():
    cost, _ = budget.cost_usd("gpt-6-sol", u(1000, 9000, 0, 100))
    assert cost == pytest.approx((1000 * 2.0 + 9000 * 0.2 + 100 * 10.0) / 1e6)


def test_tracker_stops_at_cap(tmp_path):
    log = tmp_path / "calls.jsonl"
    t = budget.BudgetTracker("pilot", [log])                 # pilot cap is $25
    for _ in range(4):
        budget.append_jsonl(log, {"budget_key": "pilot", "cost_usd": 6.0})
    budget.append_jsonl(log, {"budget_key": "e1", "cost_usd": 100.0})   # other key: ignored
    t.check()                                                # $24 < $25
    budget.append_jsonl(log, {"budget_key": "pilot", "cost_usd": 1.0})
    with pytest.raises(budget.BudgetExceeded):
        t.check()


def test_tracker_ignores_partial_last_line(tmp_path):
    log = tmp_path / "calls.jsonl"
    t = budget.BudgetTracker("pilot", [log])
    log.write_text(json.dumps({"budget_key": "pilot", "cost_usd": 1.0}) + "\n" + '{"budget_key": "pi')
    assert t.spent() == pytest.approx(1.0)
