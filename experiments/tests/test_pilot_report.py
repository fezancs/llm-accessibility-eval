"""Pilot report on synthetic pilot logs (tmp runs/ only)."""
import json

from harness import pilot_report
from tests import fakes


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_pilot_report(monkeypatch, tmp_path):
    fakes.install(monkeypatch, tmp_path, lambda *a: "", [])
    runs = tmp_path / "runs"
    monkeypatch.setattr(pilot_report, "PILOT", runs / "pilot")
    monkeypatch.setattr(pilot_report, "OUT", runs / "pilot_cost_projection.md")
    base = {"provider": "anthropic", "status": "ok", "ttft_s": 0.5, "total_s": 3.0, "sample_idx": 0,
            "input_tokens": 1000, "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 2000}
    calls = [{**base, "model_key": "claude-haiku-5-5", "item_id": f"gen/G0{i}/nudged", "cost_usd": 0.002} for i in (1, 2, 3)]
    calls += [{**base, "model_key": "claude-haiku-5-5", "item_id": "repair/B01-V1", "cost_usd": 0.01, "input_tokens": 9000},
              {**base, "model_key": "claude-haiku-5-5", "item_id": "audit/B01-V1", "cost_usd": 0.004},
              {**base, "model_key": "claude-haiku-5-5", "item_id": "audit-retry/B01-V1", "cost_usd": 0.001,
               "status": "error", "total_s": 1.0}]
    for pos in range(1, 7):
        for k in range(2):
            calls.append({**base, "model_key": "claude-haiku-5-5", "cost_usd": 0.001 * pos,
                          "item_id": f"e2/claude-haiku-5-5/run0/pos{pos:02d}/F01/session/call{k}"})
    _write(runs / "pilot" / "calls.jsonl", calls)
    _write(runs / "pilot" / "scores" / "gen.jsonl", [
        {"model_key": "claude-haiku-5-5", "html_found": True, "selfassess": {"parse_ok": True}},
        {"model_key": "claude-haiku-5-5", "html_found": False, "selfassess": {"parse_ok": False}}])
    _write(runs / "pilot" / "scores" / "steps.jsonl", [
        {"model_key": "claude-haiku-5-5", "arm": "session", "position": p, "cost_usd": 0.002 * p, "tool_calls": 5,
         "malformed_tool_calls": 2 if p == 1 else 0} for p in range(1, 7)])
    pilot_report.main()
    text = (runs / "pilot_cost_projection.md").read_text()
    assert "1/2" in text                                  # one generation without an html block
    assert "| claude-haiku-5-5 | 30 | 2 | 6.7% | ok |" in text
    assert "## Projection: full profile" in text and "## Projection: lean profile" in text
    assert "| e1_repair | claude-haiku-5-5 | 1 | 0.0100 |" in text
    assert "token-projected (same vendor)" in text       # Opus/Sonnet projected from Haiku tokens
