"""E2 agent loop (harness/e2_agent.py) with the offline fake provider: tool cap (including a
mid-response cut-off), context exhaustion, API errors and deterministic replay."""
import json

from harness import e2_agent, e2_steps, llm
from harness.e2_tools import LIMIT_MESSAGE, Workspace
from tests import fakes


def n_tool_msgs_since_user(messages):
    n = 0
    for m in reversed(messages):
        if m["role"] == "user":
            break
        n += m["role"] == "tool"
    return n


def write_call(i):
    return {"name": "write_file", "arguments": {"path": f"f{i}.txt", "content": str(i)}}


def test_tool_cap_mid_response(monkeypatch, tmp_path):
    def responder(model, system, messages, tools):
        k = n_tool_msgs_since_user(messages)
        return {"text": "", "tool_calls": [write_call(4 * k + j) for j in range(4)]}   # 4 calls per response, forever
    adapter = fakes.install(monkeypatch, tmp_path, responder, [fakes.fake_model("fake-a")])
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    msgs = [llm.user("go")]
    res = e2_agent.run_turn(llm.LLM("e2"), "fake-a", "sys", e2_agent.TOOLS, msgs, Workspace(ws_dir),
                            item_prefix="e2/t/run0/pos01/F01/session", sample_idx=0, cap=10)
    assert len(adapter.calls) == 3 and res["llm_calls"] == 3      # no call after the cap
    assert res["tool_calls"] == 10 and res["cap_reached"] and res["cut_off_calls"] == 2
    assert res["status"] == "ok" and res["malformed_tool_calls"] == 0
    assert sorted(p.name for p in ws_dir.iterdir()) == sorted(f"f{i}.txt" for i in range(10))
    last = msgs[-1]
    assert last["role"] == "tool" and len(last["results"]) == 4      # ONE tool message with all results
    assert [r["content"] for r in last["results"][2:]] == [LIMIT_MESSAGE, LIMIT_MESSAGE]
    assert all(r["is_error"] for r in last["results"][2:]) and not last["results"][0]["is_error"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant", "tool", "assistant", "tool"]
    assert res["Lambda"] > 0 and res["first_request_chars"] > 0 and res["latency_s"] >= 0
    logged = [json.loads(line) for line in (tmp_path / "runs" / "e2" / "calls.jsonl").read_text().splitlines()]
    assert [r["item_id"] for r in logged] == [f"e2/t/run0/pos01/F01/session/call{k}" for k in range(3)]
    assert all(r["response_path"] for r in logged)


def test_cap_exactly_at_response_end_and_malformed_count(monkeypatch, tmp_path):
    def responder(model, system, messages, tools):
        k = n_tool_msgs_since_user(messages)
        calls = [write_call(5 * k + j) for j in range(4)] + [{"name": "nope", "arguments": {}}]
        return {"text": "", "tool_calls": calls}
    adapter = fakes.install(monkeypatch, tmp_path, responder, [fakes.fake_model("fake-a")])
    (tmp_path / "ws").mkdir()
    msgs = [llm.user("go")]
    res = e2_agent.run_turn(llm.LLM("e2"), "fake-a", "sys", e2_agent.TOOLS, msgs, Workspace(tmp_path / "ws"),
                            item_prefix="x", sample_idx=0, cap=10)
    assert len(adapter.calls) == 2 and res["tool_calls"] == 10 and res["cut_off_calls"] == 0
    assert res["malformed_tool_calls"] == 2 and res["cap_reached"]


def test_stops_without_tool_calls(monkeypatch, tmp_path):
    def responder(model, system, messages, tools):
        if n_tool_msgs_since_user(messages) == 0:
            return {"text": "Looking.", "tool_calls": [{"name": "list_files", "arguments": {}},
                                                       {"name": "read_file", "arguments": None, "valid": False}]}
        return "Done."
    fakes.install(monkeypatch, tmp_path, responder, [fakes.fake_model("fake-a")])
    (tmp_path / "ws").mkdir()
    msgs = [llm.user("go")]
    res = e2_agent.run_turn(llm.LLM("e2"), "fake-a", "sys", e2_agent.TOOLS, msgs, Workspace(tmp_path / "ws"),
                            item_prefix="x", sample_idx=0, cap=10)
    assert res["llm_calls"] == 2 and res["tool_calls"] == 2 and res["malformed_tool_calls"] == 1
    assert res["final_text"] == "Done." and not res["cap_reached"]
    assert msgs[-1]["role"] == "assistant"


def _step_responder(fail_on: str, error: str):
    def responder(model, system, messages, tools):
        last_user = next(fakes.text_of(m) for m in reversed(messages) if m["role"] == "user")
        if fail_on in last_user:
            return {"error": error}
        if n_tool_msgs_since_user(messages) == 0:
            return {"text": "", "tool_calls": [{"name": "write_file",
                                                "arguments": {"path": "log.txt", "content": last_user[:20]}}]}
        return "Done."
    return responder


def test_context_exhaustion_ends_run(monkeypatch, tmp_path):
    order = e2_steps.session_order(101)
    fail_msg = e2_steps.step(order[2])["message"][:40]
    adapter = fakes.install(monkeypatch, tmp_path, _step_responder(fail_msg, "prompt is too long: 1200000 tokens > 1000000 maximum"),
                            [fakes.fake_model("fake-a")])
    s = e2_agent.Session(llm.LLM("e2"), "e2", "fake-a", 0, 101, order, positions=6)
    started = []
    summary = s.run(on_step_start=lambda pos, sid, prev: started.append((pos, sid, prev.name)))
    assert [r["status"] for r in s.rows] == ["ok", "ok", "context_exhausted"]
    assert summary["context_exhausted_at"] == 3 and summary["completed_steps"] == 2
    assert s.rows[2]["context_exhausted"] and s.rows[2]["Lambda"] is None
    assert started == [(1, "F01", "initial_project"), (2, "F02", "pos01_F01"), (3, "F03", "pos02_F02")]
    assert len(adapter.calls) == 2 + 2 + 0                          # the failing request is not retried
    root = tmp_path / "runs" / "e2" / "workspaces" / "fake-a" / "run0"
    assert (root / "pos03_F03").is_dir() and not (root / "pos04_F04").exists()
    tr = json.loads((tmp_path / "runs" / "e2" / "transcripts" / "fake-a" / "run0.json").read_text())
    assert [st["status"] for st in tr["steps"]] == ["ok", "ok", "context_exhausted"]
    assert s.rows[1]["cumulative_tool_calls"] == 2


def test_other_api_error_ends_run(monkeypatch, tmp_path):
    order = e2_steps.session_order(101)
    fail_msg = e2_steps.step(order[1])["message"][:40]
    fakes.install(monkeypatch, tmp_path, _step_responder(fail_msg, "invalid_request_error: bad tool schema"),
                  [fakes.fake_model("fake-a")])
    s = e2_agent.Session(llm.LLM("e2"), "e2", "fake-a", 0, 101, order, positions=6)
    summary = s.run()
    assert [r["status"] for r in s.rows] == ["ok", "error"]
    assert summary["context_exhausted_at"] is None and s.stopped == "error"
    assert "bad tool schema" in s.rows[1]["error"]


def test_session_replay_is_free(monkeypatch, tmp_path):
    adapter = fakes.install(monkeypatch, tmp_path, _step_responder("never-matches", ""), [fakes.fake_model("fake-a")])
    order = e2_steps.session_order(202)
    s1 = e2_agent.Session(llm.LLM("e2"), "e2", "fake-a", 1, 202, order, positions=4)
    s1.run()
    n = len(adapter.calls)
    assert n == 8
    s2 = e2_agent.Session(llm.LLM("e2"), "e2", "fake-a", 1, 202, order, positions=4)
    s2.run()
    assert len(adapter.calls) == n                                   # every request served from the cache
    strip = lambda rows: [{k: v for k, v in r.items() if k != "tool_time_s"} for r in rows]   # noqa: E731
    assert strip(s1.rows) == strip(s2.rows)
    assert all(r["Lambda"] for r in s2.rows)
