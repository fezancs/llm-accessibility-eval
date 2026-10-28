"""E2 controls (harness/e2_controls.py): fresh control and padded-length matching, with the
offline fake provider and a synthetic padding transcript."""
import json

from harness import e2_agent, e2_controls, e2_padding, e2_steps, llm
from tests import fakes


def synthetic_padding(n_turns=12):
    msgs, bounds = [], []
    for t in range(1, n_turns + 1):
        msgs.append(llm.user(f"Padding request number {t}: add a feature to the CSV tool."))
        msgs.append({"role": "assistant", "provider": "neutral", "content": [],
                     "tool_calls": [{"id": f"pad_t{t:02d}_c0", "name": "write_file",
                                     "arguments": {"path": f"m{t}.py", "content": "x = 1\n" * (40 + 7 * t)}}]})
        msgs.append({"role": "tool", "results": [{"tool_call_id": f"pad_t{t:02d}_c0", "name": "write_file",
                                                  "content": "Wrote 240 bytes.", "is_error": False}]})
        msgs.append({"role": "assistant", "provider": "neutral", "content": [{"type": "text", "text": f"Done with {t}."}],
                     "tool_calls": []})
        bounds.append(len(msgs))
    return {"model_key": "fake", "system": "pad", "messages": msgs, "turn_boundaries": bounds, "n_turns": n_turns}


RATIO = 0.3


def install(monkeypatch, tmp_path, responder):
    """Fake provider whose input-token count is exactly RATIO x request_chars (so the
    length-matching arithmetic can be checked exactly)."""
    adapter = fakes.install(monkeypatch, tmp_path, responder, [fakes.fake_model("fake-a")])
    orig = adapter.call

    def call(model_cfg, system, messages, tools, *a):
        out = orig(model_cfg, system, messages, tools, *a)
        out.usage["input_tokens"] = fake_tokens(system, messages, tools)
        return out
    adapter.call = call
    return adapter


def fake_tokens(system, messages, tools=e2_agent.TOOLS):
    return int(RATIO * e2_agent.request_chars(system, tools, messages))


def done_responder(model, system, messages, tools):
    return "Done."


def test_choose_prefix():
    chars = [100, 300, 600, 1000, 1500]
    assert e2_controls.choose_prefix(chars, 0.25, 150) == 2       # 0.25 * 600 = 150
    assert e2_controls.choose_prefix(chars, 0.25, 10) == 0
    assert e2_controls.choose_prefix(chars, 0.25, 10_000) == 4     # longest available
    assert e2_controls.choose_prefix([100, 300], 1.0, 200) == 0   # tie -> fewer turns


def test_prefix_chars_match_request_chars():
    pad = synthetic_padding()
    msg = e2_steps.step("R07")["message"]
    chars = e2_controls.prefix_chars(pad, msg)
    assert len(chars) == pad["n_turns"] + 1 and chars == sorted(chars)
    for n in (0, 1, 5, 12):
        msgs = e2_padding.prefix(pad, n) + [e2_controls.bridge_message(msg)]
        assert chars[n] == e2_agent.request_chars(e2_agent.SYSTEM, e2_agent.TOOLS, msgs)
    first = e2_controls.bridge_message(msg)
    assert fakes.text_of(first) == "That task is finished. New task for the CityCare project:\n\n" + msg
    assert e2_padding.prefix(pad, 3)[-1]["role"] == "assistant" and len(e2_padding.prefix(pad, 3)) == 12


def test_padded_rebuilds_with_observed_ratio(monkeypatch, tmp_path):
    adapter = install(monkeypatch, tmp_path, done_responder)
    pad = synthetic_padding()
    sid = "R07"
    msg = e2_steps.step(sid)["message"]
    target_n = 5
    target = fake_tokens(e2_agent.SYSTEM, e2_padding.prefix(pad, target_n) + [e2_controls.bridge_message(msg)])
    chars = e2_controls.prefix_chars(pad, msg)
    # A misleading session ratio (half the true tokens/char) makes the first pick too long.
    session_chars = int(2 * target / RATIO)
    prev = e2_steps.INITIAL_PROJECT
    row = e2_controls.run_padded(llm.LLM("e2"), "e2", "fake-a", 0, 101, 6, sid, prev, target, session_chars, pad)
    att = row["padded_attempts"]
    assert len(att) == 2 and att[0]["prefix_turns"] > target_n
    assert abs(att[0]["match_error"]) > 0.10 and not att[0]["kept"] and att[1]["kept"]
    assert att[1]["prefix_turns"] == row["prefix_turns"] == target_n and abs(row["length_match_error"]) <= 0.01
    assert row["length_matched"] and row["arm"] == "padded" and row["Lambda"] == att[1]["realised"]
    assert row["Lambda_target"] == target and row["status"] == "ok"
    assert [a["request_chars"] for a in att] == [chars[a["prefix_turns"]] for a in att]
    assert att[1]["rho"] == att[0]["realised"] / chars[att[0]["prefix_turns"]]      # observed ratio
    assert row["cost_usd"] == att[0]["cost_usd"] + att[1]["cost_usd"]
    assert len(adapter.calls) == 2
    root = tmp_path / "runs" / "e2" / "workspaces" / "fake-a" / "run0"
    assert sorted(p.name for p in root.iterdir()) == ["pos06_R07_padded"]          # try folders removed
    logged = [json.loads(line) for line in (tmp_path / "runs" / "e2" / "calls.jsonl").read_text().splitlines()]
    assert [r["item_id"] for r in logged] == ["e2/fake-a/run0/pos06/R07/padded/call0",
                                              "e2/fake-a/run0/pos06/R07/padded-retry/call0"]
    assert {r["sample_idx"] for r in logged} == {200}
    # The request sent: system, prefix of complete turns, then ONE bridge user message.
    row2 = e2_controls.run_padded(llm.LLM("e2"), "e2", "fake-a", 0, 101, 6, sid, prev, target, session_chars, pad)
    assert len(adapter.calls) == 2 and row2["padded_attempts"] == row["padded_attempts"]   # replay is free


def test_padded_within_tolerance_runs_once(monkeypatch, tmp_path):
    adapter = install(monkeypatch, tmp_path, done_responder)
    pad = synthetic_padding()
    msg = e2_steps.step("F02")["message"]
    msgs = e2_padding.prefix(pad, 7) + [e2_controls.bridge_message(msg)]
    target = fake_tokens(e2_agent.SYSTEM, msgs)
    row = e2_controls.run_padded(llm.LLM("e2"), "e2", "fake-a", 2, 303, 2, "F02", e2_steps.INITIAL_PROJECT,
                                 target, int(target / RATIO), pad)
    assert len(adapter.calls) == 1 and len(row["padded_attempts"]) == 1
    assert row["prefix_turns"] == 7 and abs(row["length_match_error"]) < 0.01 and row["length_matched"]


def test_padded_same_prefix_not_rerun(monkeypatch, tmp_path):
    adapter = install(monkeypatch, tmp_path, done_responder)
    pad = synthetic_padding(3)
    msg = e2_steps.step("F02")["message"]
    huge = 10 * fake_tokens(e2_agent.SYSTEM, e2_padding.prefix(pad, 3) + [e2_controls.bridge_message(msg)])
    row = e2_controls.run_padded(llm.LLM("e2"), "e2", "fake-a", 0, 101, 2, "F02", e2_steps.INITIAL_PROJECT,
                                 huge, int(huge / RATIO), pad)     # transcript too short for the target
    assert len(adapter.calls) == 1 and row["prefix_turns"] == 3
    assert not row["length_matched"] and row["length_match_error"] < -0.10
    assert "not re-run" in row["padded_attempts"][0]["rebuild"]


def test_fresh_control(monkeypatch, tmp_path):
    def responder(model, system, messages, tools):
        assert len([m for m in messages if m["role"] == "user"]) == 1        # no history
        if messages[-1]["role"] == "user":
            return {"text": "", "tool_calls": [{"name": "write_file", "arguments": {"path": "new.html", "content": "x"}}]}
        return "Done."
    adapter = fakes.install(monkeypatch, tmp_path, responder, [fakes.fake_model("fake-a")])
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / "index.html").write_text("<p>prev</p>", encoding="utf-8")
    row = e2_controls.run_fresh(llm.LLM("e2"), "e2", "fake-a", 1, 202, 3, "F03", prev)
    dest = tmp_path / "runs" / "e2" / "workspaces" / "fake-a" / "run1" / "pos03_F03_fresh"
    assert sorted(p.name for p in dest.iterdir()) == ["index.html", "new.html"]
    assert (prev / "index.html").exists() and not (prev / "new.html").exists()   # S_{t-1} untouched
    assert row["arm"] == "fresh" and row["tool_calls"] == 1 and row["status"] == "ok" and row["Lambda"] > 0
    assert row["snapshot"] == "runs/e2/workspaces/fake-a/run1/pos03_F03_fresh"
    assert len(adapter.calls) == 2
    logged = [json.loads(line) for line in (tmp_path / "runs" / "e2" / "calls.jsonl").read_text().splitlines()]
    assert logged[0]["item_id"] == "e2/fake-a/run1/pos03/F03/fresh/call0" and logged[0]["sample_idx"] == 101
