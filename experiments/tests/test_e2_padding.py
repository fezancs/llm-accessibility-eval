"""E2 padding transcript helpers (harness/e2_padding.py). Generation itself is exercised with
the fake provider in test_e2_end_to_end.py."""
import json

import pytest

from harness import e2_padding


def test_parse_prompt():
    system, turns = e2_padding.parse_prompt()
    assert system.startswith("You are a senior Python developer.") and system.endswith("short summary.")
    assert len(turns) == 30
    assert turns[0].startswith("Create csvtool.py") and turns[29] == "Review the whole package for bugs and fix any you find."
    assert not any("accessib" in t.lower() or "html" in t.lower() for t in turns)
    with pytest.raises(ValueError):
        e2_padding.parse_prompt("no structure here")


def test_to_neutral_renames_ids_and_drops_raw():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "t1"}]},
            {"role": "assistant", "provider": "anthropic", "raw": [{"secret": 1}],
             "content": [{"type": "text", "text": "Working."}],
             "tool_calls": [{"id": "toolu_A", "name": "list_files", "arguments": {}, "valid": True},
                            {"id": "toolu_B", "name": "read_file", "arguments": None, "valid": False}]},
            {"role": "tool", "results": [{"tool_call_id": "toolu_A", "name": "list_files", "content": "x", "is_error": False},
                                         {"tool_call_id": "toolu_B", "name": "read_file", "content": "Error", "is_error": True}]},
            {"role": "assistant", "provider": "anthropic", "raw": [], "content": [{"type": "text", "text": "Done."}],
             "tool_calls": []}]
    out = e2_padding.to_neutral(msgs)
    a = out[1]
    assert a == {"role": "assistant", "provider": "neutral", "content": [{"type": "text", "text": "Working."}],
                 "tool_calls": [{"id": "pad_t01_c0", "name": "list_files", "arguments": {}},
                                {"id": "pad_t01_c1", "name": "read_file", "arguments": {}}]}
    assert [r["tool_call_id"] for r in out[2]["results"]] == ["pad_t01_c0", "pad_t01_c1"]
    assert all("raw" not in m for m in out)
    assert out[3]["content"] == [{"type": "text", "text": "Done."}] and out[3]["tool_calls"] == []


def test_prefix_and_load(tmp_path, monkeypatch):
    data = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}]},
                         {"role": "assistant", "provider": "neutral", "content": [], "tool_calls": []},
                         {"role": "user", "content": [{"type": "text", "text": "b"}]},
                         {"role": "assistant", "provider": "neutral", "content": [], "tool_calls": []}],
            "turn_boundaries": [2, 4]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(e2_padding, "TRANSCRIPT", p)
    loaded = e2_padding.load()
    assert e2_padding.prefix(loaded, 0) == [] and len(e2_padding.prefix(loaded, 1)) == 2
    pre = e2_padding.prefix(loaded, 2)
    pre[0]["content"][0]["text"] = "changed"                       # prefixes are copies
    assert loaded["messages"][0]["content"][0]["text"] == "a"
    data["messages"][1]["provider"] = "anthropic"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        e2_padding.load()
    monkeypatch.setattr(e2_padding, "TRANSCRIPT", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        e2_padding.load()


def test_cli_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    p = tmp_path / "existing.json"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(e2_padding, "TRANSCRIPT", p)
    assert e2_padding.main([]) == 1
    assert "exists" in capsys.readouterr().err and p.read_text() == "{}"
