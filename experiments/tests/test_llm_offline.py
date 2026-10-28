"""Offline checks of the LLM client: message conversion, cache keys and request gating.
No network calls are made here."""
import pytest

from harness import cache, llm
from harness.providers import anthropic_adapter, chat_adapter, gemini_adapter, openai_adapter
from harness.providers.base import parse_arguments


def _history(provider, raw):
    return [llm.user("hi"),
            {"role": "assistant", "provider": provider, "content": [], "raw": raw},
            llm.tool_message([{"tool_call_id": "c1", "name": "list_files", "content": "a.html 10"}])]


def test_anthropic_conversion_replays_raw_and_tool_results():
    raw = [{"type": "thinking", "thinking": "", "signature": "s"},
           {"type": "tool_use", "id": "c1", "name": "list_files", "input": {}}]
    msgs = anthropic_adapter.to_messages(_history("anthropic", raw))
    assert msgs[1]["content"] == raw                       # thinking block passed back unchanged
    assert msgs[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "a.html 10"}]}


def test_cross_provider_replay_is_rejected():
    with pytest.raises(ValueError):
        anthropic_adapter.to_messages(_history("openai", []))


def test_openai_conversion():
    raw = [{"type": "function_call", "call_id": "c1", "name": "list_files", "arguments": "{}"}]
    items = openai_adapter.to_input(_history("openai", raw))
    assert items[0]["content"][0] == {"type": "input_text", "text": "hi"}
    assert items[1] == raw[0]
    assert items[2] == {"type": "function_call_output", "call_id": "c1", "output": "a.html 10"}


def test_chat_conversion_with_system():
    raw = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}]}
    msgs = chat_adapter.to_messages("SYS", _history("openrouter", raw), "openrouter")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "a.html 10"}


def test_gemini_conversion_keeps_signature():
    raw = [{"function_call": {"id": "c1", "name": "list_files", "args": {}}, "thought_signature": "AAFzaWf_"}]
    contents = gemini_adapter.to_contents(_history("google", raw))
    assert contents[1].parts[0].thought_signature == b"\x00\x01sig\xff"
    assert contents[2].parts[0].function_response.name == "list_files"


def test_cache_key_depends_on_sample_and_hashes_images(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8abc")
    msgs = [llm.user({"type": "image", "path": str(img), "media_type": "image/jpeg"}, "describe")]
    k0 = cache.request_key("m", None, msgs, None, {"a": 1}, 0)
    assert k0 == cache.request_key("m", None, msgs, None, {"a": 1}, 0)
    assert k0 != cache.request_key("m", None, msgs, None, {"a": 1}, 1)
    img.write_bytes(b"\xff\xd8abd")                        # same path, different bytes
    assert k0 != cache.request_key("m", None, msgs, None, {"a": 1}, 0)


def test_parse_arguments():
    assert parse_arguments('{"path": "a"}') == ({"path": "a"}, True)
    assert parse_arguments('{"path": ') == (None, False)
    assert parse_arguments("[1]") == (None, False)
    assert parse_arguments("") == ({}, True)


def test_unverified_model_is_refused(monkeypatch, tmp_path):
    from tests import fakes
    m = fakes.fake_model("fake-unverified")
    m.update(provider="google", provider_model_id="VERIFY")
    fakes.install(monkeypatch, tmp_path, lambda *a: "x", [m])
    with pytest.raises(llm.ModelUnavailable):
        llm.LLM("pilot").call("fake-unverified", [llm.user("x")], item_id="t")


def test_context_error_detection():
    e = llm.LLMError("BadRequestError: prompt is too long: 1200000 tokens > 1000000 maximum", 400, [], 0.0)
    assert e.context_exhausted
    assert not llm.LLMError("rate limited", 429, [], 0.0).context_exhausted


NEUTRAL = {"role": "assistant", "provider": "neutral", "content": [{"type": "text", "text": "Reading."}],
           "tool_calls": [{"id": "toolu_1", "name": "read_file", "arguments": {"path": "a.py"}}]}


def test_neutral_turn_converts_for_every_provider():
    hist = [llm.user("hi"), NEUTRAL,
            llm.tool_message([{"tool_call_id": "toolu_1", "name": "read_file", "content": "print(1)"}]),
            llm.user("next")]
    a = anthropic_adapter.to_messages(hist)
    assert a[1]["content"][1] == {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}
    o = openai_adapter.to_input(hist)
    assert o[1] == {"role": "assistant", "content": "Reading."}
    assert o[2]["type"] == "function_call" and o[2]["arguments"] == '{"path": "a.py"}'
    c = chat_adapter.to_messages(None, hist, "deepseek")
    assert c[1]["tool_calls"][0]["function"]["name"] == "read_file"
    g = gemini_adapter.to_contents(hist)
    assert g[1].parts[1].function_call.name == "read_file"
    assert g[1].parts[1].thought_signature == gemini_adapter.INJECTED_SIGNATURE


def test_fake_provider_end_to_end(monkeypatch, tmp_path):
    from tests import fakes
    adapter = fakes.install(monkeypatch, tmp_path, lambda m, s, msgs, t: "pong", [fakes.fake_model("fake-a")])
    client = llm.LLM("pilot")
    r = client.call("fake-a", [llm.user("ping")], item_id="t/1")
    assert r.text == "pong" and r.status == "ok" and not r.cached and r.cost_usd > 0
    r2 = client.call("fake-a", [llm.user("ping")], item_id="t/1")
    assert r2.cached and len(adapter.calls) == 1                 # cached success is never re-requested
    log = (tmp_path / "runs" / "pilot" / "calls.jsonl").read_text().splitlines()
    assert len(log) == 1 and '"status": "ok"' in log[0]


def test_missing_price_refused_before_any_request(monkeypatch, tmp_path):
    from harness import budget
    from tests import fakes
    m = fakes.fake_model("fake-noprice")
    m["price"] = {"input": None, "output": None}
    adapter = fakes.install(monkeypatch, tmp_path, lambda *a: "x", [m])
    with pytest.raises(budget.PriceMissing):
        llm.LLM("pilot").call("fake-noprice", [llm.user("x")], item_id="t")
    assert adapter.calls == []                                  # nothing was sent


def test_openai_usage_bills_cache_writes():
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage
    u = ResponseUsage(input_tokens=10_000, output_tokens=500, total_tokens=10_500,
                      input_tokens_details=InputTokensDetails(cached_tokens=6_000, cache_write_tokens=3_000),
                      output_tokens_details=OutputTokensDetails(reasoning_tokens=200))
    usage = openai_adapter._usage(u)
    assert (usage["input_tokens"], usage["cached_input_tokens"], usage["cache_write_tokens"]) == (1_000, 6_000, 3_000)
    assert usage["output_tokens"] == 500 and usage["reasoning_tokens"] == 200


def test_gemini_injected_signature_encodings():
    import base64
    hist = [llm.user("hi"), NEUTRAL,
            llm.tool_message([{"tool_call_id": "toolu_1", "name": "read_file", "content": "print(1)"}])]
    raw = gemini_adapter.to_contents(hist, "bytes")[1].parts[1].thought_signature
    assert raw == b"skip_thought_signature_validator"
    b64 = gemini_adapter.to_contents(hist, "base64")[1].parts[1].thought_signature
    assert base64.urlsafe_b64encode(b64).decode() == "skip_thought_signature_validator"
    with pytest.raises(ValueError):
        gemini_adapter.to_contents(hist, "hex")


def test_check_keys_host_names():
    from harness import check_keys
    assert check_keys.host_matches("deepinfra/bf16", {"DeepInfra"}) is True
    assert check_keys.host_matches("siliconflow/fp8", {"SiliconFlow"}) is True
    assert check_keys.host_matches("crusoe/bf16", {"Crusoe", "Novita"}) is False
    assert check_keys.host_matches("parasail/fp8", set()) is None


def test_check_keys_injected_history_converts_everywhere():
    from harness import check_keys
    hist = check_keys.injected_history()
    assert anthropic_adapter.to_messages(hist)[1]["content"][0]["type"] == "tool_use"
    assert openai_adapter.to_input(hist)[1]["type"] == "function_call"
    assert chat_adapter.to_messages(None, hist, "openrouter")[1]["tool_calls"][0]["id"] == "pad_t01_c0"
    assert gemini_adapter.to_contents(hist)[1].parts[0].function_call.name == "list_files"


def test_gemini_merges_consecutive_user_turns_and_drops_synthetic_ids():
    raw = [{"function_call": {"name": "list_files", "args": {}}, "thought_signature": "AAFzaWf_"}]   # no id from the API
    hist = [llm.user("step 1"),
            {"role": "assistant", "provider": "google", "content": [], "raw": raw},
            llm.tool_message([{"tool_call_id": gemini_adapter.SYNTHETIC_ID + "0", "name": "list_files",
                               "content": "a.html 10"}]),
            llm.user("step 2")]                                      # the step ended on tool results
    contents = gemini_adapter.to_contents(hist)
    assert [c.role for c in contents] == ["user", "model", "user"]
    last = contents[-1].parts
    assert last[0].function_response.id is None and last[0].function_response.name == "list_files"
    assert last[1].text == "step 2"
