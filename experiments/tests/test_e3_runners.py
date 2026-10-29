"""E3 generation and judging logic without any model call (fake clients only)."""
import collections
import itertools
import types

import pytest

from harness import config, e3_data, e3_generate as gen, e3_judge as jd, llm

FIX = config.DATA / "e3" / "test_fixtures"
GENS8 = ["claude-opus-5-5", "claude-haiku-5-5", "gpt-6-sol", "gpt-6-luna", "gemini-3-8-flash",
         "qwen3-8-27b", "gemma-4-31b", "llama-4-maverick"]


@pytest.fixture(scope="module")
def items(tmp_path_factory):
    img = tmp_path_factory.mktemp("img") / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake")
    rel = img                                             # absolute path; image_part joins with ROOT
    its = e3_data.parse_blv(FIX / "blv") + e3_data.parse_cr(FIX / "cr")
    for i in its:
        i.update(retrieved=True, image_path=str(rel), media_type="image/jpeg")
    return its


# ------------------------------------------------------------------ anonymisation

@pytest.mark.parametrize("raw, clean, fired", [
    ('"A dog runs."', "A dog runs.", ["quotes"]),
    ('Alt text: "a dog on a beach"', "A dog on a beach", ["alt_text_prefix", "quotes"]),
    ("alt-text: two cats", "Two cats", ["alt_text_prefix"]),
    ("Image of a red barn.", "A red barn.", ["image_of"]),
    ("An image of a red barn", "A red barn", ["image_of"]),
    ("Picture of two cats", "Two cats", ["picture_of"]),
    ("**Bold** text with `code`", "Bold text with code", ["markdown"]),
    ("[Map](http://x) of the region", "Map of the region", ["markdown"]),
    ('"The dog\'s bone"', "The dog's bone", ["quotes"]),
    ('"A" and "B"', '"A" and "B"', []),
    ("  multiple   spaces\nnewline ", "Multiple spaces newline", []),
    ("imagery of mountains", "Imagery of mountains", []),
    ("5 * 3 apples", "5 * 3 apples", []),
    ("“Curly quoted”", "Curly quoted", ["quotes"]),
])
def test_anonymise(raw, clean, fired):
    assert gen.anonymise(raw) == (clean, fired)


def test_opaque_ids_are_stable_and_distinct():
    a = gen.opaque_id("gpt-6-sol", "cr-1")
    assert a == gen.opaque_id("gpt-6-sol", "cr-1")
    assert a != gen.opaque_id("gpt-6-luna", "cr-1") and len(a) == 16
    assert "gpt" not in a


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("text, alt", [
    ('{"alt": "A dog."}', "A dog."),
    ('```json\n{"alt": "A dog."}\n```', "A dog."),
    ('Sure! {"alt": "A dog."} Hope it helps', "A dog."),
    ('{"alt": 3}', None),
    ("A dog.", None),
    ("", None),
])
def test_parse_alt(text, alt):
    assert gen.parse_alt(text) == alt


def test_parse_judge_replies():
    assert jd.parse_score('{"score": 4}') == 4
    assert jd.parse_score('{"score": "5"}') == 5
    assert jd.parse_score('{"score": 4.0}') == 4
    assert jd.parse_score('{"score": 6}') is None
    assert jd.parse_score('{"score": true}') is None
    assert jd.parse_better('{"better": 2, "confidence": 4}') == (2, 4)
    assert jd.parse_better('{"better": "1"}') == (1, None)
    assert jd.parse_better('{"better": 3, "confidence": 4}') is None
    assert jd.parse_better("Candidate 1 is better") is None


# ------------------------------------------------------------------ prompts

def _texts(msg):
    return [p for p in msg["content"] if p["type"] == "text"]


def test_generation_prompt(items):
    it = next(i for i in items if i["dataset"] == "cr")
    (msg,) = gen.generation_messages(it)
    assert msg["content"][0]["type"] == "image"
    text = msg["content"][1]["text"]
    assert "[image]" not in text and "{" + "ARTICLE_TITLE}" not in text
    assert "Article title: Riverside" in text and 'Respond with JSON only: {"alt": "<your alt text>"}' in text


def test_pairwise_prompt_split_and_cache_marker(items):
    it = items[0]
    (msg,) = jd.pairwise_messages(it, "First {CANDIDATE_2} text", "Second")
    parts = msg["content"]
    assert parts[0]["type"] == "image"
    head, tail = _texts(msg)
    assert head.get("cache_breakpoint") is True and not tail.get("cache_breakpoint")
    assert tail["text"].startswith('Candidate 1: "First {CANDIDATE_2} text"')   # no re-substitution
    assert 'Candidate 2: "Second"' in tail["text"]
    full = head["text"] + tail["text"]
    template = gen.prompt_template("e3_judge_pairwise.txt")
    assert full == jd.fill(template, jd.context_values(it["context"]) |
                           {"CANDIDATE_1": "First {CANDIDATE_2} text", "CANDIDATE_2": "Second"})
    assert "Implementation notes" not in full and "---" not in full
    # the shared prefix is identical for every pair of the same item
    (msg2,) = jd.pairwise_messages(it, "x", "y")
    assert _texts(msg2)[0]["text"] == head["text"]


def test_pointwise_prompt(items):
    (msg,) = jd.pointwise_messages(items[0], "A table.")
    head, tail = _texts(msg)
    assert tail["text"].startswith('Candidate alt text: "A table."')
    assert '{"score": 1, 2, 3, 4 or 5}' in tail["text"]


# ------------------------------------------------------------------ degradations and pairs

def test_degradations(items):
    pool = [i for i in items if i["dataset"] == "blv"]
    it = next(i for i in pool if i["description"].startswith("A table set"))
    orig = it["description"]
    assert jd.degrade(it, "repeat", pool) == f"{orig} {orig}"
    assert jd.degrade(it, "truncate", pool) == "A table set"
    app = jd.degrade(it, "append", pool)
    assert app.startswith(orig + " ") and app == f"{orig} A church with a green copper roof."  # other image only
    assert jd.degrade(it, "swap", pool) == "A church with a green copper roof."
    assert jd.degrade(it, "append", pool) == app                     # seeded
    short = next(i for i in pool if i["description"] == "Plates.")
    assert jd.degrade(short, "truncate", pool) is None


def test_first_sentence():
    assert jd.first_sentence("A long table. Six chairs.") == "A long table."
    assert jd.first_sentence("No period here") == "No period here"
    assert jd.first_sentence("Dr. Who? Yes.") == "Dr."


def test_known_pairs_counts(items):
    pairs, skipped = jd.known_pairs(items)
    assert len(pairs) + len(skipped) == 4 * len(items)
    assert all(s["degradation"] == "truncate" for s in skipped)       # only 'Plates.'-style descriptions


def test_blv_pairs_same_article_only(items):
    ps = jd.blv_pairs(items)
    assert len(ps) == 1
    p = ps[0]
    assert {p["a"]["description"], p["b"]["description"]} == {"A table set for dinner with candles.", "Plates."}
    winner = p["a"] if p["human_prefers"] == "a" else p["b"]
    assert winner["description"] == "A table set for dinner with candles."


@pytest.mark.parametrize("n_pairs, degree", [(16, 4), (12, 3)])
def test_sample_pairs_balanced(n_pairs, degree):
    for img in ("cr-1", "cr-2", "cr-3"):
        pairs = jd.sample_pairs(GENS8, img, n_pairs)
        assert len(pairs) == n_pairs and len(set(pairs)) == n_pairs
        assert all(a != b for a, b in pairs)
        deg = collections.Counter(itertools.chain.from_iterable(pairs))
        assert set(deg.values()) == {degree} and set(deg) == set(GENS8)
        assert pairs == jd.sample_pairs(GENS8, img, n_pairs)            # seeded per image


def test_sample_pairs_cover_all_pairs_evenly():
    counts = collections.Counter()
    for k in range(120):
        counts.update(jd.sample_pairs(GENS8, f"cr-{k}", 16))
    assert len(counts) == 28                                             # every pair occurs
    assert min(counts.values()) >= 40 and max(counts.values()) <= 100    # expected 120*16/28 = 68.6


def test_sample_pairs_edge_cases():
    assert len(jd.sample_pairs(GENS8[:3], "x", 5)) == 3                  # all pairs when n >= C(g,2)
    pairs = jd.sample_pairs(GENS8[:5], "x", 3)                           # 2n not a multiple of g
    deg = collections.Counter(itertools.chain.from_iterable(pairs))
    assert len(pairs) == 3 and max(deg.values()) - min(deg.values()) <= 1


def test_gen_pair_plan_drops_missing():
    outputs = {(g, "cr-1"): {} for g in GENS8 if g != "gemma-4-31b"}
    plan, dropped = jd.gen_pair_plan(GENS8, ["cr-1"], outputs, 16)
    assert len(plan) + len(dropped) == 16
    assert len(dropped) == 4 and all(d["missing"] == ["gemma-4-31b"] for d in dropped)


def test_image_sample_is_seeded_and_nested(items):
    many = [dict(items[-1], image_id=f"cr-{k}", item_id=f"cr-{k}") for k in range(200)]
    full = gen.sample_images(many, 120)
    assert len(full) == 120 == len(set(full))
    assert gen.sample_images(many, 80) == full[:80] and gen.sample_images(many, 10) == full[:10]
    assert full == gen.sample_images(many, 120)


def test_pilot_subset(items):
    sub = jd.pilot_subset(items, 4)
    assert collections.Counter(i["dataset"] for i in sub) == {"blv": 2, "cr": 2}


# ------------------------------------------------------------------ fake-client runs

class FakeClient:
    """Returns scripted replies; records the messages it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def call(self, model, messages, item_id, purpose="eval", **kw):
        self.sent.append((model, messages, purpose))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return types.SimpleNamespace(text=r, status="ok", cost_usd=0.01, total_s=1.0, ttft_s=0.5,
                                     call_ids=[f"c{len(self.sent)}"], model_returned=model,
                                     assistant={"role": "assistant", "provider": "x", "content": [], "raw": r})


def test_judge_once_retry_then_ok(items):
    c = FakeClient(["I prefer the first.", '{"better": 1, "confidence": 3}'])
    out = jd.judge_once(c, "gpt-6-sol", jd.pairwise_messages(items[0], "a", "b"), jd.parse_better, "t")
    assert out["value"] == (1, 3) and out["retried"] and out["parse_ok"]
    assert out["cost_usd"] == pytest.approx(0.02) and len(out["call_ids"]) == 2
    retry_msgs = c.sent[1][1]
    assert retry_msgs[-1]["content"][0]["text"] == "Respond with valid JSON only."
    assert all(p == "judge" for _, _, p in c.sent)


def test_judge_once_gives_up_after_one_retry(items):
    c = FakeClient(["no", "still no"])
    out = jd.judge_once(c, "gpt-6-sol", jd.pointwise_messages(items[0], "a"), jd.parse_score, "t")
    assert out["value"] is None and out["retried"] and not out["parse_ok"] and out["status"] == "ok"


def test_judge_once_records_api_errors(items):
    c = FakeClient([llm.LLMError("HTTP 500", 500, ["c1"], 0.0)])
    out = jd.judge_once(c, "gpt-6-sol", jd.pointwise_messages(items[0], "a"), jd.parse_score, "t")
    assert out["status"] == "error" and out["value"] is None


def test_generate_one_anonymises(items):
    it = next(i for i in items if i["dataset"] == "cr")
    c = FakeClient(['{"alt": "Image of a stone bridge over a river."}'])
    rec = gen.generate_one(c, "gpt-6-luna", it)
    assert rec["status"] == "ok" and rec["alt_clean"] == "A stone bridge over a river."
    assert rec["transforms"] == ["image_of"] and rec["length"] == len(rec["alt_clean"])
    assert c.sent[0][2] == "eval"


def test_generate_one_parse_failure(items):
    it = next(i for i in items if i["dataset"] == "cr")
    rec = gen.generate_one(FakeClient(["A bridge.", "A bridge, again."]), "gpt-6-luna", it)
    assert rec["status"] == "parse_error" and rec["retried"] and rec["alt_clean"] is None


def test_known_pair_records(items, tmp_path):
    pairs, _ = jd.known_pairs(items[:1] + [i for i in items if i["dataset"] == "blv"])
    pairs = [p for p in pairs if p["item"]["item_id"] == items[0]["item_id"]][:1]
    c = FakeClient(['{"better": 1}', '{"better": 1}'])                  # always prefers the first shown
    sink = jd.Sink(tmp_path / "known.jsonl")
    jd.do_known(c, ["gpt-6-sol"], pairs, sink, workers=1)
    sink.finish(lambda r: not r["original_first"])
    a, b = sink.records
    assert a["original_first"] and a["correct"] is True and a["y"] == 1
    assert not b["original_first"] and b["correct"] is False and b["y"] == 1


def test_leakage_check_detects_style():
    recs = []
    for k in range(30):
        recs.append({"generator": "claude-opus-5-5", "alt_clean": f"A photograph showing scene number {k} with care."})
        recs.append({"generator": "gpt-6-sol", "alt_clean": f"Close-up: item {k}; bright; crisp; sharp."})
    res = gen.leakage_check(recs)
    assert res["majority_baseline"] == 0.5 and res["accuracy"] > 0.9 and res["folds"] == 5


def test_plan_counts(items, monkeypatch):
    monkeypatch.setattr(jd, "load_items", lambda: items)
    monkeypatch.setattr(jd, "sample_images", lambda its, n: ["cr-11", "cr-22"][:n])
    out = jd.plan_counts(["gpt-6-sol", "claude-sonnet-5-5"], jd.PROCEDURES, pilot=False, lean=False)
    assert out["pointwise"] == 2 * len(items)
    assert out["known"] == 2 * 2 * (4 * len(items) - 1)                # one truncate does not apply
    assert out["blv_pairs"] == 2 * 2 * 1
    assert out["gen_pairs"] == 2 * 2 * 2 * 16
    assert out["total"] == out["pointwise"] + out["known"] + out["blv_pairs"] + out["gen_pairs"]
