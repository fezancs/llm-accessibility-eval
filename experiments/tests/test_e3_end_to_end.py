"""End-to-end: e3_generate and e3_judge CLIs through the fake provider (tests/fakes.py).
Synthetic replies only; runs/ and the response cache live in a pytest tmp directory."""
import json

from harness import config, e3_data, e3_generate as gen, e3_judge as jd
from tests import fakes

FIX = config.DATA / "e3" / "test_fixtures"
GENS = ["fake-g1", "fake-g2", "fake-g3"]
JUDGES = ["fake-j1", "fake-j2"]


def _items(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0synthetic")
    blv = [i for i in e3_data.parse_blv(FIX / "blv") if i["context"]["article_title"] == "Dinner"]   # one BLV pair
    cr = [i for i in e3_data.parse_cr(FIX / "cr") if i["item_id"] in ("cr-11", "cr-22")]
    items = blv + cr
    for i in items:
        i.update(retrieved=True, image_path=str(img), media_type="image/jpeg")
    return items


def responder(model, system, messages, tools):
    last = fakes.text_of(messages[-1])
    if last == "Respond with valid JSON only.":
        return '{"alt": "A stone bridge."}' if model.startswith("fake-g") else '{"better": 2, "confidence": 5}'
    prompt = fakes.text_of(messages[0])
    if '{"alt":' in prompt:                                        # generation
        if model == "fake-g2":
            return "Sure, here is the alt text: a stone bridge."  # invalid JSON -> one retry
        return {"fake-g1": '{"alt": "Image of a stone bridge at dusk."}',
                "fake-g3": '{"alt": "\\"A **bridge** over water\\""}'}[model]
    if '"score"' in prompt:                                        # pointwise
        return '{"score": 4}'
    if model == "fake-j2" and "Candidate 1:" in prompt and len(messages) == 1 and "Plates." in prompt:
        return "I cannot decide."                                  # judge retry path
    return '{"better": 1, "confidence": 3}'                        # always prefers the first shown


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_e3_generate_and_judge_end_to_end(monkeypatch, tmp_path):
    models = [fakes.fake_model(k, family=f"fam-{k}") for k in GENS + JUDGES]
    adapter = fakes.install(monkeypatch, tmp_path, responder, models)
    items = _items(tmp_path)
    monkeypatch.setattr(gen, "load_items", lambda: items)
    monkeypatch.setattr(jd, "load_items", lambda: items)
    runs = config.RUNS
    assert str(runs).startswith(str(tmp_path))

    assert gen.main(["--models", ",".join(GENS), "--workers", "4"]) == 0
    generated = _read(runs / "e3" / "generated.jsonl")
    assert len(generated) == 2 * len(GENS)                         # 2 D_CR images x 3 generators
    by = {(r["generator"], r["image_id"]): r for r in generated}
    g1, g2, g3 = (by[(g, "cr-11")] for g in GENS)
    assert g1["alt_clean"] == "A stone bridge at dusk." and g1["transforms"] == ["image_of"]
    assert g2["retried"] and g2["alt_clean"] == "A stone bridge." and g2["status"] == "ok"
    assert g3["alt_clean"] == "A bridge over water" and set(g3["transforms"]) == {"markdown", "quotes"}
    assert len({r["opaque_id"] for r in generated}) == len(generated)
    meta = json.loads((runs / "e3" / "generated_meta.json").read_text())
    assert meta["generators"] == GENS and sorted(meta["images"]) == ["cr-11", "cr-22"]

    assert jd.main(["--models", ",".join(JUDGES), "--workers", "4"]) == 0
    jdir = runs / "e3" / "judgments"
    pw = _read(jdir / "pointwise.jsonl")
    assert len(pw) == len(items) * len(JUDGES) and {r["score"] for r in pw} == {4}
    known = _read(jdir / "known_pairs.jsonl")
    # BLV: one image only -> append/swap not applicable; 'Plates.' cannot be truncated.
    # CR: two images -> all four degradations.  (2 + 1) + (4 + 4) = 11 pairs x 2 orders x 2 judges
    assert len(known) == 11 * 2 * len(JUDGES)
    assert all(r["correct"] == r["original_first"] for r in known if r["judge"] == "fake-j1")
    retried = [r for r in known if r["retried"]]
    assert retried and all(r["judge"] == "fake-j2" and r["better"] == 2 for r in retried)
    blv = _read(jdir / "blv_pairs.jsonl")
    assert len(blv) == 1 * 2 * len(JUDGES) and {r["human_prefers"] for r in blv} == {"a"}
    gp = _read(jdir / "gen_pairs.jsonl")
    assert len(gp) == 2 * 3 * 2 * len(JUDGES)                      # 2 images x C(3,2) pairs x 2 orders x 2 judges
    assert {(r["gen_first"], r["gen_second"]) for r in gp if r["image_id"] == "cr-11" and r["judge"] == "fake-j1"} \
        == {(a, b) for a in GENS for b in GENS if a != b}
    assert all(r["len_first"] > 0 and r["y"] == 1 for r in gp)
    report = json.loads((jdir / "run_report.json").read_text())
    assert report["gen_pairs"]["pairs_dropped"] == 0

    calls = _read(runs / "e3" / "calls.jsonl")
    assert all(c["cost_usd"] > 0 and c["budget_key"] == "e3" for c in calls)
    assert {c["purpose"] for c in calls if c["model_key"] in JUDGES} == {"judge"}
    assert {c["purpose"] for c in calls if c["model_key"] in GENS} == {"eval"}
    assert not list(runs.glob("**/*.partial"))

    # Resumable: a second run is served entirely from the response cache.
    n = len(adapter.calls)
    assert gen.main(["--models", ",".join(GENS)]) == 0
    assert jd.main(["--models", ",".join(JUDGES)]) == 0
    assert len(adapter.calls) == n
    assert _read(jdir / "gen_pairs.jsonl") == gp
