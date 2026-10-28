"""E1 end to end through the fake provider (synthetic responses, tmp runs/ only).

fake-good returns accessible pages, perfect repairs (the base page) and the true faults;
fake-bad returns inaccessible pages, the faulty page unchanged and an empty audit. The
scores must separate them exactly as SPEC 5.1-5.4 define.
"""
import json

import pytest

from harness import config, e1_audit, e1_corpus, e1_gen, e1_repair
from tests import fakes

FIX = config.HARNESS / "probe_fixtures"
GT_PATH = e1_corpus.CORPUS / "ground_truth.json"
PAGES = ["B01-V1", "B05-V1"]
pytestmark = pytest.mark.skipif(not GT_PATH.exists(), reason="corpus not built yet")


def _fenced(html: str) -> str:
    return "Here it is.\n```html\n" + html + "\n```\n"


def responder(model, system, messages, tools):
    last = fakes.text_of(messages[-1])
    good = model == "fake-good"
    if "conform to WCAG 2.2 Level AA? Answer with JSON only" in last or "Respond with valid JSON only" in last:
        return json.dumps({"conforms_wcag22_aa": True, "known_issues": []})
    if "Return one complete, self-contained HTML document" in last:
        return _fenced((FIX / ("tabs_behavior__pass.html" if good else "tabs_behavior__fail.html")).read_text())
    gt = json.loads(GT_PATH.read_text())
    vid = next(v for v in gt if (e1_corpus.VARIANTS / f"{v}.html").read_text() in last)
    if last.startswith("The HTML page below has accessibility problems"):
        src = e1_corpus.BASE / f"{gt[vid]['base_id']}.html" if good else e1_corpus.VARIANTS / f"{vid}.html"
        return _fenced(src.read_text())
    if last.startswith("Audit the HTML page below"):
        if not good:
            return "[]"
        element = {"M06": "html", "M12": "title", "M10": "a:focus"}
        return json.dumps([{"success_criterion": f["scs"][0],
                            "element": element.get(f["operator"], "#" + f["target_id"].split(">")[0]),
                            "problem": "seeded"} for f in gt[vid]["faults"]])
    raise AssertionError(f"unexpected prompt: {last[:80]}")


@pytest.fixture
def env(monkeypatch, tmp_path):
    models = [fakes.fake_model("fake-good", roles=("e1",)), fakes.fake_model("fake-bad", roles=("e1",))]
    return fakes.install(monkeypatch, tmp_path, responder, models), tmp_path / "runs"


def _rows(runs, name):
    return [json.loads(line) for line in (runs / "e1" / "scores" / name).read_text().splitlines()]


def test_e1_gen_end_to_end(env):
    adapter, runs = env
    e1_gen.main(["--models", "fake-good,fake-bad", "--tasks", "G03", "--samples", "1", "--workers", "2"])
    rows = {(r["model_key"], r["condition"]): r for r in _rows(runs, "gen.jsonl")}
    assert len(rows) == 4
    for (model, _), r in rows.items():
        assert r["status"] == "ok" and r["html_found"] and r["functional"], r
        assert r["selfassess"]["parse_ok"] and r["selfassess"]["conforms"] is True
        assert r["cost_usd"] > 0 and r["latency_s"] is not None and r["checks"]
    assert rows[("fake-good", "nudged")]["a"] == 1.0
    assert rows[("fake-bad", "nudged")]["a"] < rows[("fake-good", "nudged")]["a"]
    failed = {c["id"] for c in rows[("fake-bad", "nudged")]["checks"] if not c["passed"]}
    assert "probe:tabs_behavior" in failed
    meta = json.loads((runs / "meta.json").read_text())
    assert meta["axe_version"] and meta["chromium_version"] and meta["probe_count"] == 27
    n = len(adapter.calls)
    e1_gen.main(["--models", "fake-good,fake-bad", "--tasks", "G03", "--samples", "1", "--workers", "2"])
    assert len(adapter.calls) == n                         # resumable: everything served from the cache


def test_e1_repair_and_audit_end_to_end(env):
    _, runs = env
    args = ["--models", "fake-good,fake-bad", "--pages", ",".join(PAGES), "--workers", "4"]
    e1_repair.main(args)
    rep = {(r["model_key"], r["page_id"]): r for r in _rows(runs, "repair.jsonl")}
    for vid in PAGES:
        g, b = rep[("fake-good", vid)], rep[("fake-bad", vid)]
        assert all(f["resolved"] for f in g["faults"]), g["faults"]
        assert g["regressions"] == 0 and g["preserved"] and g["score"] == 1.0 and g["full_repair"], g
        assert not any(f["resolved"] for f in b["faults"]), b["faults"]
        assert b["regressions"] == 0 and b["score"] == 0.0, b
        assert g["provenance"] == ("w3c" if vid.startswith("B01") else "authored")
    e1_audit.main(args)
    aud = {(r["model_key"], r["page_id"]): r for r in _rows(runs, "audit.jsonl")}
    for vid in PAGES:
        assert (aud[("fake-good", vid)]["tp"], aud[("fake-good", vid)]["fp"]) == (4, 0), aud[("fake-good", vid)]
        assert (aud[("fake-bad", vid)]["tp"], aud[("fake-bad", vid)]["fn"]) == (0, 4)
        assert aud[("fake-bad", vid)]["parse_ok"]
