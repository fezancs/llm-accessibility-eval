"""Integration: experiment scripts -> scored records -> analysis -> LaTeX outputs.

Small slices of E1, E2 and E3 run through fake models (synthetic replies) into a pytest tmp
study directory; then analysis.run_all writes values.tex, the three table bodies and the two
figures into a tmp paper/ directory. Nothing touches runs/ or paper/. This checks that the
record formats the runners write are the ones the analysis reads (harness/SCHEMAS.md).
"""
import json
import re
import shutil

import pytest
import yaml

from harness import config, e1_audit, e1_corpus, e1_gen, e1_repair, e2_padding, e2_run, e3_generate, e3_judge, llm, runmeta
from tests import fakes
from tests import test_e1_end_to_end as e1t
from tests import test_e2_end_to_end as e2t

pytestmark = pytest.mark.skipif(not (e1_corpus.CORPUS / "ground_truth.json").exists(), reason="corpus not built")

FAMILIES = {"fake-good": "fam-a", "fake-lazy": "fam-b", "fake-judge-c": "fam-c", "fake-judge-d": "fam-d"}


def e3_reply(model, last):
    if '{"alt"' in last:
        return json.dumps({"alt": f"A clinic scene described by {model}, with people and signs."})
    if "Candidate alt text:" in last:
        n = len(re.search(r'Candidate alt text: "(.*)"', last, re.S).group(1))
        return json.dumps({"score": 2 + min(3, n // 40)})
    c1 = re.search(r'Candidate 1: "(.*?)"\nCandidate 2:', last, re.S).group(1)
    c2 = re.search(r'Candidate 2: "(.*?)"\n', last, re.S).group(1)
    if model == "fake-judge-d":
        return json.dumps({"better": 1, "confidence": 3})                     # pure position bias
    better = 1 if (len(set(c1.split())) / max(1, len(c1.split()))) * min(len(c1), 200) >= \
        (len(set(c2.split())) / max(1, len(c2.split()))) * min(len(c2), 200) else 2
    return json.dumps({"better": better, "confidence": 4})


def responder(model, system, messages, tools):
    last = fakes.text_of(messages[-1]) if messages[-1]["role"] != "tool" else ""
    last_user = next(fakes.text_of(m) for m in reversed(messages) if m["role"] == "user")
    if system and (system.startswith("You are a senior") or "CityCare" in system):
        return e2t.responder("fake-good" if model == "fake-good" else "fake-lazy", system, messages, tools)
    if "alt text" in last_user.lower() and "blind or have low vision" in last_user:
        return e3_reply(model, last_user)
    return e1t.responder("fake-good" if model == "fake-good" else "fake-bad", system, messages, tools)


def test_pipeline_to_latex(monkeypatch, tmp_path):
    models = [fakes.fake_model(k, family=f, access="open" if k == "fake-lazy" else "closed")
              for k, f in FAMILIES.items()]
    fakes.install(monkeypatch, tmp_path, responder, models)
    registry = {**config.models_config(), "models": models, "judges": {f: k for k, f in FAMILIES.items()}}
    monkeypatch.setattr(config, "models_config", lambda: registry)
    monkeypatch.setattr(runmeta, "prereg_hash", lambda: "abc123def456")
    monkeypatch.setattr(e2_padding, "TRANSCRIPT", tmp_path / "data" / "e2_padding_transcript.json")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.yaml").write_text(yaml.safe_dump(registry))
    (tmp_path / "data" / "e3").mkdir(parents=True)
    shutil.copy(config.DATA / "e3" / "items.jsonl", tmp_path / "data" / "e3" / "items.jsonl")

    e1 = ["--models", "fake-good,fake-lazy", "--workers", "4"]
    e1_gen.main(e1 + ["--tasks", "G03", "--samples", "1"])
    e1_repair.main(e1 + ["--pages", "B01-V1,B05-V1"])
    e1_audit.main(e1 + ["--pages", "B01-V1,B05-V1"])
    e2_padding.generate(llm.LLM("e2"), "fake-good")
    assert e2_run.main(["--models", "fake-good,fake-lazy", "--runs", "1", "--positions", "6", "--padded", "6",
                        "--workers", "4", "--threads", "4"]) == 0
    e3_generate.main(["--pilot", "--models", "fake-good,fake-lazy,fake-judge-c,fake-judge-d"])
    e3_judge.main(["--pilot", "--models", ",".join(FAMILIES)])
    runs = tmp_path / "runs"
    shutil.copytree(runs / "pilot" / "e3", runs / "e3", dirs_exist_ok=True)      # pilot-sized E3 as study data

    from analysis import run_all
    paper = tmp_path / "paper"
    assert run_all.main(["--root", str(tmp_path), "--paper", str(paper), "--summary", str(tmp_path / "summary.md"),
                         "--B", "50"]) == 0
    values = (paper / "results" / "values.tex").read_text()
    keys = dict(re.findall(r"\\setres\{([^}]+)\}\{([^}]*)\}", values))
    for k in ("n-models-e1", "n-models-e2", "axe-version", "chromium-version", "probe-count", "prereg-hash",
              "total-calls", "total-cost", "e2-mean-tool-cycles"):
        assert k in keys, k
    assert keys["probe-count"] == "27" and keys["prereg-hash"].startswith("abc123")
    # Values that follow from the scripted models: fake-good is perfect on E1, fake-lazy is not.
    assert keys["n-models-e1"] == "2" and keys["n-models-e2"] == "2"
    assert keys["sgen-fake-good"] == "1.00" and float(keys["sgen-fake-lazy"]) < 1.0
    assert keys["srep-fake-good"] == "1.00" and keys["srep-fake-lazy"] == "0.00"
    assert keys["saudit-fake-good"] == "1.00"
    assert "dcdi-fake-good" in keys and "dcdi-fake-lazy" in keys
    assert int(keys["e3c-judgments"]) > 0
    main_rows = (paper / "results" / "tab_main_body.tex").read_text()
    assert "Fake Good" in main_rows and "Fake Lazy" in main_rows
    for fig in ("fig_drift.pdf", "fig_pareto.pdf"):
        assert (paper / "figures" / fig).stat().st_size > 1000
    summary = (tmp_path / "summary.md").read_text()
    for section in ("## E1-Gen", "## E1-Repair", "## E1-Audit", "## E2", "## E3", "## Hypotheses"):
        assert section in summary, section
