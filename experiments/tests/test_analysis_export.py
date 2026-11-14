"""End to end: run_all on synthetic data in temporary directories (never paper/ or runs/)."""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import types

import pandas as pd
import pytest

from analysis import export_latex, run_all
from analysis.bootstrap import Est
from tests.test_analysis_synth import build

EXPERIMENTS = pathlib.Path(__file__).resolve().parent.parent
REAL_PAPER = EXPERIMENTS.parent / "paper"


def _digest(root: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    before = _digest(REAL_PAPER)
    root = build(tmp_path_factory.mktemp("exp") / "experiments", n_images=24)
    # The pilot's E2 calls count as E2 study calls (E2 run 0 replays them); its other calls do not.
    # One failed E1 attempt. All at $0, inside the study dates.
    extra = {"runs/pilot/calls.jsonl": [
        {"call_id": "pe2a", "experiment": "pilot", "item_id": "e2/claude-opus-5-5/run0/pos01/F01/session/call0",
         "model_key": "claude-opus-5-5", "status": "ok", "cost_usd": 0.0, "started_at": "2026-11-04T09:00:00+00:00"},
        {"call_id": "pe2b", "experiment": "pilot", "item_id": "e2/claude-opus-5-5/run0/pos01/F01/fresh/call0",
         "model_key": "claude-opus-5-5", "status": "error", "cost_usd": 0.0, "started_at": "2026-11-04T09:01:00+00:00"}],
        "runs/e1/calls.jsonl": [
        {"call_id": "e1err", "experiment": "e1", "item_id": "repair/B01-V1", "model_key": "claude-opus-5-5",
         "status": "error", "cost_usd": 0.0, "started_at": "2026-11-02T11:00:00+00:00"}]}
    for rel, recs in extra.items():
        with open(root / rel, "a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
    # Corpus variants with two byte-identical pairs, as in the frozen corpus (B02-V1 = B02-V2, B03-V1 = B03-V2).
    var = root / "data" / "corpus" / "variants"
    var.mkdir(parents=True)
    for bp in range(1, 13):
        for v in range(1, 5):
            same = bp in (2, 3) and v == 2
            (var / f"B{bp:02d}-V{v}.html").write_text(f"<p>B{bp:02d}-V{1 if same else v}</p>")
    paper = tmp_path_factory.mktemp("paper")
    summary = tmp_path_factory.mktemp("out") / "summary.md"
    assert run_all.main(["--root", str(root), "--paper", str(paper), "--summary", str(summary), "--B", "60"]) == 0
    assert _digest(REAL_PAPER) == before, "the analysis tests must not touch the real paper/ directory"
    return root, paper, summary


def _keys(values_tex: pathlib.Path) -> dict:
    return dict(re.findall(r"\\setres\{([^}]+)\}\{(.*)\}", values_tex.read_text()))


def test_required_keys_present(run):
    _, paper, _ = run
    keys = _keys(paper / "results" / "values.tex")
    required = ["n-models-e1", "n-models-e2", "n-models-e3gen", "axe-version", "chromium-version", "run-dates",
                "total-calls", "total-cost", "e2-mean-tool-cycles", "probe-count", "prereg-hash"]
    required += [f"dm-{c}{s}" for c in ("enterprise", "research", "public", "edge") for s in ("", "-runner")]
    missing = [k for k in required if k not in keys]
    assert not missing, missing
    assert keys["n-models-e1"] == "10" and keys["n-models-e2"] == "7" and keys["n-models-e3gen"] == "8"
    assert keys["total-calls"] == "12"                     # 9 + 1 E1 error + the pilot's 2 E2 calls (not its others)
    assert keys["total-calls-failed"] == "2" and keys["total-calls-answered"] == "10"
    assert keys["e2-calls-from-pilot"] == "2"
    assert keys["total-cost"] == "\\$11.25"
    assert keys["run-dates"] == "2--5 November 2026"
    assert keys["prereg-hash"] == "abc1234"


def test_every_main_tex_key_outside_results_is_defined(run):
    """Keys the current paper text uses (abstract, design, decision matrix) all exist for full data."""
    if not (REAL_PAPER / "main.tex").exists():
        pytest.skip("paper/main.tex is not on this branch (the local-run paper is paper-local/main.tex)")
    _, paper, _ = run
    keys = _keys(paper / "results" / "values.tex")
    used = set(re.findall(r"\\res\{([^}]+)\}", (REAL_PAPER / "main.tex").read_text()))
    assert used, "main.tex should use \\res keys"
    assert not sorted(used - set(keys))


def test_estimates_come_with_cis(run):
    _, paper, _ = run
    keys = _keys(paper / "results" / "values.tex")
    for k in ("dcdi-claude-opus-5-5", "sgen-gpt-6-luna", "srep-qwen3-8-27b", "saudit-gemma-4-31b",
              "s4-llama-4-maverick", "acc-known-cmbpe", "r-blv-claude-sonnet-5-5", "ccs-claude-opus-5-5"):
        assert k in keys and f"{k}-ci" in keys and f"{k}-lo" in keys and f"{k}-hi" in keys, k
    assert keys["h1c-result"] == "not tested"                      # Argon did not run in E2
    assert keys["h1a-result"] in ("supported", "not supported")


def test_e3_difference_length_and_reliability_keys(run):
    """Paired E3 differences, common-item r, alt-text lengths, alpha CIs and per-judge H4a p-values."""
    import itertools

    from tests.test_analysis_synth import GEN, JUDGES
    _, paper, _ = run
    keys = _keys(paper / "results" / "values.tex")
    full = lambda k: [f"{k}{s}" for s in ("", "-ci", "-lo", "-hi", "-n") if f"{k}{s}" not in keys]
    need = [f"s4-diff-{a}-{b}" for a, b in itertools.combinations(sorted(GEN), 2)]
    need += ["acc-known-cmbpe-minus-naive"] + [f"acc-known-{d}-cmbpe-minus-naive"
                                               for d in ("append", "repeat", "swap", "truncate")]
    need += [f"r-{t}-length-minus-{m}" for t in ("blv", "crowd") for m in ("cmbpe", "naive")]
    need += [f"r-{t}-{m}-common" for t in ("blv", "crowd") for m in JUDGES + ["cmbpe", "naive", "length"]]
    need += ["kalpha-e3b", "kalpha-e3c"]
    missing = [k for n in need for k in full(n)]
    assert not missing, missing
    assert f"s4-diff-{sorted(GEN)[1]}-{sorted(GEN)[0]}" not in keys                  # one order per pair
    assert keys["r-blv-cmbpe-common-n"] == keys["r-blv-cmbpe-n"] == "68"            # synthetic judges score every item
    for g in GEN:
        assert re.fullmatch(r"\d+", keys[f"e3c-meanlen-{g}"]), g                     # integer, no interval
        assert f"e3c-meanlen-{g}-ci" not in keys
    assert re.fullmatch(r"\d+\.\d", keys["e3c-sdlen"])
    for j in JUDGES:
        assert re.fullmatch(r"\d\.\d{3}", keys[f"h4a-p-{j}"]), j
    assert "h4a-max-eta" in keys and "h4a-p" in keys


def test_exclusion_count_and_progress_keys(run):
    """E3 exclusions and counts, E2 session extent, E1 duplicate-page sensitivity (SPEC 10)."""
    _, paper, _ = run
    keys = _keys(paper / "results" / "values.tex")
    judges = ["claude-sonnet-5-5", "gpt-6-sol", "gemini-3-8-flash", "qwen3-8-27b"]
    for k in ["e3c-valid", "e3c-aggregated", "e3b-pairs", "e3-images"] + \
            [f"{p}-{j}" for j in judges for p in ("e3c-missing", "e3c-firstpref", "e3b-missing",
                                                   "pw-missing-blv", "pw-missing-crowd")]:
        assert k in keys, k
    assert keys["e3c-pairs"] == "384"                               # 24 images x 16 pairs
    assert keys["e3b-pairs"] == "1{,}088" and keys["e3-images"] == "136"   # 272 items x 4 degradations; 34 + 102
    assert keys["e3gen-ok-gpt-6-sol"] == "24"
    m = "claude-opus-5-5"
    assert keys[f"e2-completed-{m}"] == "24" and keys[f"e2-exhausted-runs-{m}"] == "0"
    assert f"e2-exhausted-at-{m}" not in keys and f"e2-lambda-max-{m}" in keys
    for k in ("checkpoints", "padded-unmatched", "missing-session", "functional-drops"):
        assert f"e2-{k}-{m}" in keys, k
    root, _, _ = run
    steps = pd.read_json(root / "runs/e2/scores/steps.jsonl", lines=True)
    ck = steps[(steps["model_key"] == m) & (steps["arm"] == "session") & steps["position"].isin([8, 12, 16, 20, 24])
               & steps["step_type"].isin(["foundation", "regular"])]
    assert keys[f"e2-checkpoints-{m}"] == str(len(ck))               # regular steps at checkpoint positions
    assert keys["e1-distinct-pages"] == "46" and keys["e1-distinct-faults"] == "184"
    for k in ("srep-dedup", "fullrepair-dedup", "saudit-dedup", "audit-recall-dedup"):
        assert f"{k}-{m}" in keys and f"{k}-{m}-ci" in keys, k
    assert "srep-" + m in keys and keys[f"srep-{m}-n"] == "48" and keys[f"srep-dedup-{m}-n"] == "46"


def _local_inputs(tier_provider: dict) -> types.SimpleNamespace:
    models = {k: {"key": k, "display": k.upper(), "tier": t, "provider": p, "access": "open",
                  "price": {"input": 0, "output": 0}} for k, (t, p) in tier_provider.items()}
    empty = pd.DataFrame()
    return types.SimpleNamespace(models=models, gen=pd.DataFrame({"model_key": list(models)}), repair=empty,
                                 audit=empty, steps=empty, generated=empty, pointwise=empty, known_pairs=empty,
                                 gen_pairs=empty, blv_pairs=empty)


def test_no_optional_dagger_or_hosting_note_for_local_models():
    inputs = _local_inputs({"local-a": ("optional", "local"), "local-b": ("core", "local")})
    body = export_latex.table_models(inputs)
    assert "dagger" not in body and "hosting provider" not in body and "multicolumn" not in body
    res = {"dims": {"local-a": {}}, "ccs": {}, "dcdi": {"models": {}}, "cpds": {"cpds": {}, "pi": {}}}
    assert "dagger" not in export_latex.table_main(res, inputs)
    hosted = _local_inputs({"cloud-a": ("optional", "openrouter"), "local-b": ("core", "local")})
    body = export_latex.table_models(hosted)
    assert "CLOUD-A$^{\\dagger}$" in body and "Optional (budget permitting)" in body and "hosting provider" in body


def test_one_cluster_dcdi_prints_bare_estimate_and_no_ci_keys():
    from analysis.bootstrap import no_ci
    inputs = _local_inputs({"m": ("core", "local")})
    e = no_ci(Est.from_reps(0.25, [0.25] * 20, n=1), 1)
    res = {"dims": {"m": {}}, "ccs": {}, "dcdi": {"models": {"m": {"dcdi": e}}}, "cpds": {"cpds": {}, "pi": {}}}
    row = [r for r in export_latex.table_main(res, inputs).splitlines() if r.startswith("M ")][0]
    assert row.split("&")[8].strip() == "0.25"                       # no "[0.25, 0.25]"
    V = export_latex.Values()
    V.est("dcdi-m", e)
    assert V.v == {"dcdi-m": "0.25", "dcdi-m-n": "1"}


def _load_check_keys(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("check_keys", EXPERIMENTS.parent / "paper-local" / "check_keys.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "main.tex").write_text("\\res{a} \\res{b}\n")
    monkeypatch.setattr(mod, "HERE", tmp_path)
    monkeypatch.setattr(mod, "VALUES", tmp_path / "values.tex")
    monkeypatch.setattr(mod, "HARDWARE", tmp_path / "extra_values.tex")
    return mod


def test_check_keys_flags_stale_extra_values(tmp_path, monkeypatch, capsys):
    mod = _load_check_keys(tmp_path, monkeypatch)
    mod.VALUES.write_text("\\setres{a}{1}\n")
    mod.HARDWARE.write_text("\\setres{b}{2}\n")
    os.utime(mod.HARDWARE, (1_000_000, 1_000_000))                   # extra values older than values.tex
    assert mod.main() == 1 and "  stale:" in capsys.readouterr().out
    os.utime(mod.HARDWARE, None)
    os.utime(mod.VALUES, (1_000_000, 1_000_000))
    assert mod.main() == 0 and "  stale:" not in capsys.readouterr().out


def test_extra_values_prompt_rate_uses_uncached_tokens(tmp_path, monkeypatch, capsys):
    import sys
    spec = importlib.util.spec_from_file_location("extra_values", EXPERIMENTS / "local" / "extra_values.py")
    xv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xv)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.yaml").write_text((EXPERIMENTS / "config" / "models.yaml").read_text())
    base = {"provider": "local", "status": "ok", "item_id": "pointwise/x", "output_tokens": 10, "total_s": 5.0}
    recs = [dict(base, model_key="m", input_tokens=1200, cached_input_tokens=534, ttft_s=2.0),   # 600 tok/s
            dict(base, model_key="m", input_tokens=2000, cached_input_tokens=2000, ttft_s=4.0),  # 500 tok/s
            dict(base, model_key="m", input_tokens=300, cached_input_tokens=5000, ttft_s=0.5)]   # < 1,000 new: out
    (tmp_path / "runs" / "e3").mkdir(parents=True)
    (tmp_path / "runs" / "e3" / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(xv, "EXP", tmp_path)
    monkeypatch.setattr(xv, "RUNS", tmp_path / "runs")
    out = tmp_path / "extra_values.tex"
    monkeypatch.setattr(sys, "argv", ["extra_values.py", "--out", str(out)])
    assert xv.main() == 0
    text = out.read_text()
    assert "\\setres{prompt-tps-m}{550}" in text and "\\setres{prompt-tps-m-n}{2}" in text


def _rows(p: pathlib.Path):
    return [l for l in p.read_text().splitlines() if l and not l.startswith("%") and "\\midrule" not in l
            and "\\multicolumn" not in l]


def test_table_layouts(run):
    _, paper, _ = run
    res = paper / "results"
    main_rows = _rows(res / "tab_main_body.tex")
    assert len(main_rows) == 10 and all(r.count("&") == 11 for r in main_rows)
    # models measured on only part of the dimensions show "--" (e.g. Sonnet: E1 only)
    sonnet = next(r for r in main_rows if r.startswith("Claude Sonnet 5.5"))
    assert sonnet.split("&")[3].strip() == "--"
    model_rows = _rows(res / "tab_models_body.tex")
    assert all(r.count("&") == 6 for r in model_rows)
    assert len(model_rows) == 10                                    # models actually run
    assert "multicolumn{7}" in (res / "tab_models_body.tex").read_text()
    cm_rows = _rows(res / "tab_cmbpe_body.tex")
    assert len(cm_rows) == 7 and all(r.count("&") == 6 for r in cm_rows)
    assert cm_rows[0].startswith("Claude Sonnet 5.5 (single)")     # panel order follows models.yaml
    assert "\\tbd" not in (res / "tab_main_body.tex").read_text()


def test_figures_and_summary(run):
    _, paper, summary = run
    for name in ("fig_drift.pdf", "fig_pareto.pdf"):
        p = paper / "figures" / name
        assert p.exists() and p.read_bytes()[:4] == b"%PDF"
    text = summary.read_text()
    for section in ("## E1-Gen", "## E1-Repair", "## E1-Audit", "## E2", "## E3", "## CPDS", "## Hypotheses",
                    "## Decision matrix"):
        assert section in text, section


def test_no_records_leaves_paper_untouched(tmp_path):
    root = tmp_path / "experiments"
    (root / "config").mkdir(parents=True)
    (root / "config" / "models.yaml").write_text((EXPERIMENTS / "config" / "models.yaml").read_text())
    paper = tmp_path / "paper"
    (paper / "results").mkdir(parents=True)
    (paper / "results" / "values.tex").write_text("% placeholder\n")
    summary = tmp_path / "summary.md"
    assert run_all.main(["--root", str(root), "--paper", str(paper), "--summary", str(summary), "--B", "10"]) == 0
    assert (paper / "results" / "values.tex").read_text() == "% placeholder\n"
    assert not (paper / "figures").exists()
    assert "No experiment records exist" in summary.read_text()


def test_partial_data_writes_only_what_ran(tmp_path):
    """Only E1 records: no E2/E3 keys, no drift figure, and no decision a missing experiment drives."""
    root = build(tmp_path / "experiments", n_images=4)
    for rel in ("runs/e2", "runs/e3", "data/e3"):
        for p in sorted((root / rel).rglob("*.jsonl")):
            p.rename(p.with_suffix(".off"))
    paper = tmp_path / "paper"
    assert run_all.main(["--root", str(root), "--paper", str(paper), "--summary", str(tmp_path / "s.md"),
                         "--B", "20"]) == 0
    keys = _keys(paper / "results" / "values.tex")
    assert "n-models-e1" in keys and "n-models-e2" not in keys and "n-models-e3gen" not in keys
    assert not any(k.startswith("dcdi-") or k.startswith("s4-") for k in keys)
    assert "dm-enterprise" not in keys and "dm-public" in keys
    assert not (paper / "figures" / "fig_drift.pdf").exists()
    assert (paper / "figures" / "fig_pareto.pdf").exists()


def test_e1_paired_difference_keys(run):
    """E1 model differences use the shared resamples: one key per pair (a < b), a minus b."""
    import itertools

    from tests.test_analysis_synth import E1
    _, paper, _ = run
    keys = _keys(paper / "results" / "values.tex")
    a, b = sorted(E1)[:2]
    for stem in ("sgen", "sgen-unguided", "gap", "overclaim", "srep", "fullrepair", "saudit", "audit-recall"):
        for s in ("", "-ci", "-lo", "-hi", "-n"):
            assert f"{stem}-diff-{a}-{b}{s}" in keys, (stem, s)
        assert f"{stem}-diff-{b}-{a}" not in keys
    want = float(keys[f"srep-{a}"]) - float(keys[f"srep-{b}"])
    assert abs(float(keys[f"srep-diff-{a}-{b}"].replace("$-$", "-")) - want) <= 0.011    # rounding of the parts
    assert sum(1 for k in keys if re.fullmatch(r"sgen-diff-.*-n", k)) == len(list(itertools.combinations(E1, 2)))
