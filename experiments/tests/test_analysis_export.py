"""End to end: run_all on synthetic data in temporary directories (never paper/ or runs/)."""
import hashlib
import pathlib
import re

import pytest

from analysis import run_all
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
    assert keys["total-calls"] == "9"                              # pilot calls excluded
    assert keys["total-cost"] == "\\$11.25"
    assert keys["run-dates"] == "2--5 November 2026"
    assert keys["prereg-hash"] == "abc1234"


def test_every_main_tex_key_outside_results_is_defined(run):
    """Keys the current paper text uses (abstract, design, decision matrix) all exist for full data."""
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
