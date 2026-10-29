"""E2 step loader, frozen permutations and the initial project (zero WCAG A/AA axe
violations, all generic probes pass)."""
import json
import random

import pytest

from harness import e2_steps, oracle


def test_step_file_shape():
    ids = [s["id"] for s in e2_steps.steps()]
    assert e2_steps.foundation_ids() == ["F01", "F02", "F03", "F04"]
    assert len(e2_steps.permuted_ids()) == 20 and len(set(ids)) == 24
    assert e2_steps.seeds() == [101, 202, 303]
    assert e2_steps.max_tool_calls() == 10
    assert e2_steps.padded_checkpoints() == [8, 12, 16, 20, 24]
    for s in e2_steps.steps():
        assert s["scope"]["pages"] and s["functional"]
        assert all("page" in c for c in s["functional"])


def test_permutations_deterministic_and_frozen():
    non_f = e2_steps.permuted_ids()
    for seed in e2_steps.seeds():
        order = e2_steps.permutation(seed)
        assert order == random.Random(seed).sample(non_f, k=20)
        assert order == e2_steps.permutation(seed)                   # deterministic
        assert sorted(order) == sorted(non_f)                        # a permutation
        assert e2_steps.session_order(seed) == ["F01", "F02", "F03", "F04"] + order
    orders = [e2_steps.permutation(s) for s in e2_steps.seeds()]
    assert len({tuple(o) for o in orders}) == 3                      # three different orders
    frozen = json.loads(e2_steps.PERMUTATIONS.read_text(encoding="utf-8"))
    assert frozen == e2_steps.compute_orders()
    assert e2_steps.frozen_orders() == {s: e2_steps.session_order(s) for s in e2_steps.seeds()}
    assert frozen["sessions"]["101"][:6] == ["F01", "F02", "F03", "F04", "P05", "R07"]


def test_frozen_orders_detect_tampering(tmp_path, monkeypatch):
    data = e2_steps.compute_orders()
    data["sessions"]["101"] = list(reversed(data["sessions"]["101"]))
    p = tmp_path / "perm.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(e2_steps, "PERMUTATIONS", p)
    with pytest.raises(RuntimeError):
        e2_steps.frozen_orders()


def test_initial_project_files():
    root = e2_steps.INITIAL_PROJECT
    assert sorted(p.name for p in root.iterdir()) == ["app.js", "index.html", "styles.css"]
    html = (root / "index.html").read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert '<nav aria-label="Main">' in html and html.count("<a ") == 1 and ">Home</a>" in html
    assert "<h1>Welcome to CityCare</h1>" in html and "1 Market Street, Old Town" in html and "Phone:" in html
    assert (root / "app.js").read_text(encoding="utf-8") == ""
    assert ":focus-visible" in (root / "styles.css").read_text(encoding="utf-8")


def test_initial_project_is_clean():
    from harness.render import Renderer
    with Renderer() as r:
        ev = oracle.evaluate(r, e2_steps.INITIAL_PROJECT / "index.html", generic=True)
        with r.page(e2_steps.INITIAL_PROJECT / "index.html") as page:
            page.keyboard.press("Tab")
            nav_outline = page.evaluate("() => document.activeElement.matches('nav a') && getComputedStyle(document.activeElement).outlineStyle")
    assert ev["axe_violations"] == [] and ev["violating_nodes"] == 0
    assert ev["a"] == 1.0 and ev["n_checks"] > 10
    generic = {p["probe"]: p for p in ev["probes"]}
    assert set(generic) == {"tab_reachability", "focus_visible", "no_keyboard_trap"}
    assert all(p["applicable"] and p["passed"] for p in generic.values()), generic
    assert any(c["id"] == "axe:color-contrast" and c["passed"] for c in ev["checks"])
    # P04's intent check needs the browser focus ring (outline-style auto) on nav links at the start
    assert nav_outline == "auto"
