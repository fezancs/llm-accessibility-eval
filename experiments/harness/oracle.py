"""Rule-level adherence a(x) for one artifact (SPEC 3.1).

Q(x) = every axe rule that evaluated at least one node in scope (passes or violations)
plus every applicable probe (report-only probes excluded). A rule passes with zero
violating nodes; a probe passes per its definition. a(x) = passing / |Q(x)|.
"""
from __future__ import annotations

import contextlib
import pathlib

from . import axe_runner, probes, render

MOTION_RECHECK_S = 6.0


PRE_ACTIONS = {
    "scroll_to_bottom": "() => { window.scrollTo(0, document.documentElement.scrollHeight); }",
}


def _factory(renderer: render.Renderer, path, pre_action: str | None):
    @contextlib.contextmanager
    def open_page():
        with renderer.page(path) as page:
            if pre_action:
                page.evaluate(PRE_ACTIONS[pre_action])
                page.wait_for_timeout(600)
            yield page
    return open_page


def evaluate(renderer: render.Renderer, path: pathlib.Path | str, *, scope: str | None = None,
             axe_rules: list[str] | None = None, probe_names: list[str] = (), anchors: dict | None = None,
             generic: bool = True, pre_action: str | None = None) -> dict:
    """Score one page. `axe_rules=None` runs every WCAG 2.x A/AA rule; a list restricts it."""
    open_page = _factory(renderer, path, pre_action)
    with open_page() as page:
        ax = axe_runner.run_axe(page, scope, axe_rules)
        counts = page.evaluate("""(sel) => { const roots = sel ? sel.split(',').flatMap(s => Array.from(document.querySelectorAll(s.trim()))) : [document.documentElement];
            return roots.reduce((n, r) => n + 1 + r.querySelectorAll('*').length, 0); }""", scope)
        moving = render.has_motion(page)
        if moving:
            waited = page.evaluate("() => performance.now() / 1000")
            page.wait_for_timeout(max(0, int((MOTION_RECHECK_S - waited) * 1000)))
            ax = axe_runner.merge(ax, axe_runner.run_axe(page, scope, axe_rules))
    names = (list(probes.GENERIC) if generic else []) + [p for p in probe_names if p not in probes.GENERIC or not generic]
    probe_results = [probes.run_probe(n, open_page, scope, anchors) for n in names]

    checks = [{"id": f"axe:{rid}", "kind": "axe", "passed": ok} for rid, ok in sorted(ax["rules"].items())]
    report_only = []
    for pr in probe_results:
        if not pr["applicable"]:
            continue
        if pr["report_only"]:
            report_only.append({"id": f"probe:{pr['probe']}", "passed": pr["passed"]})
            continue
        checks.append({"id": f"probe:{pr['probe']}", "kind": "probe", "passed": bool(pr["passed"])})
    n_pass = sum(c["passed"] for c in checks)
    violating = sum(len(v["nodes"]) for v in ax["violations"])
    return {
        "a": (n_pass / len(checks)) if checks else None,
        "n_checks": len(checks), "n_pass": n_pass, "checks": checks, "report_only": report_only,
        "missing_scope": ax["missing_scope"], "moving": moving,
        "n_elements": counts, "violating_nodes": violating,
        "violations_per_100": (100.0 * violating / counts) if counts else None,
        "incomplete": sum(len(v["nodes"]) for v in ax["incomplete"]),
        "axe_violations": [{"id": v["id"], "targets": [n["target"] for n in v["nodes"]][:10]} for v in ax["violations"]],
        "probes": probe_results, "axe_version": ax["version"],
    }
