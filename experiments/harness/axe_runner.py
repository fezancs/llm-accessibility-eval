"""axe-core runner (SPEC 3): WCAG 2.x A/AA rules, optional scope, rule-level summary."""
from __future__ import annotations

import json

from playwright.sync_api import Page

from . import config

WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]
_AXE_SOURCE: str | None = None


def axe_source() -> str:
    global _AXE_SOURCE
    if _AXE_SOURCE is None:
        _AXE_SOURCE = config.AXE_JS.read_text(encoding="utf-8")
    return _AXE_SOURCE


def axe_version() -> str:
    pkg = config.AXE_JS.parent / "package.json"
    return json.loads(pkg.read_text(encoding="utf-8"))["version"]


def split_selector(selector: str | None) -> list[str] | None:
    """'#a, #b' -> ['#a', '#b'] (top-level commas only)."""
    if not selector:
        return None
    parts, depth, cur = [], 0, ""
    for ch in selector:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


_RUN = """
async ([include, tags, rules]) => {
  const ctx = include ? {include: include.filter(s => document.querySelector(s)).map(s => [s])} : document;
  if (include && ctx.include.length === 0) return {missingScope: true};
  const opts = rules ? {runOnly: {type: 'rule', values: rules}} : {runOnly: {type: 'tag', values: tags}};
  opts.resultTypes = ['violations', 'incomplete'];
  const r = await axe.run(ctx, opts);
  const nodes = ns => ns.map(n => ({target: n.target.map(String).join(' '), html: (n.html || '').slice(0, 300)}));
  const pick = arr => arr.map(x => ({id: x.id, impact: x.impact, tags: x.tags, nodes: nodes(x.nodes)}));
  return {violations: pick(r.violations), passes: r.passes.map(x => ({id: x.id, n: x.nodes.length})),
          incomplete: pick(r.incomplete), inapplicable: r.inapplicable.map(x => x.id),
          version: axe.version};
}
"""


def run_axe(page: Page, scope: str | None = None, rules: list[str] | None = None) -> dict:
    """Run axe on the page (or within `scope`). `rules` restricts to named rules.

    Returns {rules: {rule_id: passed_bool}, violations, incomplete, missing_scope, version}.
    A rule is "evaluated" if it had passing or violating nodes in scope (SPEC 3.1).
    """
    if not page.evaluate("() => typeof window.axe !== 'undefined'"):
        page.add_script_tag(content=axe_source())
    res = page.evaluate(_RUN, [split_selector(scope), WCAG_TAGS, rules])
    if res.get("missingScope"):
        return {"rules": {}, "violations": [], "incomplete": [], "missing_scope": True, "version": None}
    status = {p["id"]: True for p in res["passes"] if p["n"] > 0}
    for v in res["violations"]:
        status[v["id"]] = False
    return {"rules": status, "violations": res["violations"], "incomplete": res["incomplete"],
            "missing_scope": False, "version": res["version"]}


def merge(a: dict, b: dict) -> dict:
    """Combine two runs (t = 0 s and t = 6 s): a rule fails if it failed in either run."""
    rules = dict(a["rules"])
    for k, ok in b["rules"].items():
        rules[k] = rules.get(k, True) and ok
    seen = {v["id"] for v in a["violations"]}
    return {"rules": rules, "violations": a["violations"] + [v for v in b["violations"] if v["id"] not in seen],
            "incomplete": a["incomplete"], "missing_scope": a["missing_scope"] and b["missing_scope"],
            "version": a["version"] or b["version"]}
