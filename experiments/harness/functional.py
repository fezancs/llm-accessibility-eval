"""Functional checks from the task files (E1-Gen `functional`, E2 `functional`).

A sample or step that fails any functional check scores 0 (E1) or is dropped from the
DCDI comparison (E2). Checks are lenient about visibility on purpose: they test that the
requested thing was built, not that it is accessible.
"""
from __future__ import annotations

from playwright.sync_api import Page

_TEXT_JS = """(sel) => { const roots = sel ? Array.from(document.querySelectorAll(sel)) : [document.body];
    return roots.map(r => [r.textContent, r.innerText,
        ...Array.from(r.querySelectorAll('input[type=submit],input[type=button],input[type=reset]')).map(i => i.value),
        ...Array.from(r.querySelectorAll('[aria-label]')).map(e => e.getAttribute('aria-label'))].join(' '))
      .join(' ').replace(/\\s+/g, ' ').toLowerCase(); }"""


def _text(page: Page, selector: str | None) -> str:
    return page.evaluate(_TEXT_JS, selector)


def check_one(page: Page, chk: dict) -> bool:
    t = chk["type"]
    if t == "min_count":
        return page.evaluate("(s) => document.querySelectorAll(s).length", chk["selector"]) >= chk["n"]
    if t == "text_present":
        return chk["text"].lower() in _text(page, None)
    if t == "text_in":
        return chk["text"].lower() in _text(page, chk["selector"])
    if t == "exists":
        return bool(page.evaluate("(s) => !!document.querySelector(s)", chk["selector"]))
    if t == "link_to":
        return bool(page.evaluate("""(target) => Array.from(document.querySelectorAll('a[href]')).some(a => {
            try { return new URL(a.getAttribute('href'), location.href).pathname.endsWith('/' + target); } catch (e) { return false; } })""",
                                  chk["href"]))
    if t == "blocks_empty_submit":
        page.evaluate("() => { window.__se4allMarker = 1; }")
        url = page.url
        submitted = page.evaluate("""(sel) => { const f = document.querySelector(sel); if (!f) return null;
            const b = f.querySelector('button:not([type=button]):not([type=reset]), input[type=submit]');
            try { if (b) b.click(); else f.requestSubmit(); } catch (e) { return 'error'; } return 'ok'; }""", chk["selector"])
        if submitted is None:
            return False
        page.wait_for_timeout(800)
        try:
            alive = page.evaluate("() => window.__se4allMarker === 1")
        except Exception:                       # noqa: BLE001 - navigated away
            return False
        return alive and page.url.split("#")[0] == url.split("#")[0]
    raise ValueError(f"unknown functional check {t}")


def run_checks(renderer, path, checks: list[dict], page_key: str = "page") -> list[dict]:
    """Run checks; E2 checks name the page they apply to (`page`), resolved against `path`'s folder."""
    out = []
    for chk in checks:
        target = path.parent / chk[page_key] if page_key in chk else path
        if not target.exists():
            out.append({**chk, "passed": False, "why": "page missing"})
            continue
        with renderer.page(target) as page:
            try:
                ok = check_one(page, chk)
            except Exception as exc:            # noqa: BLE001
                out.append({**chk, "passed": False, "why": f"{type(exc).__name__}: {exc}"})
                continue
        out.append({**chk, "passed": bool(ok)})
    return out
