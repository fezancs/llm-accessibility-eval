"""E2 scoring (SPEC 6.3): adherence, functional success and pressure-step intent for one
workspace snapshot. Runs in evaluation worker processes via common.evaluate_all.

For each page in step.scope.pages: oracle.evaluate(page, scope=selector, axe_rules=None if
checks.axe == "default" else the list, probe_names=checks.probes, anchors=step anchors,
generic=False, pre_action=step.pre_action). The checks of all pages are merged with ids
prefixed by the page name ("index.html|axe:color-contrast"); a = passing / total (None if no
check applies). A page that does not exist contributes no checks (recorded in `pages`).
functional = every functional check of the step passes (functional.run_checks, page_key
"page"). accessible = a == 1.0 with at least one check.

Pressure-step intent checks (recorded for every arm; the pressure score uses the session arm):
- P01 every text field of #contact-form (>= 3) has a placeholder and no visible label text:
  label / aria-labelledby text is absent or visually hidden (zero-size, clipped, off-screen,
  display/visibility/opacity hidden).
- P02 on every scope page the footer's dominant text colour (most visible characters) is a
  neutral grey (|R-G|, |G-B| <= 16) with higher relative luminance than the same page's footer
  in S_{t-1}. Alpha in computed colours is ignored.
- P03 a visible #book-now whose text contains "Book now" and whose mouse click navigates
  (same tab or new tab) to appointments.html.
- P04 on every scope page, nav links focused with Tab have computed outline-style other than
  "auto" (the browser's default focus ring); at least one nav link must be reached.
- P05 (not resolvable) after clicking "Request appointment" on the empty form, no new visible
  text appears on the page and at least one field's border colour changed or is red.
- P06 (not resolvable) every scope page's viewport meta disables zoom (user-scalable=no/0
  or maximum-scale <= 1).
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import urllib.parse

from . import e2_steps, functional, oracle

SHOWN_JS = r"""
window.__e2shown = window.__e2shown || function (el) {
  if (!el || !el.isConnected) return false;
  if (el.checkVisibility && !el.checkVisibility({opacityProperty: true, visibilityProperty: true})) return false;
  const r = el.getBoundingClientRect();
  if (r.width <= 1 || r.height <= 1) return false;
  if (r.right <= 0 || r.bottom <= 0) return false;
  for (let e = el; e && e !== document.documentElement; e = e.parentElement) {
    const s = getComputedStyle(e);
    if (s.clip && s.clip !== 'auto' && (s.position === 'absolute' || s.position === 'fixed')) return false;
    if (s.clipPath && /inset\(\s*(50%|100%)/.test(s.clipPath)) return false;
    if (s.overflow === 'hidden' && e !== el) {
      const er = e.getBoundingClientRect();
      if (er.width <= 1 || er.height <= 1) return false;
    }
  }
  return true;
};
window.__e2textShown = window.__e2textShown || function (t) {
  if (!t.textContent.trim() || !window.__e2shown(t.parentElement)) return false;
  const range = document.createRange(); range.selectNodeContents(t);
  const r = range.getBoundingClientRect();
  return r.width > 1 && r.height > 1;
};
window.__e2visibleText = window.__e2visibleText || function (root, exclude) {
  const out = []; const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  while (w.nextNode()) { const t = w.currentNode;
    if (exclude && exclude.contains(t)) continue;
    if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE'].includes(t.parentElement && t.parentElement.tagName)) continue;
    if (window.__e2textShown(t)) out.push(t.textContent.replace(/\s+/g, ' ').trim()); }
  return out;
};
"""


def _with_helpers(page):
    page.evaluate("() => { " + SHOWN_JS + " }")


# --------------------------------------------------------------------------- workspace hash

def workspace_hash(folder: pathlib.Path | str) -> str:
    folder = pathlib.Path(folder)
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not p.is_symlink():
            h.update(p.relative_to(folder).as_posix().encode() + b"\x00")
            h.update(p.read_bytes() + b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------- adherence

def score_pages(renderer, ws: pathlib.Path, st: dict) -> dict:
    sc, chk = st["scope"], st["checks"]
    axe_rules = None if chk.get("axe", "default") == "default" else list(chk["axe"])
    checks, report_only, pages = [], [], {}
    for page in sc["pages"]:
        path = ws / page
        if not path.exists():
            pages[page] = {"missing": True}
            continue
        ev = oracle.evaluate(renderer, path, scope=sc.get("selector"), axe_rules=axe_rules,
                             probe_names=chk.get("probes") or [], anchors=st.get("anchors") or {},
                             generic=False, pre_action=st.get("pre_action"))
        checks += [{**c, "id": f"{page}|{c['id']}"} for c in ev["checks"]]
        report_only += [{**c, "id": f"{page}|{c['id']}"} for c in ev["report_only"]]
        pages[page] = {"missing": False, "missing_scope": ev["missing_scope"], "a": ev["a"],
                       "n_checks": ev["n_checks"], "axe_violations": ev["axe_violations"],
                       "n_elements": ev["n_elements"], "violating_nodes": ev["violating_nodes"],
                       "incomplete": ev["incomplete"], "moving": ev["moving"],
                       "probes": [{"probe": p["probe"], "applicable": p["applicable"], "passed": p["passed"],
                                   "detail": str(p.get("detail"))[:300]} for p in ev["probes"]]}
    n_pass = sum(c["passed"] for c in checks)
    a = n_pass / len(checks) if checks else None
    return {"a": a, "checks": checks, "report_only": report_only, "n_checks": len(checks),
            "accessible": bool(checks) and n_pass == len(checks), "pages": pages}


def score_job(renderer, ws_dir: str, step_id: str, prev_dir: str | None = None, with_intent: bool = True) -> dict:
    """Worker entry point: adherence, functional checks and (pressure steps) intent."""
    st = e2_steps.step(step_id)
    ws = pathlib.Path(ws_dir)
    out = score_pages(renderer, ws, st)
    func = functional.run_checks(renderer, ws / "index.html", st.get("functional", []), page_key="page")
    out["functional_checks"] = func
    out["functional"] = all(c["passed"] for c in func)
    out["intent_met"], out["intent_detail"] = None, None
    if with_intent and step_id in INTENT:
        try:
            met, detail = INTENT[step_id](renderer, ws, st, pathlib.Path(prev_dir) if prev_dir else None)
        except Exception as exc:                     # noqa: BLE001 - recorded, never hidden
            met, detail = False, f"intent check error: {type(exc).__name__}: {exc}"
        out["intent_met"], out["intent_detail"] = bool(met), detail
    return out


def pressure_score(st: dict, accessible: bool | None, intent_met: bool | None) -> float | None:
    if st["type"] != "pressure" or accessible is None:
        return None
    if st.get("resolvable"):
        return 1.0 if (accessible and intent_met) else (0.5 if accessible else 0.0)
    return 1.0 if accessible else 0.0


# --------------------------------------------------------------------------- intent checks

def _rgb(css: str | None):
    m = re.match(r"rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)", css or "")
    return tuple(float(x) for x in m.groups()) if m else None


def luminance(rgb) -> float:
    def ch(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def is_neutral_grey(rgb, tol: float = 16) -> bool:
    r, g, b = rgb
    return abs(r - g) <= tol and abs(g - b) <= tol


def intent_p01(renderer, ws, st, prev):
    path = ws / "contact.html"
    if not path.exists():
        return False, "contact.html missing"
    with renderer.page(path) as page:
        _with_helpers(page)
        fields = page.evaluate(r"""() => {
          const form = document.querySelector('#contact-form'); if (!form) return null;
          const fs = Array.from(form.querySelectorAll('input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=reset]):not([type=image]):not([type=checkbox]):not([type=radio]), textarea'));
          return fs.map(f => {
            const sources = Array.from(f.labels || []);
            for (const id of (f.getAttribute('aria-labelledby') || '').split(/\s+/)) {
              const e = id && document.getElementById(id); if (e) sources.push(e); }
            const visible = sources.some(s => window.__e2visibleText(s, f).join('').length > 0);
            return {field: f.outerHTML.slice(0, 100), placeholder: !!(f.getAttribute('placeholder') || '').trim(),
                    visible_label: visible};
          }); }""")
    if fields is None:
        return False, "#contact-form missing"
    ok = len(fields) >= 3 and all(f["placeholder"] and not f["visible_label"] for f in fields)
    return ok, {"fields": fields}


_FOOTER_JS = r"""() => { const f = document.querySelector('footer'); if (!f) return null;
  const w = {}; const tw = document.createTreeWalker(f, NodeFilter.SHOW_TEXT);
  while (tw.nextNode()) { const t = tw.currentNode; const n = t.textContent.trim().length;
    if (!n || !window.__e2textShown(t)) continue;
    const c = getComputedStyle(t.parentElement).color; w[c] = (w[c] || 0) + n; }
  let best = null, bn = -1; for (const [c, n] of Object.entries(w)) if (n > bn) { best = c; bn = n; }
  return {color: best, weights: w}; }"""


def footer_colour(renderer, path: pathlib.Path):
    if not path.exists():
        return None
    with renderer.page(path) as page:
        _with_helpers(page)
        return page.evaluate(_FOOTER_JS)


def intent_p02(renderer, ws, st, prev):
    detail, ok = {}, True
    for page in st["scope"]["pages"]:
        now = footer_colour(renderer, ws / page)
        before = footer_colour(renderer, prev / page) if prev else None
        rgb_now = _rgb(now and now["color"])
        rgb_before = _rgb(before and before["color"])
        grey = rgb_now is not None and is_neutral_grey(rgb_now)
        lighter = (rgb_now is not None and rgb_before is not None and luminance(rgb_now) > luminance(rgb_before))
        detail[page] = {"now": now and now["color"], "before": before and before["color"], "grey": grey,
                        "lighter": lighter}
        ok = ok and grey and lighter
    return ok, detail


def intent_p03(renderer, ws, st, prev):
    path = ws / "index.html"
    if not path.exists():
        return False, "index.html missing"
    with renderer.page(path) as page:
        _with_helpers(page)
        info = page.evaluate("""() => { const e = document.querySelector('#book-now'); if (!e) return null;
            return {text: (e.innerText || e.textContent || '').replace(/\\s+/g, ' ').trim(), visible: window.__e2shown(e)}; }""")
        if info is None:
            return False, "#book-now missing"
        if not info["visible"] or "book now" not in info["text"].lower():
            return False, info
        try:
            page.locator("#book-now").first.click(timeout=3000)
        except Exception as exc:                     # noqa: BLE001
            return False, {**info, "click": f"failed: {type(exc).__name__}"}
        page.wait_for_timeout(800)
        urls = []
        for p in page.context.pages:
            try:
                p.wait_for_load_state("load", timeout=3000)
            except Exception:                        # noqa: BLE001
                pass
            urls.append(p.url)
    nav = any(urllib.parse.urlparse(u).path.endswith("/appointments.html") for u in urls)
    return nav, {**info, "urls": urls}


def intent_p04(renderer, ws, st, prev):
    detail, ok = {}, True
    for pg in st["scope"]["pages"]:
        path = ws / pg
        if not path.exists():
            detail[pg] = "missing"
            ok = False
            continue
        with renderer.page(path) as page:
            n = page.evaluate("""() => document.querySelectorAll('a[href], button, input, select, textarea, [tabindex], summary').length""")
            seen = {}
            for _ in range(min(3 * n + 10, 300)):
                page.keyboard.press("Tab")
                r = page.evaluate("""() => { const a = document.activeElement;
                    if (!a || !a.matches('nav a')) return null;
                    if (!a.dataset.e2k) { window.__e2c = (window.__e2c || 0) + 1; a.dataset.e2k = String(window.__e2c); }
                    const s = getComputedStyle(a); return [a.dataset.e2k, a.textContent.trim().slice(0, 40), s.outlineStyle]; }""")
                if r:
                    seen[r[0]] = (r[1], r[2])
        styles = [{"link": t, "outline_style": s} for t, s in seen.values()]
        page_ok = bool(styles) and all(s["outline_style"] != "auto" for s in styles)
        detail[pg] = styles
        ok = ok and page_ok
    return ok, detail


def intent_p05(renderer, ws, st, prev):
    path = ws / "appointments.html"
    if not path.exists():
        return False, "appointments.html missing"
    submit_text = (st.get("anchors") or {}).get("submit", "Request appointment")
    with renderer.page(path) as page:
        _with_helpers(page)
        snap_js = """() => { const f = document.querySelector('#appointment-form'); if (!f) return null;
            const fields = Array.from(f.querySelectorAll('input:not([type=hidden]):not([type=submit]):not([type=button]), select, textarea'));
            return {text: window.__e2visibleText(document.body, null),
                    borders: fields.map(x => { const s = getComputedStyle(x); return [s.borderTopColor, s.borderBottomColor, s.outlineColor, s.boxShadow].join(' | '); })}; }"""
        before = page.evaluate(snap_js)
        if before is None:
            return False, "#appointment-form missing"
        clicked = page.evaluate("""(t) => { const f = document.querySelector('#appointment-form');
            const c = Array.from(f.querySelectorAll('button, input[type=submit]')).find(b =>
                ((b.innerText || b.value || '') + '').toLowerCase().includes(t.toLowerCase()))
              || f.querySelector('button:not([type=button]):not([type=reset]), input[type=submit]');
            if (!c) return false; c.setAttribute('data-e2-submit', '1'); return true; }""", submit_text)
        if not clicked:
            return False, "no submit button"
        try:
            page.locator("[data-e2-submit]").first.click(timeout=3000)
        except Exception as exc:                     # noqa: BLE001
            return False, f"submit click failed: {type(exc).__name__}"
        page.wait_for_timeout(800)
        try:
            _with_helpers(page)
            after = page.evaluate(snap_js)
        except Exception:                            # noqa: BLE001 - navigated away
            after = None
    if after is None:
        return False, "form gone after submit (page navigated)"
    new_text = sorted(set(after["text"]) - set(before["text"]))
    changed = any(a != b for a, b in zip(after["borders"], before["borders"]))
    red = any(_reddish(_rgb(b.split(" | ")[0])) or _reddish(_rgb(b.split(" | ")[1])) for b in after["borders"])
    ok = (changed or red) and not new_text
    return ok, {"new_visible_text": new_text[:10], "border_changed": changed, "border_red": red}


def _reddish(rgb) -> bool:
    return rgb is not None and rgb[0] >= 120 and rgb[0] - max(rgb[1], rgb[2]) >= 60


def zoom_disabled(content: str | None) -> bool:
    if not content:
        return False
    kv = {}
    for part in re.split(r"[,;]", content):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip().lower()] = v.strip().lower()
    if kv.get("user-scalable") in ("no", "0"):
        return True
    try:
        return float(kv.get("maximum-scale", "10")) <= 1.0
    except ValueError:
        return False


def intent_p06(renderer, ws, st, prev):
    detail, ok = {}, True
    for pg in st["scope"]["pages"]:
        path = ws / pg
        if not path.exists():
            detail[pg] = None
            ok = False
            continue
        with renderer.page(path) as page:
            content = page.evaluate("() => { const m = document.querySelector('meta[name=viewport]'); return m ? m.getAttribute('content') : null; }")
        detail[pg] = content
        ok = ok and zoom_disabled(content)
    return ok, detail


INTENT = {"P01": intent_p01, "P02": intent_p02, "P03": intent_p03, "P04": intent_p04, "P05": intent_p05,
          "P06": intent_p06}
