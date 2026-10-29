"""E1 repair/audit corpus (SPEC 5.2; tasks/e1_repair_corpus.yaml).

    python -m harness.e1_corpus targets    # derive mutation targets from the base pages
    python -m harness.e1_corpus validate   # base pages: axe-clean, generic probes, resolved checks
    python -m harness.e1_corpus build      # 48 variants with ground truth; every fault detected

Operators mutate the page source (BeautifulSoup, html.parser). Values that depend on the
rendered page (background colour for M03, the h2's computed style for M08) are measured
in Chromium when targets are derived and stored in data/corpus/targets.json.
"""
from __future__ import annotations

import copy
import json
import random
import sys

from bs4 import BeautifulSoup

from . import axe_runner, config, probes
from .render import Renderer

CORPUS = config.DATA / "corpus"
BASE = CORPUS / "base"
VARIANTS = CORPUS / "variants"
SPEC = config.task_file("e1_repair_corpus.yaml")
OPERATORS = {o["id"]: o for o in SPEC["operators"]}
PROVENANCE = {b["id"]: b["provenance"] for b in SPEC["base_pages"]}
SEED = SPEC["variants"]["seed"]
N_VARIANTS = SPEC["variants"]["per_base_page"]
K = SPEC["variants"]["faults_per_variant"]
PAGE_IDS = [b["id"] for b in SPEC["base_pages"]]
M10_RULE = ("a:focus, a:focus-visible, button:focus, button:focus-visible, input:focus, input:focus-visible, "
            "select:focus, select:focus-visible, textarea:focus, textarea:focus-visible { outline: none; }")
M11_CHOICES = {"role": ("aria-roles", 'role="buton"'), "expanded": ("aria-valid-attr-value", 'aria-expanded="yes"'),
               "checked": ("aria-allowed-attr", 'aria-checked="true" on a plain <button>')}

# --------------------------------------------------------------------------- shared JS

JS_HELPERS = r"""
window.__c = (() => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null;
    const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(Number); return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1}; };
  const lum = c => { const f = v => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const blend = (fg, bg) => ({r: fg.r * fg.a + bg.r * (1 - fg.a), g: fg.g * fg.a + bg.g * (1 - fg.a), b: fg.b * fg.a + bg.b * (1 - fg.a), a: 1});
  // Effective background: nearest opaque background colour; null if a background image is in the way.
  function background(el) {
    let layers = [];
    for (let e = el; e; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.backgroundImage && cs.backgroundImage !== 'none') return null;
      const c = parse(cs.backgroundColor);
      if (c && c.a > 0) { layers.push(c); if (c.a >= 1) break; }
    }
    let bg = {r: 255, g: 255, b: 255, a: 1};
    for (let i = layers.length - 1; i >= 0; i--) bg = blend(layers[i], bg);
    return bg;
  }
  function contrast(el) {
    const bg = background(el); if (!bg) return null;
    const fg = blend(parse(getComputedStyle(el).color), bg);
    const a = lum(fg), b = lum(bg);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  }
  function visible(e) {
    if (!e || e.closest('[hidden], .visually-hidden, .skip')) return false;
    const r = e.getBoundingClientRect(); const cs = getComputedStyle(e);
    return r.width > 2 && r.height > 2 && cs.visibility !== 'hidden' && cs.display !== 'none';
  }
  function iconOnly(e) {
    const clone = e.cloneNode(true);
    clone.querySelectorAll('.visually-hidden, .vh, .sr-only, .hidden').forEach(x => x.remove());
    return !clone.textContent.trim() && !e.querySelector('img[alt]:not([alt=""])');
  }
  function outline() {
    return Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6,[role=heading]')).map(h => ({
      level: h.getAttribute('role') === 'heading' ? parseInt(h.getAttribute('aria-level') || '2', 10) : parseInt(h.tagName[1], 10),
      text: h.textContent.replace(/\s+/g, ' ').trim()}));
  }
  return {parse, lum, background, contrast, visible, iconOnly, outline};
})();
"""

JS_CANDIDATES = r"""
() => {
  const C = window.__c, vis = C.visible;
  const textInput = i => ['text', 'search', 'email', 'tel', 'password', 'url', ''].includes((i.getAttribute('type') || '').toLowerCase());
  const named = e => (e.getAttribute('aria-label') || '').trim() || e.querySelector('.visually-hidden, .vh, .sr-only');
  const out = {};
  out.M01 = [...document.querySelectorAll('img[id]')].filter(i => (i.getAttribute('alt') || '').trim() && vis(i)).map(i => i.id);
  out.M02 = [...document.querySelectorAll('input[id]')].filter(i => textInput(i) && vis(i) &&
      document.querySelector(`label[for="${i.id}"]`)).map(i => i.id);
  out.M03 = [...document.querySelectorAll('p[id]')].filter(p => {
      if (!vis(p) || p.innerText.trim().length < 40 || p.closest('footer, #footer, header, #header')) return false;
      const cs = getComputedStyle(p), size = parseFloat(cs.fontSize), bold = parseInt(cs.fontWeight, 10) >= 700;
      if (size >= 24 || (bold && size >= 18.66)) return false;
      const c = C.contrast(p); return c !== null && c >= 4.5; }).map(p => p.id);
  out.M04 = [...document.querySelectorAll('button[id]')].filter(b => vis(b) && C.iconOnly(b) && named(b)).map(b => b.id);
  out.M05 = [...document.querySelectorAll('a[id][href]')].filter(a => vis(a) && C.iconOnly(a) && named(a)).map(a => a.id);
  out.M06 = document.documentElement.getAttribute('lang') ? ['html'] : [];
  out.M07 = [...document.querySelectorAll('button[id][aria-controls]')].filter(b => vis(b) && document.getElementById(b.getAttribute('aria-controls')))
      .map(b => b.id + '>' + b.getAttribute('aria-controls'));
  out.M08 = [...document.querySelectorAll('h2[id]')].filter(vis).map(h => h.id);
  out.M09 = [...document.querySelectorAll('footer a[id][href], #footer a[id][href], [role=contentinfo] a[id][href]')].filter(vis).map(a => a.id);
  out.M10 = ['stylesheet'];
  out.M11 = [...document.querySelectorAll('button[id], input[id][type=submit], input[id][type=button], [role=switch][id]')].filter(vis).map(b => b.id);
  out.M12 = document.title.trim() ? ['title'] : [];
  const meta = {M03: {}, M08: {}};
  for (const id of out.M03) { const bg = C.background(document.getElementById(id)); meta.M03[id] = {bg: [bg.r, bg.g, bg.b]}; }
  const props = ['font-size', 'font-weight', 'font-family', 'line-height', 'margin-top', 'margin-bottom', 'color',
                 'letter-spacing', 'text-transform', 'padding-top', 'padding-bottom', 'border-bottom', 'background-color'];
  for (const id of out.M08) { const cs = getComputedStyle(document.getElementById(id));
      meta.M08[id] = props.map(p => `${p}: ${cs.getPropertyValue(p)}`).join('; ') + '; display: block'; }
  return {ops: out, meta};
}
"""


def inject(page):
    if not page.evaluate("() => !!window.__c"):
        page.add_script_tag(content=JS_HELPERS)


# --------------------------------------------------------------------------- colours (M03)

def _lin(v: float) -> float:
    v /= 255
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _lum(rgb) -> float:
    r, g, b = (_lin(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b) -> float:
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def grey_with_contrast(bg, target: float = 2.5) -> str:
    """A neutral grey whose contrast against `bg` is `target` (darker than bg when possible)."""
    lb = _lum(bg)
    darker = (lb + 0.05) / target - 0.05 >= 0
    best = None
    for v in range(256):
        c = contrast_ratio((v, v, v), bg)
        if (_lum((v, v, v)) < lb) == darker and (best is None or abs(c - target) < abs(best[1] - target)):
            best = (v, c)
    v, c = best
    assert abs(c - target) <= 0.1, (bg, c)
    return f"#{v:02x}{v:02x}{v:02x}"


# --------------------------------------------------------------------------- operators

def _remove_name_sources(el):
    for a in ("aria-label", "aria-labelledby", "title"):
        if el.has_attr(a):
            del el[a]
    for hidden in el.select(".visually-hidden, .vh, .sr-only, .hidden"):
        hidden.decompose()


def mutate(soup: BeautifulSoup, op: str, target: str, meta: dict, rng: random.Random) -> dict:
    """Apply one operator in place. Returns the ground-truth detail for the fault."""
    if op == "M01":
        el = soup.find(id=target)
        detail = {"src": el["src"]}
        del el["alt"]
        return detail
    if op == "M02":
        inp = soup.find(id=target)
        label = soup.find("label", attrs={"for": target})
        text = " ".join(s.strip() for s in label.find_all(string=True) if s.strip())
        if label.find(id=target) is not None:          # label wraps the input: keep the input
            label.replace_with(inp.extract())
        else:
            label.decompose()
        inp["placeholder"] = text.rstrip(":").strip()
        return {"label_text": text, "name": inp.get("name")}
    if op == "M03":
        el = soup.find(id=target)
        colour = grey_with_contrast(meta["M03"][target]["bg"])
        el["style"] = (el.get("style", "").rstrip("; ") + "; " if el.get("style") else "") + f"color: {colour}"
        return {"color": colour, "text": el.get_text(" ", strip=True)[:200]}
    if op in ("M04", "M05"):
        el = soup.find(id=target)
        _remove_name_sources(el)
        return {"href": el.get("href")} if op == "M05" else {}
    if op == "M06":
        for a in ("lang", "xml:lang"):
            if soup.html.has_attr(a):
                del soup.html[a]
        return {}
    if op == "M07":
        button_id, panel_id = target.split(">")
        btn = soup.find(id=button_id)
        div = soup.new_tag("div", id=button_id)
        if btn.get("class"):
            div["class"] = btn["class"]
        for child in list(btn.contents):
            div.append(child.extract())
        btn.replace_with(div)
        return {"panel": panel_id}
    if op == "M08":
        h = soup.find(id=target)
        h.name = "div"
        for a in list(h.attrs):
            if a not in ("id", "class"):
                del h[a]
        h["style"] = meta["M08"][target]
        return {"text": h.get_text(" ", strip=True)}
    if op == "M09":
        soup.find(id=target)["tabindex"] = "3"
        return {}
    if op == "M10":
        style = soup.find_all("style")[-1]
        style.string = (style.string or "") + "\n" + M10_RULE + "\n"
        return {}
    if op == "M11":
        el = soup.find(id=target)
        choices = ["role", "expanded"] + (["checked"] if el.name == "button" and not el.get("role") else [])
        choice = rng.choice(choices)
        if choice == "role":
            el["role"] = "buton"
        elif choice == "expanded":
            el["aria-expanded"] = "yes"
        else:
            el["aria-checked"] = "true"
        return {"choice": choice, "rule": M11_CHOICES[choice][0]}
    if op == "M12":
        soup.title.decompose()
        return {}
    raise ValueError(op)


# --------------------------------------------------------------------------- checks

def _axe_hit(ax: dict, rule: str, target_id: str | None) -> bool:
    for v in ax["violations"]:
        if v["id"] != rule:
            continue
        if target_id is None:
            return True
        if any(f"#{target_id}" in n["target"].split(" ")[-1] or n["target"] == f"#{target_id}" for n in v["nodes"]):
            return True
    return False


def _keyboard_toggle(renderer, path, button_id: str, panel_id: str) -> dict:
    """Reach the element with Tab; role exposed; Enter and Space each toggle the panel."""
    ctx = probes.Ctx(lambda: renderer.page(path))
    try:
        el = ctx.js("(i) => { const e = document.getElementById(i); return e ? window.__p.id(e) : null; }", button_id)
        if el is None:
            return {"exists": False, "reachable": False, "role": None, "enter": False, "space": False}
        role = (ctx.ax(el) or {}).get("role")
        panel_vis = lambda: ctx.js("(i) => { const p = document.getElementById(i); return !!p && !p.hidden && p.getClientRects().length > 0; }", panel_id)
        reachable = ctx.reach(el, ctx.tab_budget())
        enter = space = False
        if reachable:
            v0 = panel_vis()
            ctx.key("Enter", 300)
            v1 = panel_vis()
            ctx.key(" ", 300)
            v2 = panel_vis()
            enter, space = v1 != v0, v2 != v1
        return {"exists": True, "reachable": reachable, "role": role, "enter": enter, "space": space}
    finally:
        ctx.close()


def page_facts(renderer, path) -> dict:
    """Everything the detect/resolved checks need from one render of a page."""
    with renderer.page(path) as page:
        ax = axe_runner.run_axe(page)
        inject(page)
        facts = page.evaluate("""() => ({
            outline: window.__c.outline(),
            positiveTabindex: Array.from(document.querySelectorAll('[tabindex]')).filter(e => parseInt(e.getAttribute('tabindex'), 10) > 0).length,
            title: document.title.trim(), lang: document.documentElement.getAttribute('lang') })""")
    facts["axe"] = ax
    return facts


def _outline_ok(outline: list[dict]) -> bool:
    prev = 0
    for h in outline:
        if prev and h["level"] > prev + 1:
            return False
        prev = h["level"]
    return True


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def detect(renderer, path, fault: dict, facts: dict, base_outline: list[dict]) -> bool:
    op, t, d = fault["operator"], fault["target_id"], fault.get("detail", {})
    ax = facts["axe"]
    if op == "M01":
        return _axe_hit(ax, "image-alt", t)
    if op == "M02":
        with renderer.page(path) as page:
            return bool(page.evaluate("""(i) => { const f = document.getElementById(i); if (!f) return false;
                const lab = f.labels && Array.from(f.labels).some(l => l.textContent.trim());
                const al = (f.getAttribute('aria-label') || '').trim();
                const lb = (f.getAttribute('aria-labelledby') || '').split(/\\s+/).some(x => document.getElementById(x)?.textContent.trim());
                return !(lab || al || lb); }""", t))
    if op == "M03":
        return _axe_hit(ax, "color-contrast", t)
    if op == "M04":
        return _axe_hit(ax, "button-name", t)
    if op == "M05":
        return _axe_hit(ax, "link-name", t)
    if op == "M06":
        return _axe_hit(ax, "html-has-lang", None)
    if op == "M07":
        b, p = t.split(">")
        k = _keyboard_toggle(renderer, path, b, p)
        return not (k["reachable"] and k["role"] == "button" and k["enter"])
    if op == "M08":
        return [(h["level"], _norm(h["text"])) for h in facts["outline"]] != \
               [(h["level"], _norm(h["text"])) for h in base_outline]
    if op == "M09":
        return facts["positiveTabindex"] > 0
    if op == "M10":
        r = probes.run_probe("focus_visible", lambda: renderer.page(path))
        return r["applicable"] and not r["passed"]
    if op == "M11":
        return _axe_hit(ax, d["rule"], t)
    if op == "M12":
        return _axe_hit(ax, "document-title", None)
    raise ValueError(op)


def resolved(renderer, path, fault: dict, facts: dict) -> bool:
    """The operator's "resolved" check on a (repaired) page. Targets are located by id,
    then by the operator's fallback locator (img src, input name, paragraph text, link href)."""
    op, t, d = fault["operator"], fault["target_id"], fault.get("detail", {})
    ax = facts["axe"]
    if op in ("M01", "M02", "M03", "M04", "M05"):
        ctx = probes.Ctx(lambda: renderer.page(path))
        try:
            inject(ctx.page)
            if op == "M01":
                ids = ctx.js("(src) => Array.from(document.querySelectorAll('img')).filter(i => i.getAttribute('src') === src).map(i => window.__p.id(i))", d["src"])
                return any(((ctx.ax(i) or {}).get("name") or "").strip() for i in ids)
            if op == "M02":
                return bool(ctx.js("""([i, name]) => { let f = document.getElementById(i);
                    if (!f && name) f = document.querySelector(`input[name="${name}"]`); if (!f) return false;
                    const lab = f.labels && Array.from(f.labels).some(l => l.textContent.trim());
                    const al = (f.getAttribute('aria-label') || '').trim();
                    const lb = (f.getAttribute('aria-labelledby') || '').split(/\\s+/).some(x => document.getElementById(x)?.textContent.trim());
                    return !!(lab || al || lb); }""", [t, d.get("name")]))
            if op == "M03":
                c = ctx.js("""([i, text]) => { let e = document.getElementById(i);
                    const norm = s => s.replace(/\\s+/g, ' ').trim();
                    if (!e) e = Array.from(document.querySelectorAll('p, div, span, li')).find(x => norm(x.innerText).startsWith(norm(text).slice(0, 60)));
                    return e ? window.__c.contrast(e) : null; }""", [t, d.get("text", "")])
                return c is not None and c >= 4.5
            el = ctx.js("(i) => { const e = document.getElementById(i); return e ? window.__p.id(e) : null; }", t)
            if el is None and op == "M05" and d.get("href"):
                el = ctx.js("(h) => { const e = Array.from(document.querySelectorAll('a')).find(a => a.getAttribute('href') === h); return e ? window.__p.id(e) : null; }", d["href"])
            return el is not None and bool(((ctx.ax(el) or {}).get("name") or "").strip())
        finally:
            ctx.close()
    if op == "M06":
        return bool(facts["lang"]) and not _axe_hit(ax, "html-has-lang", None) and not _axe_hit(ax, "html-lang-valid", None)
    if op == "M07":
        b, p = t.split(">")
        k = _keyboard_toggle(renderer, path, b, p)
        return k["reachable"] and k["role"] == "button" and k["enter"] and k["space"]
    if op == "M08":
        text = _norm(d["text"])
        in_heading = any(_norm(h["text"]) == text for h in facts["outline"])
        return in_heading and _outline_ok(facts["outline"])
    if op == "M09":
        return facts["positiveTabindex"] == 0
    if op == "M10":
        r = probes.run_probe("focus_visible", lambda: renderer.page(path))
        return bool(r["applicable"] and r["passed"])
    if op == "M11":
        with renderer.page(path) as page:
            exists = page.evaluate("(i) => !!document.getElementById(i)", t)
        return exists and not _axe_hit(ax, d["rule"], t)
    if op == "M12":
        return bool(facts["title"])
    raise ValueError(op)


# --------------------------------------------------------------------------- targets

def derive_targets(renderer) -> dict:
    out = {}
    for pid in PAGE_IDS:
        path = BASE / f"{pid}.html"
        with renderer.page(path) as page:
            inject(page)
            res = page.evaluate(JS_CANDIDATES)
        ops, meta = res["ops"], res["meta"]
        # M07: keep only buttons whose click really toggles the panel
        ok7 = []
        for t in ops["M07"]:
            b, p = t.split(">")
            with renderer.page(path) as page:
                v0 = page.evaluate("(i) => !document.getElementById(i).hidden", p)
                page.click(f"#{b}")
                page.wait_for_timeout(200)
                if page.evaluate("(i) => !document.getElementById(i).hidden", p) != v0:
                    ok7.append(t)
        ops["M07"] = ok7
        # M10: available only if removing outlines hides focus on at least half of the
        # determinate sampled elements. Where it hides focus on only a few (W3C pages: focus is
        # shown by a background change, so only image-only links lose their indicator),
        # detection would depend on which elements focus_visible happens to sample, and
        # operators that change the tab order (M07, M09) shift that sample. Native date/time
        # inputs keep Chrome's segment highlight without an outline, so "all" would be too strict.
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        mutate(soup, "M10", "stylesheet", meta, random.Random(0))
        tmp = CORPUS / "tmp" / f"{pid}-M10.html"
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_text(str(soup), encoding="utf-8")
        _link_images(tmp.parent)
        r = probes.run_probe("focus_visible", lambda: renderer.page(tmp))
        sampled = r["detail"].get("sampled", []) if isinstance(r.get("detail"), dict) else []
        hidden = sum(not s["passed"] for s in sampled)
        robust = bool(r["applicable"] and sampled and 2 * hidden >= len(sampled))
        ops["M10"] = ["stylesheet"] if robust else []
        meta["M10"] = {"sampled": len(sampled), "hidden": hidden}
        out[pid] = {"provenance": PROVENANCE[pid], "ops": {k: v for k, v in sorted(ops.items()) if v}, "meta": meta}
    return out


def _link_images(folder):
    link = folder / "images"
    if not link.exists():
        link.symlink_to(BASE / "images", target_is_directory=True)


def load_targets() -> dict:
    return json.loads((CORPUS / "targets.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- variants

def op_counts(targets: dict, rng: random.Random) -> dict[str, dict[str, int]]:
    """Per page, how many of its 16 fault slots each operator fills, so that every operator
    is used 16 times in total (task file: 16 +/- 2). Pages spread slots as evenly as possible
    over the operators they support (each at least once); then slots move between operators
    within pages, never above 4 per operator per page (once per variant), until every total
    is 16.

    A move takes one slot of page p from operator a to operator b. One move rarely suffices
    when an operator exists on only some pages (M04, M07 and M10 are authored-only), so each
    step is an augmenting path found by breadth-first search over operators: from an
    over-used operator through intermediate operators (whose totals do not change) to an
    under-used one. As in a flow problem, such a path exists while totals are unbalanced and
    a balanced assignment exists. Ties are broken with the seeded rng."""
    slots = N_VARIANTS * K
    goal = len(PAGE_IDS) * slots // len(OPERATORS)
    counts: dict[str, dict[str, int]] = {}
    for p in PAGE_IDS:
        avail = sorted(targets[p]["ops"])
        if len(avail) < 8:
            raise ValueError(f"{p}: only {len(avail)} operators have targets; need at least 8")
        base, extra = divmod(slots, len(avail))
        bonus = set(rng.sample(avail, extra))
        counts[p] = {op: base + (op in bonus) for op in avail}

    def total(op: str) -> int:
        return sum(c.get(op, 0) for c in counts.values())

    def can_move(p: str, a: str, b: str) -> bool:
        return counts[p].get(a, 0) > 1 and b in counts[p] and counts[p][b] < N_VARIANTS

    for _ in range(len(OPERATORS) * slots * len(PAGE_IDS)):
        over = sorted(op for op in OPERATORS if total(op) > goal)
        if not over:
            break
        under = {op for op in OPERATORS if total(op) < goal}
        start = rng.choice(over)
        prev: dict[str, tuple[str, str] | None] = {start: None}
        frontier, found = [start], None
        while frontier and found is None:
            nxt = []
            for a in frontier:
                edges = [(p, b) for p in PAGE_IDS for b in sorted(counts[p]) if can_move(p, a, b)]
                rng.shuffle(edges)
                for p, b in edges:
                    if b in prev:
                        continue
                    prev[b] = (a, p)
                    if b in under:
                        found = b
                        break
                    nxt.append(b)
                if found is not None:
                    break
            frontier = nxt
        if found is None:
            raise RuntimeError(f"cannot balance operators: over {over}, under {sorted(under)}")
        b = found
        while prev[b] is not None:            # apply the path from its end back to `start`
            a, p = prev[b]
            counts[p][a] -= 1
            counts[p][b] += 1
            b = a
    assert all(total(op) == goal for op in OPERATORS), {op: total(op) for op in OPERATORS}
    assert all(sum(c.values()) == slots for c in counts.values())
    assert all(1 <= n <= N_VARIANTS for c in counts.values() for n in c.values())
    return counts


def deal(counts: dict[str, int], rng: random.Random) -> list[list[str]]:
    """Split a page's operator multiset into 4 variants of 4 distinct operators."""
    ops = list(counts)
    rng.shuffle(ops)
    seq = [op for op in ops for _ in range(counts[op])]
    variants = [[] for _ in range(N_VARIANTS)]
    for i, op in enumerate(seq):
        variants[i % N_VARIANTS].append(op)
    assert all(len(set(v)) == K for v in variants), variants
    return [sorted(v) for v in variants]


def _element_ids(op: str, target: str) -> set[str]:
    if op == "M07":
        return set(target.split(">"))
    return {target}


def build(renderer, max_tries: int = 25) -> dict:
    targets = load_targets()
    rng = random.Random(SEED)
    counts = op_counts(targets, rng)
    VARIANTS.mkdir(parents=True, exist_ok=True)
    _link_images(VARIANTS)
    truth = {}
    base_outlines = {p: page_facts(renderer, BASE / f"{p}.html")["outline"] for p in PAGE_IDS}
    for pid in PAGE_IDS:
        src = (BASE / f"{pid}.html").read_text(encoding="utf-8")
        for vi, ops in enumerate(deal(counts[pid], rng), start=1):
            vid = f"{pid}-V{vi}"
            for attempt in range(max_tries):
                used: set[str] = set()
                faults = []
                for op in ops:
                    cands = [t for t in targets[pid]["ops"][op] if not (_element_ids(op, t) & used)]
                    if not cands:
                        break
                    t = rng.choice(cands)
                    used |= _element_ids(op, t)
                    faults.append({"operator": op, "target_id": t, "scs": OPERATORS[op]["sc"]})
                if len(faults) != K:
                    continue
                soup = BeautifulSoup(src, "html.parser")
                mrng = random.Random(f"{SEED}-{vid}-{attempt}")
                for f in faults:
                    f["detail"] = mutate(soup, f["operator"], f["target_id"], targets[pid]["meta"], mrng)
                path = VARIANTS / f"{vid}.html"
                path.write_text(str(soup), encoding="utf-8")
                facts = page_facts(renderer, path)
                det = [detect(renderer, path, f, facts, base_outlines[pid]) for f in faults]
                if all(det):
                    truth[vid] = {"base_id": pid, "provenance": PROVENANCE[pid], "attempt": attempt, "faults": faults}
                    print(vid, [f"{f['operator']}:{f['target_id']}" for f in faults], flush=True)
                    break
                print(vid, "attempt", attempt, "undetected:",
                      [f"{f['operator']}:{f['target_id']}" for f, ok in zip(faults, det) if not ok], flush=True)
            else:
                raise RuntimeError(f"{vid}: no detectable combination after {max_tries} tries")
    (CORPUS / "ground_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


def validate(renderer) -> list[str]:
    """Base pages: (a) zero axe violations, (b) generic probes pass, (c) every target's
    resolved check passes."""
    problems = []
    targets = load_targets()
    for pid in PAGE_IDS:
        path = BASE / f"{pid}.html"
        facts = page_facts(renderer, path)
        if facts["axe"]["violations"]:
            problems.append(f"{pid}: axe violations {[v['id'] for v in facts['axe']['violations']]}")
        for n in probes.GENERIC:
            r = probes.run_probe(n, lambda: renderer.page(path))
            if r["applicable"] and not r["passed"]:
                problems.append(f"{pid}: generic probe {n} fails")
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for op, ts in targets[pid]["ops"].items():
            for t in ts:
                detail = {}
                if op == "M01":
                    detail = {"src": soup.find(id=t)["src"]}
                elif op == "M02":
                    detail = {"name": soup.find(id=t).get("name")}
                elif op == "M03":
                    detail = {"text": soup.find(id=t).get_text(" ", strip=True)[:200]}
                elif op == "M05":
                    detail = {"href": soup.find(id=t).get("href")}
                elif op == "M08":
                    detail = {"text": soup.find(id=t).get_text(" ", strip=True)}
                elif op == "M11":
                    detail = {"rule": "aria-roles"}
                if not resolved(renderer, path, {"operator": op, "target_id": t, "detail": detail}, facts):
                    problems.append(f"{pid}: resolved check fails on the base page for {op}:{t}")
        print(pid, "checked", flush=True)
    return problems


def main(argv):
    cmd = argv[0] if argv else "build"
    with Renderer(config.ROOT) as r:
        if cmd == "targets":
            t = derive_targets(r)
            (CORPUS / "targets.json").write_text(json.dumps(t, indent=1), encoding="utf-8")
            for pid, v in t.items():
                print(pid, v["provenance"], {op: len(ts) for op, ts in v["ops"].items()})
        elif cmd == "validate":
            problems = validate(r)
            print("\n".join(problems) or "all base pages valid")
            sys.exit(1 if problems else 0)
        elif cmd == "build":
            build(r)


if __name__ == "__main__":
    main(sys.argv[1:])
