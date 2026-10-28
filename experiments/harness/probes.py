"""Keyboard and semantics probes (SPEC 4).

Each probe is `probe(ctx) -> {"applicable": bool, "passed": bool, "detail": ...}` and runs on
a fresh page. Probes find widgets by the visible-text anchors in the task files, never by
implementation details, and accept any keyboard path that works (WCAG, not APG). Exposed
roles, names and states are read from Chrome's accessibility tree over CDP.

Clarifications fixed before the freeze (also noted in SPEC 4.3):
- switch_behavior also accepts a toggle button whose state is exposed with aria-pressed
  (4.1.2 requires a programmatically determinable state; the switch role is an APG choice).
- no_keyboard_trap allows up to 8 consecutive Tabs on native date/time inputs and media
  elements with controls, which move focus among their internal parts.
- form_errors_associated identifies invalid fields by any of: required/aria-required,
  aria-invalid="true", :invalid, newly shown associated error text, or a field whose
  border/outline/background changed after the empty submit.
"""
from __future__ import annotations

import io
import pathlib
import re

import numpy as np
from PIL import Image

LIB = (pathlib.Path(__file__).parent / "probe_lib.js").read_text(encoding="utf-8")
FOCUS_DIFF_THRESHOLD = 1.0      # mean |pixel diff| (0-255); calibrated in tests/test_probes.py
SAMPLE_FOCUS = 10
REPORT_ONLY = {"current_page_indicated"}        # AAA (2.4.8): reported, excluded from pass rates
GENERIC = ["tab_reachability", "focus_visible", "no_keyboard_trap"]


def res(applicable: bool, passed: bool, detail=None) -> dict:
    return {"applicable": applicable, "passed": bool(passed) if applicable else None, "detail": detail}


def na(detail) -> dict:
    return res(False, False, detail)


class Ctx:
    """A fresh page plus helpers. `reopen()` replaces the page with a new fresh one."""

    def __init__(self, page_factory, scope: str | None = None, anchors: dict | None = None):
        self._factory = page_factory          # () -> context manager yielding a Playwright page
        self.scope = scope
        self.anchors = anchors or {}
        self._cm = None
        self.page = None
        self._cdp = None
        self.reopen()

    def reopen(self):
        self.close()
        self._cm = self._factory()
        self.page = self._cm.__enter__()
        self._cdp = None
        return self

    def close(self):
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            finally:
                self._cm = None

    # -- JS bridge -------------------------------------------------------
    def _inject(self):
        if not self.page.evaluate("() => !!window.__p"):
            self.page.add_script_tag(content=LIB)

    def call(self, fn: str, *args):
        self._inject()
        return self.page.evaluate(f"(a) => window.__p.{fn}(...a)", list(args))

    def js(self, expr: str, arg=None):
        self._inject()
        return self.page.evaluate(expr, arg)

    def on(self, el_id: str, body: str, arg=None):
        """Evaluate `body` with `el` bound to the element with this id."""
        self._inject()
        return self.page.evaluate(f"([i, arg]) => {{ const el = window.__p.byId(i); {body} }}", [el_id, arg])

    def wait(self, ms: int):
        self.page.wait_for_timeout(ms)

    # -- keyboard ----------------------------------------------------------
    def key(self, k: str, wait: int = 120):
        self.page.keyboard.press(k)
        if wait:
            self.page.wait_for_timeout(wait)

    def active(self) -> str | None:
        return self.js("() => window.__p.id(window.__p.active())")

    def active_within(self, el_id: str) -> bool:
        return bool(self.on(el_id, "const a = window.__p.active(); return !!(el && a && (el === a || el.contains(a)));"))

    def to_top(self):
        """Move the sequential-focus starting point to the top of the document."""
        self.js("""() => { let s = document.querySelector('[data-se4all-sentinel]');
            if (!s) { s = document.createElement('span'); s.setAttribute('tabindex', '-1');
                      s.setAttribute('data-se4all-sentinel', ''); s.style.cssText = 'position:absolute;top:0;left:0;width:1px;height:1px;';
                      document.body.prepend(s); }
            s.focus({preventScroll: true}); }""")

    def walk(self, n: int) -> list[str | None]:
        seq = []
        for _ in range(n):
            self.page.keyboard.press("Tab")
            a = self.active()
            seq.append(None if a is None or self.js("(i) => !!window.__p.byId(i)?.hasAttribute('data-se4all-sentinel')", a) else a)
        return seq

    def reach(self, el_id: str, max_n: int) -> bool:
        """Tab from the top until focus is on (or inside) the element."""
        self.to_top()
        for _ in range(max_n):
            self.page.keyboard.press("Tab")
            if self.active_within(el_id):
                return True
        return False

    def tab_budget(self) -> int:
        return 3 * len(self.call("interactives", None)) + 10

    # -- lookup ------------------------------------------------------------
    def find(self, text: str, exact: bool = False, actionable_only: bool = False) -> str | None:
        ids = self.call("findText", text, self.scope, exact)
        if not ids and self.scope:
            ids = self.call("findText", text, None, exact)
        if actionable_only:
            ids = [i for i in ids if self.call("isActionable", i)] or ids
        if not ids:
            return None
        return self.call("actionable", ids[0])

    def visible(self, el_id: str) -> bool:
        return bool(self.call("rendered", None) if el_id is None else
                    self.on(el_id, "return window.__p.rendered(el);"))

    def seen(self, el_id: str) -> bool:
        return bool(self.on(el_id, "return window.__p.seen(el);"))

    def text_visible(self, text: str) -> bool:
        return bool(self.js("""(t) => { const ids = window.__p.findText(t, null, false);
            return ids.some(i => window.__p.seen(window.__p.byId(i))); }""", text))

    def outer(self, el_id: str, n: int = 160) -> str:
        return self.on(el_id, "return el ? el.outerHTML.slice(0, arg) : null;", n) or ""

    # -- accessibility tree --------------------------------------------------
    def ax(self, el_id: str) -> dict | None:
        if self._cdp is None:
            self._cdp = self.page.context.new_cdp_session(self.page)
        r = self._cdp.send("Runtime.evaluate", {"expression":
                           f"document.querySelector('[data-se4all-id=\"{el_id}\"]')"})
        oid = r.get("result", {}).get("objectId")
        if not oid:
            return None
        tree = self._cdp.send("Accessibility.getPartialAXTree", {"objectId": oid, "fetchRelatives": False})
        nodes = tree.get("nodes") or []
        if not nodes:
            return None
        node = nodes[-1] if len(nodes) > 1 else nodes[0]
        props = {p["name"]: p.get("value", {}).get("value") for p in node.get("properties", [])}
        return {"role": (node.get("role") or {}).get("value"), "name": (node.get("name") or {}).get("value") or "",
                "ignored": node.get("ignored", False), "value": (node.get("value") or {}).get("value"), **props}

    # -- screenshots ---------------------------------------------------------
    def shot(self, clip: dict) -> np.ndarray:
        vp = self.page.viewport_size
        x = max(0.0, clip["x"]); y = max(0.0, clip["y"])
        w = min(vp["width"] - x, clip["w"] - (x - clip["x"]))
        h = min(vp["height"] - y, clip["h"] - (y - clip["y"]))
        if w < 1 or h < 1:
            return np.zeros((1, 1, 3), dtype=np.int16)
        png = self.page.screenshot(clip={"x": x, "y": y, "width": w, "height": h}, animations="allow")
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).mean())


def _truthy(v) -> bool:
    return str(v).lower() in ("true", "mixed")


# ===================================================================== generic

def tab_reachability(ctx: Ctx) -> dict:
    items = ctx.call("interactives", ctx.scope)
    if not items:
        return na("no visible interactive elements in scope")
    ctx.call("startRecording")
    ctx.walk(3 * len(items) + 10)
    unreached, groups = [], {}
    for it in items:
        r = ctx.call("reached", it["id"])
        if it["group"]:
            groups.setdefault(it["group"], []).append((it["id"], r))
        elif not r:
            unreached.append(it["id"])
    for g, members in groups.items():
        if not any(r for _, r in members):
            unreached.append(members[0][0])
    return res(True, not unreached, {"n_interactive": len(items),
                                     "unreached": [ctx.outer(i) for i in unreached[:5]],
                                     "n_unreached": len(unreached)})


def _tab_order(ctx: Ctx) -> list[str]:
    """Distinct elements in keyboard order, from the top until focus wraps or leaves."""
    seq = ctx.walk(ctx.tab_budget())
    order = []
    for a in seq:
        if a is None:
            if order:
                break
            continue
        if order and a == order[0]:
            break
        if a not in order:
            order.append(a)
    return order


def focus_visible(ctx: Ctx) -> dict:
    """Screenshot each sampled element's region unfocused (pristine page) and focused via Tab.

    Elements are re-identified on each fresh page by their DOM path (pages load
    deterministically). The page is scrolled back to the baseline position before the
    focused screenshot; if the element moved, the comparison is skipped as indeterminate.
    """
    order = _tab_order(ctx)
    in_scope = [i for i in order if ctx.scope is None or ctx.on(
        i, "return window.__p.scopeRoots(arg).some(r => r === el || r.contains(el));", ctx.scope)]
    if not in_scope:
        return na("no keyboard-focusable elements in scope")
    if len(in_scope) > SAMPLE_FOCUS:
        idx = np.linspace(0, len(in_scope) - 1, SAMPLE_FOCUS).round().astype(int)
        sample = [in_scope[i] for i in sorted(set(idx))]
    else:
        sample = in_scope
    paths = {i: ctx.on(i, "return window.__p.path(el);") for i in order}
    results, indeterminate = [], []
    for target in sample:
        pos = order.index(target)
        ctx.reopen()
        el = ctx.call("byPath", paths[target])
        if el is None:
            indeterminate.append((paths[target], "element missing after reload"))
            continue
        region = ctx.call("focusRegion", el)
        scroll = ctx.js("() => [window.scrollX, window.scrollY]")
        rect0 = ctx.call("rect", el)
        ctx.wait(300)
        base = ctx.shot(region)
        ctx.wait(300)
        noise = mad(base, ctx.shot(region))
        if noise > FOCUS_DIFF_THRESHOLD:
            indeterminate.append((ctx.outer(el, 80), f"region changes without focus (noise {noise:.2f})"))
            continue
        prev = ctx.call("byPath", paths[order[pos - 1]]) if pos > 0 else None
        if prev is not None:
            ctx.on(prev, "el.focus({preventScroll: true});")
        else:
            ctx.to_top()
        ctx.key("Tab", 0)
        if not ctx.active_within(el):
            ctx.to_top()
            for _ in range(pos + 1):
                ctx.key("Tab", 0)
        if not ctx.active_within(el):
            indeterminate.append((ctx.outer(el, 80), "could not focus with Tab"))
            continue
        ctx.js("(s) => window.scrollTo(s[0], s[1])", scroll)
        ctx.wait(350)
        rect1 = ctx.call("rect", el)
        if abs(rect1["x"] - rect0["x"]) > 1 or abs(rect1["y"] - rect0["y"]) > 1:
            indeterminate.append((ctx.outer(el, 80), "layout moved when focused"))
            continue
        d = mad(base, ctx.shot(region))
        results.append({"el": ctx.outer(el, 100), "diff": round(d, 3), "passed": d > FOCUS_DIFF_THRESHOLD})
    if not results:
        return na({"indeterminate": indeterminate})
    return res(True, all(r["passed"] for r in results), {"sampled": results, "indeterminate": indeterminate})


NATIVE_SEGMENTED_HOLD = 8
NATIVE_SEGMENTED_JS = ("return (el.tagName === 'INPUT' && ['date', 'time', 'datetime-local', 'month', 'week']"
                       ".includes(el.type)) || (['VIDEO', 'AUDIO'].includes(el.tagName) && el.controls);")


def no_keyboard_trap(ctx: Ctx) -> dict:
    items = ctx.call("interactives", None)
    if not items:
        return na("no interactive elements")
    seq = ctx.walk(3 * len(items) + 10)
    first_i = next((k for k, a in enumerate(seq) if a is not None), None)
    if first_i is None:
        return res(True, False, "Tab never moved focus to any element")
    first = seq[first_i]
    run, held = 1, None
    for k in range(first_i + 1, len(seq)):
        if seq[k] is not None and seq[k] == seq[k - 1]:
            run += 1
            # Native date/time inputs and media controls move focus among their internal
            # parts while document.activeElement stays on the element: allow more presses.
            limit = NATIVE_SEGMENTED_HOLD if ctx.on(seq[k], NATIVE_SEGMENTED_JS) else 3
            if run > limit and not ctx.on(seq[k], "return window.__p.inModal(el);"):
                held = seq[k]
                break
        else:
            run = 1
    rest = seq[first_i + 1:]
    wrapped = (None in rest) or (first in rest)
    detail = {"presses": len(seq), "distinct": len({a for a in seq if a})}
    if held:
        return res(True, False, {**detail, "held_by": ctx.outer(held)})
    if not wrapped:
        return res(True, False, {**detail, "problem": "focus never left or wrapped"})
    # Shift+Tab moves backward
    ctx.reopen()
    a = ctx.walk(1)[0]
    b = None
    for _ in range(5):
        b = ctx.walk(1)[0]
        if b is not None and b != a:
            break
    if a is None or b is None or b == a:
        return res(True, True, {**detail, "shift_tab": "not testable (one focusable element)"})
    ctx.page.keyboard.press("Shift+Tab")
    back = ctx.active()
    # The element before b need not be a (b may be reached after several presses); it must differ from b.
    ok = back is not None and back != b
    return res(True, ok, {**detail, "shift_tab_moved_back": ok})


# ===================================================================== patterns

def form_errors_associated(ctx: Ctx, color_only_check: bool = False) -> dict:
    submit_text = ctx.anchors.get("submit")
    submit = ctx.find(submit_text, actionable_only=True) if submit_text else None
    if submit is None:
        return res(True, False, f"submit control {submit_text!r} not found")
    fields = ctx.call("formFields", ctx.scope)
    if not fields:
        return res(True, False, "no form fields in scope")
    before = {f: ctx.call("fieldState", f) for f in fields}
    text_before = ctx.call("visibleText")
    url_before = ctx.page.url
    ctx.on(submit, "el.focus();")
    if not ctx.active_within(submit):
        return res(True, False, "submit control cannot receive keyboard focus")
    ctx.key("Enter", 800)
    if ctx.page.url.split("#")[0] != url_before.split("#")[0]:
        return res(True, False, "empty form was submitted (page navigated)")
    try:
        text_after = ctx.call("visibleText")
    except Exception:                       # noqa: BLE001 - page reloaded on submit
        return res(True, False, "empty form was submitted (page reloaded)")
    new_text = text_after != text_before
    flagged = []
    for f in fields:
        st = ctx.call("fieldState", f)
        if st is None:
            continue
        signals = []
        if st["required"]:
            signals.append("required")
        if st["invalidAttr"] == "true":
            signals.append("aria-invalid")
        if st["cssInvalid"]:
            signals.append(":invalid")
        if st["border"] != before[f]["border"]:
            signals.append("style-changed")
        assoc = ctx.call("associatedText", f)
        assoc_err = [a for a in assoc if a["seen"] and a["text"] and a["text"] not in text_before.lower()
                     and _looks_like_error(a["text"])]
        if assoc_err:
            signals.append("associated-error-text")
        if signals:
            flagged.append((f, st, assoc_err, signals))
    if not flagged:
        return res(True, False, "no field was identified as invalid after an empty submit")
    problems = []
    for f, st, assoc_err, signals in flagged:
        # Only fields that are actually empty/invalid: required ones are, after an empty submit.
        if color_only_check:
            near = ctx.call("nearbyText", f)
            near_new = [n for n in near if (n["text"] and n["text"] not in text_before.lower()) or
                        (n["tag"] in ("IMG", "svg", "SVG") and n["alt"])]
            if not assoc_err and not near_new:
                problems.append({"field": ctx.outer(f, 120), "problem": "error shown by colour only"})
        else:
            ok_assoc = st["invalidAttr"] == "true" or bool(assoc_err)
            if not ok_assoc:
                problems.append({"field": ctx.outer(f, 120), "problem": "no aria-invalid and no associated error text"})
            elif not (assoc_err or new_text):
                problems.append({"field": ctx.outer(f, 120), "problem": "no visible error text"})
    return res(True, not problems, {"flagged": len(flagged), "problems": problems[:5]})


_ERR_RE = re.compile(r"required|enter|invalid|please|must|missing|error|select|choose|agree|provide|valid|cannot|can't|needed|wrong|check", re.I)


def _looks_like_error(text: str) -> bool:
    return bool(_ERR_RE.search(text))


def error_not_color_only(ctx: Ctx) -> dict:
    return form_errors_associated(ctx, color_only_check=True)


def dialog_behavior(ctx: Ctx) -> dict:
    a = ctx.anchors
    if "trigger" in a:
        trig = ctx.find(a["trigger"], actionable_only=True)
    else:
        sel = a.get("trigger_selector", "img")
        trig = ctx.js("""(sel) => { const el = Array.from(document.querySelectorAll(sel)).find(e => window.__p.rendered(e));
            if (!el) return null; const c = el.closest('a[href],button,[role=button],[tabindex]');
            return window.__p.id(c || el); }""", sel)
    if trig is None:
        return res(True, False, "trigger not found")
    if not ctx.reach(trig, ctx.tab_budget()):
        return res(True, False, "trigger not reachable with Tab")
    trig_focus = ctx.active()
    ctx.key("Enter", 600)
    dlg = ctx.js("() => window.__p.id(window.__p.openDialog())")
    if dlg is None:
        return res(True, False, "Enter on the trigger did not open a role=dialog/alertdialog or <dialog>")
    info = ctx.ax(dlg) or {}
    named = bool((info.get("name") or "").strip())
    focus_in = ctx.active_within(dlg)
    n_focusable = ctx.on(dlg, "return el.querySelectorAll('a[href],button,input,select,textarea,[tabindex]').length;") or 1
    contained = True
    for _ in range(n_focusable + 2):
        ctx.key("Tab", 80)
        if not ctx.active_within(dlg):
            contained = False
            break
    ctx.key("Escape", 500)
    closed = not ctx.on(dlg, "return !!el && (el.tagName !== 'DIALOG' || el.open) && window.__p.rendered(el);")
    returned = closed and (ctx.active() == trig_focus or ctx.active_within(trig))
    detail = {"named": named, "focus_moved_in": focus_in, "tab_contained": contained,
              "escape_closes": closed, "focus_returned": returned}
    return res(True, all(detail.values()), detail)


def live_region_on_action(ctx: Ctx) -> dict:
    a = ctx.anchors
    ctx.call("markLiveRegions")
    for key, value in (a.get("fill") or {}).items():
        f = ctx.call("fieldFor", key, ctx.scope)
        if f is None:
            return res(True, False, f"field {key!r} not found")
        ctx.on(f, "el.focus();")
        ctx.page.keyboard.type(str(value))
    before = ctx.call("visibleText")
    trig = ctx.find(a["trigger"], actionable_only=True) if a.get("trigger") else None
    if trig is None:
        return res(True, False, "trigger not found")
    ctx.on(trig, "el.focus();")
    if not ctx.active_within(trig):
        return res(True, False, "trigger cannot receive keyboard focus")
    ctx.key("Enter", 700)
    if a.get("confirm"):
        conf = ctx.find(a["confirm"], actionable_only=True)
        if conf is None or not ctx.visible(conf):
            return res(True, False, "confirm control not shown")
        ctx.on(conf, "el.focus();")
        ctx.key("Enter", 900)
    if a.get("result_text"):
        ids = ctx.call("findText", a["result_text"], None, False)
        msg = next((i for i in ids if ctx.seen(i)), None)
        if msg is None:
            return res(True, False, "result message not shown")
    else:
        msg = ctx.js("""(before) => { const B = new Set(before.split(/(?<=[.!?])\\s+/));
            const cands = Array.from(document.querySelectorAll('body *')).filter(e => window.__p.seen(e) &&
                e.children.length === 0 && e.innerText && !before.includes(window.__p.norm(e.innerText)));
            return cands.length ? window.__p.id(cands[0]) : null; }""", before)
        if msg is None:
            return res(True, False, "no new message text appeared")
    region_pre = ctx.on(msg, "const r = window.__p.liveRegionOf(el); return r ? r.hasAttribute('data-se4all-prelive') : null;")
    focus_moved = ctx.active_within(msg) or bool(ctx.on(msg, "const a = window.__p.active(); return !!(a && a.contains(el));"))
    ok = bool(region_pre) or focus_moved
    return res(True, ok, {"in_live_region_present_before": bool(region_pre), "focus_moved_to_message": focus_moved})


def _selected_state(info: dict | None) -> bool:
    if not info:
        return False
    return any(_truthy(info.get(k)) for k in ("selected", "checked", "pressed", "expanded")) or \
        bool(info.get("current"))


def tabs_behavior(ctx: Ctx) -> dict:
    names = ctx.anchors.get("tabs") or []
    out = []
    for name in names:
        ctx.reopen()
        t = ctx.find(name, exact=True, actionable_only=True) or ctx.find(name, actionable_only=True)
        if t is None:
            out.append({"tab": name, "ok": False, "why": "not found"})
            continue
        reached = ctx.reach(t, ctx.tab_budget())
        if not reached:
            # roving tabindex: focus may land on a sibling tab; use arrow keys from there
            ctx.to_top()
            for _ in range(ctx.tab_budget()):
                ctx.page.keyboard.press("Tab")
                a = ctx.active()
                if a and ctx.on(a, "const c = el.closest('[role=tablist]'); return !!c && c.contains(window.__p.byId(arg));", t):
                    break
            for key in ["ArrowRight"] * 4 + ["ArrowLeft"] * 8 + ["ArrowDown"] * 4:
                if ctx.active_within(t):
                    reached = True
                    break
                ctx.key(key, 80)
            reached = reached or ctx.active_within(t)
        if not reached:
            out.append({"tab": name, "ok": False, "why": "not reachable by keyboard"})
            continue
        ctx.key("Enter", 400)
        info = ctx.ax(t)
        if not _selected_state(info):
            ctx.key(" ", 400)
            info = ctx.ax(t)
        panel = ctx.on(t, """const ctl = el.getAttribute('aria-controls');
            if (ctl && document.getElementById(ctl)) return window.__p.id(document.getElementById(ctl));
            const lab = el.id && document.querySelector(`[aria-labelledby~="${el.id}"]`);
            if (lab) return window.__p.id(lab);
            const name = window.__p.norm(el.innerText);
            const hs = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6')).filter(h => h !== el && !h.contains(el)
                && window.__p.norm(h.textContent).includes(name));
            return hs.length ? window.__p.id(hs.find(h => window.__p.rendered(h)) || hs[0]) : null;""")
        panel_visible = panel is not None and ctx.seen(panel)
        out.append({"tab": name, "ok": panel_visible and _selected_state(info),
                    "panel_visible": panel_visible, "state_exposed": _selected_state(info)})
    if not names:
        return na("no tab anchors")
    return res(True, all(o["ok"] for o in out), out)


def _toggle_state(ctx: Ctx, el_id: str):
    info = ctx.ax(el_id) or {}
    det = ctx.on(el_id, "const d = el.closest('details'); return d ? d.open : null;")
    return (info.get("expanded"), det)


def accordion_behavior(ctx: Ctx) -> dict:
    out = []
    for name in ctx.anchors.get("headers") or []:
        ctx.reopen()
        h = ctx.find(name, actionable_only=True)
        if h is None:
            out.append({"header": name, "ok": False, "why": "not found"})
            continue
        if not ctx.reach(h, ctx.tab_budget()):
            out.append({"header": name, "ok": False, "why": "not reachable with Tab"})
            continue
        foc = ctx.active()
        t0, s0 = ctx.call("visibleText"), _toggle_state(ctx, foc)
        ctx.key("Enter", 450)
        t1, s1 = ctx.call("visibleText"), _toggle_state(ctx, foc)
        ctx.key(" ", 450)
        t2, s2 = ctx.call("visibleText"), _toggle_state(ctx, foc)
        enter_toggles = t1 != t0
        space_toggles = t2 != t1
        state = (s0[0] is not None and s1[0] != s0[0] and s2[0] != s1[0]) or \
                (s0[1] is not None and s1[1] != s0[1] and s2[1] != s1[1])
        out.append({"header": name, "ok": enter_toggles and space_toggles and state,
                    "enter_toggles": enter_toggles, "space_toggles": space_toggles, "state_exposed": state})
    if not out:
        return na("no header anchors")
    return res(True, all(o["ok"] for o in out), out)


def disclosure_nav_behavior(ctx: Ctx) -> dict:
    a = ctx.anchors
    tog = ctx.find(a["disclosure"], exact=True, actionable_only=True) or ctx.find(a["disclosure"], actionable_only=True)
    if tog is None:
        return res(True, False, "disclosure control not found")
    item_ids = ctx.call("findText", a["submenu_item"], None, True) or ctx.call("findText", a["submenu_item"], None, False)
    if not item_ids:
        return res(True, False, "submenu item not found")
    item = ctx.call("actionable", item_ids[0])
    if not ctx.reach(tog, ctx.tab_budget()):
        return res(True, False, "disclosure control not reachable with Tab")
    url0 = ctx.page.url
    opened_by = None
    for k in ["Enter", " ", "ArrowDown"]:
        if ctx.seen(item):
            opened_by = opened_by or "focus"
            break
        ctx.key(k, 400)
        if ctx.page.url != url0:
            return res(True, False, f"{k!r} on the disclosure navigated away instead of opening the submenu")
        if ctx.seen(item):
            opened_by = k
            break
    if not opened_by:
        return res(True, False, "submenu did not open by keyboard")
    expanded = _truthy((ctx.ax(tog) or {}).get("expanded"))
    reached = False
    for k in ["Tab"] * 8:
        if ctx.active_within(item):
            reached = True
            break
        ctx.key(k, 80)
    if not reached:
        ctx.on(tog, "el.focus();")
        for _ in range(8):
            ctx.key("ArrowDown", 80)
            if ctx.active_within(item):
                reached = True
                break
    ctx.key("Escape", 400)
    closed = not ctx.seen(item)
    if not closed:
        ctx.on(tog, "el.focus();")
        ctx.key("Enter", 400)
        closed = not ctx.seen(item)
    collapsed = not _truthy((ctx.ax(tog) or {}).get("expanded"))
    detail = {"opened_by": opened_by, "item_reachable": reached, "closes": closed,
              "state_exposed": expanded and collapsed}
    return res(True, reached and closed and expanded and collapsed, detail)


def disclosure_panel_behavior(ctx: Ctx) -> dict:
    a = ctx.anchors
    trig = ctx.find(a["trigger"], actionable_only=True)
    if trig is None:
        return res(True, False, "trigger not found")
    if not ctx.reach(trig, ctx.tab_budget()):
        return res(True, False, "trigger not reachable with Tab")
    opened = False
    for k in ["Enter", " "]:
        ctx.key(k, 500)
        inside = ctx.call("findText", a["inside"], None, False)
        if inside and any(ctx.seen(i) for i in inside):
            opened = True
            break
    if not opened:
        return res(True, False, "panel did not open by keyboard")
    inside = ctx.call("actionable", next(i for i in ctx.call("findText", a["inside"], None, False) if ctx.seen(i)))
    reached = False
    for _ in range(15):
        ctx.key("Tab", 60)
        if ctx.active_within(inside):
            reached = True
            break
    nec = ctx.call("fieldFor", "necessary", None)
    if nec is None:
        ids = ctx.call("findText", "Necessary", None, False)
        nec = ctx.on(ids[0], "const c = el.querySelector('input,[role=checkbox],[role=switch]') || el.closest('label')?.control; return c ? window.__p.id(c) : null;") if ids else None
    info = ctx.ax(nec) if nec else None
    nec_ok = bool(info) and _truthy(info.get("checked")) and _truthy(info.get("disabled"))
    detail = {"controls_reachable": reached, "necessary_disabled_and_checked": nec_ok}
    return res(True, reached and nec_ok, detail)


def table_headers(ctx: Ctx) -> dict:
    out = ctx.js("""(sel) => {
      const roots = window.__p.scopeRoots(sel);
      const tables = [];
      roots.forEach(r => { if (r.matches('table,[role=table],[role=grid]')) tables.push(r);
                           r.querySelectorAll('table,[role=table],[role=grid]').forEach(t => tables.push(t)); });
      return tables.filter(t => window.__p.rendered(t)).map(t => {
        const rows = Array.from(t.querySelectorAll('tr,[role=row]'));
        const isHdr = c => c.tagName === 'TH' || ['columnheader', 'rowheader'].includes(c.getAttribute('role'));
        const cells = r => Array.from(r.children).filter(c => c.matches('td,th,[role=cell],[role=gridcell],[role=columnheader],[role=rowheader]'));
        const headerRow = rows.length && cells(rows[0]).length && cells(rows[0]).every(isHdr);
        const dataRows = rows.filter(r => cells(r).some(c => !isHdr(c)));
        const rowHeaders = dataRows.length && dataRows.every(r => cells(r).length && isHdr(cells(r)[0]));
        const withHeaders = dataRows.length && dataRows.every(r => cells(r).filter(c => !isHdr(c)).every(c => c.hasAttribute('headers'))) ;
        const cap = t.querySelector('caption');
        const named = (cap && window.__p.norm(cap.textContent)) || t.getAttribute('aria-label') ||
          (t.getAttribute('aria-labelledby') || '').split(/\\s+/).some(i => document.getElementById(i));
        return {rows: rows.length, headerRow: !!headerRow, rowHeaders: !!rowHeaders, headersAttr: !!withHeaders, named: !!named};
      }).filter(x => x.rows > 1);
    }""", ctx.scope)
    if not out:
        return res(True, False, "no data table in scope")
    ok = all((t["headerRow"] or t["rowHeaders"] or t["headersAttr"]) and t["named"] for t in out)
    return res(True, ok, out)


def sort_control(ctx: Ctx) -> dict:
    col = ctx.anchors.get("sort_column", "")
    hdr = ctx.js("""(name) => { const n = window.__p.norm(name);
        const hs = Array.from(document.querySelectorAll('th,[role=columnheader],tr:first-child > td'))
          .filter(h => window.__p.norm(h.textContent).includes(n));
        return hs.length ? window.__p.id(hs[0]) : null; }""", col)
    if hdr is None:
        return res(True, False, f"column header {col!r} not found")
    btn = ctx.on(hdr, """const sel = 'button,[role=button]';
        const b = el.querySelector(sel) || (el.previousElementSibling && el.previousElementSibling.matches?.(sel) ? el.previousElementSibling : null)
                 || (el.nextElementSibling && el.nextElementSibling.matches?.(sel) ? el.nextElementSibling : null);
        return b ? window.__p.id(b) : null;""")
    if btn is None:
        return res(True, False, "no button inside or next to the column header")
    order0 = _first_col(ctx)
    hdr_ax0, btn_ax0 = ctx.ax(hdr) or {}, ctx.ax(btn) or {}
    if not ctx.reach(btn, ctx.tab_budget()):
        return res(True, False, "sort control not reachable with Tab")
    ctx.key("Enter", 500)
    order1 = _first_col(ctx)
    if order1 == order0:
        ctx.key(" ", 500)
        order1 = _first_col(ctx)
    reordered = order1 != order0 and sorted(order1) == sorted(order0)
    hdr_ax1, btn_ax1 = ctx.ax(hdr) or {}, ctx.ax(btn) or {}
    aria_sort = ctx.on(hdr, "return el.getAttribute('aria-sort');")
    state = (aria_sort in ("ascending", "descending")) or (btn_ax1.get("name") != btn_ax0.get("name")) or \
        _truthy(btn_ax1.get("pressed")) != _truthy(btn_ax0.get("pressed"))
    return res(True, reordered and state, {"reordered": reordered, "state_exposed": state, "aria_sort": aria_sort})


def _first_col(ctx: Ctx) -> list[str]:
    return ctx.js("""() => { const t = document.querySelector('table,[role=table],[role=grid]'); if (!t) return [];
        return Array.from(t.querySelectorAll('tbody tr, [role=rowgroup] [role=row]')).map(r => window.__p.norm((r.children[0] || {}).textContent || ''))
          .filter(Boolean); }""")


_PAUSE_RE = re.compile(r"pause|stop|play|halt|freeze|motion|animation|rotation|autoplay|auto-play|scroll", re.I)


def pause_control(ctx: Ctx) -> dict:
    region = ctx.js("""(sel) => { const r = window.__p.scopeRoots(sel)[0]; const b = r.getBoundingClientRect();
        return {x: b.x, y: b.y, w: Math.min(b.width, 1280), h: Math.min(b.height, 800)}; }""", ctx.scope)
    slides = ctx.anchors.get("slides")
    if slides:
        ids = ctx.call("findText", slides[0], None, False)
        if ids:
            region = ctx.on(ids[0], """let c = el; while (c.parentElement && c.parentElement !== document.body &&
                c.getBoundingClientRect().height < 40) c = c.parentElement; c.scrollIntoView({block: 'center'});
                const b = c.getBoundingClientRect(); return {x: b.x, y: b.y, w: b.width, h: b.height};""")
    t0_text = ctx.call("visibleText")
    s0 = ctx.shot(region)
    moving = False
    for _ in range(6):
        ctx.wait(1000)
        if ctx.call("visibleText") != t0_text or mad(s0, ctx.shot(region)) > FOCUS_DIFF_THRESHOLD:
            moving = True
            break
    if not moving:
        return na("no automatic movement or change within 6 s")
    ctrls = ctx.js("""() => Array.from(document.querySelectorAll('button,[role=button],[role=switch],[role=checkbox],input[type=checkbox],a[href]'))
        .filter(e => window.__p.rendered(e)).map(e => window.__p.id(e))""")
    pause = None
    for c in ctrls:
        name = (ctx.ax(c) or {}).get("name", "")
        if _PAUSE_RE.search(name or ""):
            pause = c
            break
    if pause is None:
        return res(True, False, "moving content has no control named pause/stop/play")
    ctx.on(pause, "el.focus();")
    if not ctx.active_within(pause):
        return res(True, False, "pause control cannot receive keyboard focus")
    ctx.key("Enter", 600)
    t1 = ctx.call("visibleText")
    s1 = ctx.shot(region)
    for _ in range(6):
        ctx.wait(2000)
        if ctx.call("visibleText") != t1 or mad(s1, ctx.shot(region)) > FOCUS_DIFF_THRESHOLD:
            return res(True, False, "content kept changing after activating the pause control with Enter")
    return res(True, True, {"control": (ctx.ax(pause) or {}).get("name")})


def combobox_behavior(ctx: Ctx) -> dict:
    a = ctx.anchors
    inp = ctx.js("""(sel) => { const roots = window.__p.scopeRoots(sel); for (const r of roots) {
        const e = r.querySelector('[role=combobox],input[list],input[type=search],input[type=text],input:not([type])');
        if (e) return window.__p.id(e); } return null; }""", ctx.scope)
    if inp is None:
        return res(True, False, "no text input")
    info = ctx.ax(inp) or {}
    named = bool((info.get("name") or "").strip())
    if not ctx.reach(inp, ctx.tab_budget()):
        return res(True, False, "input not reachable with Tab")
    ctx.page.keyboard.type(a["type_text"], delay=60)
    ctx.wait(500)
    native_list = bool(ctx.on(inp, "return !!(el.list && el.list.options.length);"))
    opt_ids = ctx.call("findText", a["expected_option"], None, True)
    shown = native_list or any(ctx.seen(i) for i in opt_ids)
    expanded = native_list or _truthy((ctx.ax(inp) or {}).get("expanded"))
    selected = False
    for _ in range(12):
        ctx.key("ArrowDown", 120)
        act = ctx.active()
        ad = ctx.on(inp, "return el.getAttribute('aria-activedescendant');")
        cur = ctx.js("(i) => { const e = document.getElementById(i); return e ? window.__p.norm(e.textContent) : null; }", ad) if ad else None
        cur_focus = ctx.on(act, "return window.__p.norm(el.textContent);") if act and act != inp else None
        if (cur and a["expected_option"].lower() in cur) or (cur_focus and a["expected_option"].lower() in cur_focus) \
                or native_list:
            ctx.key("Enter", 400)
            val = ctx.on(inp, "return el.value || el.textContent;") or ""
            selected = a["expected_option"].lower() in val.lower()
            if selected or not native_list:
                break
    detail = {"named": named, "options_shown": shown, "expanded_exposed": expanded, "selected_with_keyboard": selected}
    return res(True, all(detail.values()), detail)


def switch_behavior(ctx: Ctx) -> dict:
    a = ctx.anchors
    targets = []
    if a.get("switches_selector"):
        sw = ctx.js("""(sel) => { const e = document.querySelector(sel); if (!e) return null;
            const c = e.matches('input,button,[role=switch],[role=checkbox]') ? e : e.querySelector('input,button,[role=switch],[role=checkbox]');
            return window.__p.id(c || e); }""", a["switches_selector"])
        targets.append((a["switches_selector"], sw))
    for name in a.get("switches") or []:
        sw = ctx.call("fieldFor", name, ctx.scope) or ctx.find(name, actionable_only=True)
        targets.append((name, sw))
    out = []
    for name, sw in targets:
        if sw is None:
            out.append({"switch": name, "ok": False, "why": "not found"})
            continue
        ctx.reopen()
        sw = (ctx.js("(sel) => { const e = document.querySelector(sel); if (!e) return null; const c = e.matches('input,button,[role=switch],[role=checkbox]') ? e : e.querySelector('input,button,[role=switch],[role=checkbox]'); return window.__p.id(c || e); }", name)
              if name == a.get("switches_selector") else
              (ctx.call("fieldFor", name, ctx.scope) or ctx.find(name, actionable_only=True)))
        if not ctx.reach(sw, ctx.tab_budget()):
            out.append({"switch": name, "ok": False, "why": "not reachable with Tab"})
            continue
        foc = ctx.active()
        s0 = ctx.ax(foc) or {}
        ctx.key(" ", 400)
        s1 = ctx.ax(foc) or {}
        state_key = next((k for k in ("checked", "pressed") if k in s0 or k in s1), None)
        toggled = state_key is not None and str(s0.get(state_key)) != str(s1.get(state_key))
        if not toggled:
            ctx.key("Enter", 400)
            s1 = ctx.ax(foc) or {}
            state_key = next((k for k in ("checked", "pressed") if k in s0 or k in s1), None)
            toggled = state_key is not None and str(s0.get(state_key)) != str(s1.get(state_key))
        out.append({"switch": name, "ok": toggled, "role": s1.get("role"), "state": state_key})
    if not out:
        return na("no switch anchors")
    return res(True, all(o["ok"] for o in out), out)


def slider_behavior(ctx: Ctx) -> dict:
    sl = ctx.js("""(sel) => { const roots = window.__p.scopeRoots(sel);
        for (const r of roots) { if (r.matches('input[type=range],[role=slider]')) return window.__p.id(r);
          const e = r.querySelector('input[type=range],[role=slider]'); if (e) return window.__p.id(e); }
        return null; }""", ctx.scope)
    if sl is None:
        return res(True, False, "no input[type=range] or role=slider in scope")
    if not ctx.reach(sl, ctx.tab_budget()):
        return res(True, False, "slider not reachable with Tab")
    v0 = (ctx.ax(sl) or {}).get("value")
    ctx.key("ArrowRight", 250)
    v1 = (ctx.ax(sl) or {}).get("value")
    if v1 == v0:
        ctx.key("ArrowUp", 250)
        v1 = (ctx.ax(sl) or {}).get("value")
    exposed = v1 is not None
    return res(True, exposed and v1 != v0, {"before": v0, "after": v1})


def landmarks_present(ctx: Ctx) -> dict:
    out = ctx.js("""() => {
      const mains = Array.from(document.querySelectorAll('main,[role=main]')).filter(m => window.__p.rendered(m));
      const want = ['home', 'health information', 'tests'];
      let crumbs = null;
      for (const e of document.querySelectorAll('nav,[role=navigation],ol,ul,div,p')) {
        const t = window.__p.norm(e.innerText); let k = 0;
        for (const w of want) { const j = t.indexOf(w, k); if (j < 0) { k = -1; break; } k = j + w.length; }
        if (k >= 0 && t.length < 120) { crumbs = e; break; }
      }
      const nav = crumbs ? crumbs.closest('nav,[role=navigation]') : null;
      return {mains: mains.length, crumbs: !!crumbs, navId: nav ? window.__p.id(nav) : null};
    }""")
    nav_named = bool(out["navId"]) and bool(((ctx.ax(out["navId"]) or {}).get("name") or "").strip())
    ok = out["mains"] == 1 and nav_named
    return res(True, ok, {"main_count": out["mains"], "breadcrumb_found": out["crumbs"], "breadcrumb_nav_named": nav_named})


def heading_structure(ctx: Ctx) -> dict:
    hs = ctx.call("headingsOutline")
    h1 = sum(1 for h in hs if h["level"] == 1)
    skips = []
    prev = 0
    for h in hs:
        if prev and h["level"] > prev + 1:
            skips.append(f"h{prev} -> h{h['level']} ({h['text'][:40]})")
        prev = h["level"]
    return res(True, h1 == 1 and not skips, {"h1_count": h1, "skips": skips, "outline": [h["level"] for h in hs]})


def captions_track_present(ctx: Ctx) -> dict:
    vids = ctx.js("""() => Array.from(document.querySelectorAll('video')).map(v =>
        Array.from(v.querySelectorAll('track')).some(t => ['captions', 'subtitles'].includes((t.getAttribute('kind') || '').toLowerCase())))""")
    if not vids:
        return res(True, False, "no <video> element")
    return res(True, all(vids), {"videos": len(vids), "with_captions": sum(vids)})


def current_page_indicated(ctx: Ctx) -> dict:
    cur = ctx.anchors.get("current", "1")
    out = ctx.js("""(cur) => {
      const navs = Array.from(document.querySelectorAll('nav,[role=navigation]'));
      for (const n of navs) {
        const items = Array.from(n.querySelectorAll('a,span,li,button,[aria-current]'))
          .filter(e => window.__p.norm(e.textContent).replace(/current page|page/g, '').trim() === cur);
        if (!items.length) continue;
        const marked = items.some(e => e.getAttribute('aria-current') === 'page' || e.closest('[aria-current=page]') ||
          /current page/.test(window.__p.norm(e.textContent)));
        return {navId: window.__p.id(n), marked};
      }
      return {navId: null, marked: false};
    }""", cur)
    named = bool(out["navId"]) and bool(((ctx.ax(out["navId"]) or {}).get("name") or "").strip())
    return res(True, out["marked"] and named, {"current_marked": out["marked"], "nav_named": named})


def tooltip_behavior(ctx: Ctx) -> dict:
    tip_text = ctx.anchors["tooltip_text"]
    if ctx.text_visible(tip_text):
        return na("tooltip text is always visible")
    trig = None
    for _ in range(ctx.tab_budget()):
        ctx.page.keyboard.press("Tab")
        ctx.wait(150)
        if ctx.text_visible(tip_text):
            trig = ctx.active()
            break
    if trig is None:
        return res(True, False, "tooltip does not appear when its trigger gets keyboard focus")
    ctx.key("Escape", 300)
    hides = not ctx.text_visible(tip_text)
    focus_kept = ctx.active() == trig
    ctx.reopen()
    # hover the trigger, then move onto the tooltip itself
    trig2 = None
    for _ in range(ctx.tab_budget()):
        ctx.page.keyboard.press("Tab")
        ctx.wait(100)
        if ctx.text_visible(tip_text):
            trig2 = ctx.active()
            break
    hoverable = False
    if trig2:
        r = ctx.call("rect", trig2)
        ctx.on(trig2, "el.blur();")
        ctx.page.mouse.move(r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)
        ctx.wait(400)
        tip = next((i for i in ctx.call("findText", tip_text, None, False) if ctx.seen(i)), None)
        if tip:
            # Move the shortest way onto the tooltip (its nearest point to the trigger), then inside it.
            t = ctx.call("rect", tip)
            cx, cy = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
            nx = min(max(cx, t["x"] + 2), t["x"] + t["w"] - 2)
            ny = min(max(cy, t["y"] + 2), t["y"] + t["h"] - 2)
            ctx.page.mouse.move(nx, ny, steps=6)
            ctx.page.mouse.move(t["x"] + t["w"] / 2, t["y"] + t["h"] / 2, steps=6)
            ctx.wait(500)
            hoverable = ctx.text_visible(tip_text)
    detail = {"shows_on_focus": True, "escape_hides": hides, "focus_kept_on_escape": focus_kept,
              "hoverable": hoverable}
    return res(True, hides and focus_kept and hoverable, detail)


_FILL = {"email": "test@example.com", "tel": "07700900123", "date": "1990-01-01", "number": "30",
         "url": "https://example.com", "password": "Passw0rd!"}


def step_focus_management(ctx: Ctx) -> dict:
    nxt = ctx.find(ctx.anchors.get("next", "Next"), exact=True, actionable_only=True)
    if nxt is None:
        return res(True, False, "Next control not found")
    # fill the visible fields of step 1 so validation does not block the step change
    fields = ctx.js("""() => Array.from(document.querySelectorAll('input,select,textarea'))
        .filter(e => window.__p.rendered(e) && !['hidden','submit','button','checkbox','radio'].includes(e.type))
        .map(e => [window.__p.id(e), e.tagName.toLowerCase(), e.type, (e.name || e.id || '').toLowerCase()])""")
    for fid, tag, typ, name in fields:
        if tag == "select":
            ctx.on(fid, "if (el.options.length > 1) { el.selectedIndex = 1; el.dispatchEvent(new Event('change', {bubbles: true})); }")
            continue
        val = _FILL.get(typ) or ("test@example.com" if "mail" in name else "07700900123" if "phone" in name or "tel" in name
                                 else "1990-01-01" if "birth" in name or "dob" in name else "Alex Smith")
        ctx.on(fid, "el.focus();")
        ctx.page.keyboard.type(val)
    ctx.js("() => { document.querySelectorAll('*').forEach(e => { if (window.__p.rendered(e)) e.setAttribute('data-se4all-was-visible', '1'); }); }")
    ctx.on(nxt, "el.focus();")
    if not ctx.active_within(nxt):
        return res(True, False, "Next cannot receive keyboard focus")
    ctx.key("Enter", 600)
    act = ctx.active()
    moved = act is not None and act != nxt and bool(ctx.on(act, """
        const newlyShown = e => !e.hasAttribute('data-se4all-was-visible');
        const isHead = /^H[1-6]$/.test(el.tagName) || el.getAttribute('role') === 'heading';
        const isField = el.matches('input,select,textarea');
        const container = el.tabIndex === -1;
        return (isHead || isField || container) && (newlyShown(el) || Array.from(el.querySelectorAll('*')).some(newlyShown));"""))
    step_exposed = bool(ctx.js("""() => { const c = Array.from(document.querySelectorAll('[aria-current=step]')).filter(e => window.__p.rendered(e));
        if (c.some(e => /2|phone|contact|email/i.test(e.textContent))) return true;
        return /step\\s*2\\b|2\\s*of\\s*3/i.test(document.body.innerText); }"""))
    return res(True, moved and step_exposed, {"focus_moved_to_new_step": moved, "current_step_exposed": step_exposed})


def language_of_parts(ctx: Ctx) -> dict:
    out = []
    for text, code in (ctx.anchors.get("foreign_texts") or {}).items():
        ids = ctx.call("findText", text, ctx.scope, False)
        if not ids:
            out.append({"text": text, "ok": False, "why": "not found"})
            continue
        lang = ctx.on(ids[0], """const roots = window.__p.scopeRoots(arg);
            for (let e = el; e; e = e.parentElement) { if (e.hasAttribute('lang')) return e.getAttribute('lang');
              if (roots.includes(e)) break; } return null;""", ctx.scope)
        out.append({"text": text, "ok": bool(lang) and lang.lower().split("-")[0] == code, "lang": lang})
    if not out:
        return na("no foreign-text anchors")
    return res(True, all(o["ok"] for o in out), out)


def contrast_after_toggle(ctx: Ctx) -> dict:
    from .axe_runner import run_axe
    sel = ctx.anchors.get("switches_selector")
    el = ctx.js("(sel) => { const e = document.querySelector(sel); if (!e) return null; const c = e.matches('input,button,[role=switch],[role=checkbox]') ? e : (e.querySelector('input,button,[role=switch],[role=checkbox]') || e); return window.__p.id(c); }", sel)
    if el is None:
        return res(True, False, "toggle not found")
    ctx.page.locator(f'[data-se4all-id="{el}"]').click(force=True)
    ctx.wait(700)
    r = run_axe(ctx.page, None, ["color-contrast"])
    ok = r["rules"].get("color-contrast", True)
    return res(True, ok, {"violations": [n["target"] for v in r["violations"] for n in v["nodes"]][:5]})


def _effect_signature(ctx: Ctx) -> tuple:
    return (ctx.page.url, ctx.call("visibleText"),
            ctx.js("() => Array.from(document.querySelectorAll('[aria-expanded]')).map(e => e.getAttribute('aria-expanded')).join(',')"))


def keyboard_activation(ctx: Ctx, target_selector: str | None = None, require_space: bool = False) -> dict:
    sel = target_selector or ctx.scope
    def locate():
        return ctx.js("(sel) => { const e = document.querySelector(sel); return e ? window.__p.id(e) : null; }", sel)
    el = locate()
    if el is None:
        return res(True, False, f"target {sel!r} not found")
    sig0 = _effect_signature(ctx)
    try:
        ctx.page.locator(f'[data-se4all-id="{el}"]').click(timeout=3000)
    except Exception:                       # noqa: BLE001
        return res(True, False, "target cannot be clicked")
    ctx.wait(600)
    sig_click = _effect_signature(ctx)
    ctx.reopen()
    el = locate()
    role = (ctx.ax(el) or {}).get("role")
    if not ctx.reach(el, ctx.tab_budget()):
        return res(True, False, {"reachable": False, "role": role})
    ctx.key("Enter", 600)
    sig_enter = _effect_signature(ctx)
    enter_ok = sig_enter == sig_click and sig_click != sig0
    space_ok = True
    if require_space:
        ctx.reopen()
        el = locate()
        ctx.reach(el, ctx.tab_budget())
        ctx.key(" ", 600)
        space_ok = _effect_signature(ctx) == sig_click and sig_click != sig0
    role_ok = role in ("button", "link")
    detail = {"reachable": True, "role": role, "enter_same_as_click": enter_ok, "click_has_effect": sig_click != sig0}
    if require_space:
        detail["space_same_as_click"] = space_ok
    return res(True, role_ok and enter_ok and space_ok, detail)


def name_not_placeholder_only(ctx: Ctx) -> dict:
    out = ctx.js("""(sel) => { const roots = window.__p.scopeRoots(sel); const res = [];
      for (const r of roots) for (const f of r.querySelectorAll('input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=reset]):not([type=image]),select,textarea')) {
        const lab = f.labels && Array.from(f.labels).some(l => window.__p.norm(l.textContent));
        const al = window.__p.norm(f.getAttribute('aria-label'));
        const lb = (f.getAttribute('aria-labelledby') || '').split(/\\s+/).map(i => document.getElementById(i))
          .some(e => e && window.__p.norm(e.textContent));
        res.push({field: f.outerHTML.slice(0, 120), ok: !!(lab || al || lb)});
      } return res; }""", ctx.scope)
    if not out:
        return res(True, False, "no form fields in scope")
    return res(True, all(o["ok"] for o in out), [o for o in out if not o["ok"]][:5])


REGISTRY = {
    "tab_reachability": tab_reachability, "focus_visible": focus_visible, "no_keyboard_trap": no_keyboard_trap,
    "form_errors_associated": form_errors_associated, "error_not_color_only": error_not_color_only,
    "dialog_behavior": dialog_behavior, "live_region_on_action": live_region_on_action,
    "tabs_behavior": tabs_behavior, "accordion_behavior": accordion_behavior,
    "disclosure_nav_behavior": disclosure_nav_behavior, "disclosure_panel_behavior": disclosure_panel_behavior,
    "table_headers": table_headers, "sort_control": sort_control, "pause_control": pause_control,
    "combobox_behavior": combobox_behavior, "switch_behavior": switch_behavior,
    "slider_behavior": slider_behavior, "landmarks_present": landmarks_present,
    "heading_structure": heading_structure, "captions_track_present": captions_track_present,
    "current_page_indicated": current_page_indicated, "tooltip_behavior": tooltip_behavior,
    "step_focus_management": step_focus_management, "language_of_parts": language_of_parts,
    "contrast_after_toggle": contrast_after_toggle, "keyboard_activation": keyboard_activation,
    "name_not_placeholder_only": name_not_placeholder_only,
}


def run_probe(name: str, page_factory, scope: str | None = None, anchors: dict | None = None) -> dict:
    ctx = Ctx(page_factory, scope, anchors)
    try:
        out = REGISTRY[name](ctx)
    except Exception as exc:                 # noqa: BLE001 - a crashing probe is recorded, not hidden
        out = {"applicable": True, "passed": False, "detail": f"probe error: {type(exc).__name__}: {exc}",
               "error": True}
    finally:
        ctx.close()
    out["probe"] = name
    out["report_only"] = name in REPORT_ONLY
    return out
