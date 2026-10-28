// DOM helpers for the keyboard and semantics probes (SPEC 4). Injected into the page as
// window.__p. Elements are referred to from Python by a stable data-se4all-id attribute.
window.__p = (() => {
  const ID = 'data-se4all-id';
  let next = 1;
  const norm = s => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();

  function id(el) {
    if (!el || el.nodeType !== 1) return null;
    if (!el.hasAttribute(ID)) el.setAttribute(ID, String(next++));
    return el.getAttribute(ID);
  }
  function byId(i) { return i == null ? null : document.querySelector(`[${ID}="${i}"]`); }
  // nth-child path from <html>; stable across reloads of a deterministic page.
  function path(el) {
    const parts = [];
    for (let e = el; e && e.nodeType === 1 && e !== document.documentElement; e = e.parentElement) {
      parts.unshift(Array.prototype.indexOf.call(e.parentElement.children, e));
    }
    return parts.join('/');
  }
  function byPath(p) {
    let e = document.documentElement;
    for (const k of p.split('/').filter(x => x !== '')) { e = e && e.children[parseInt(k, 10)]; }
    return e ? id(e) : null;
  }

  function active() {
    let a = document.activeElement;
    while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
    if (!a || a === document.body || a === document.documentElement) return null;
    return a;
  }

  // Rendered at all (may be visually hidden, e.g. a 1px clipped input).
  function rendered(el) {
    if (!el || !el.isConnected) return false;
    if (el.checkVisibility && !el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }
  // Perceivable by a sighted user: rendered, not clipped away, not 1px.
  function seen(el) {
    if (!rendered(el)) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    for (let e = el; e && e !== document.documentElement; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.clip && cs.clip !== 'auto' && /rect\(0(px)?,? 0(px)?,? 0(px)?,? 0(px)?\)/.test(cs.clip.replace(/\s+/g, ' '))) return false;
      if (cs.clipPath && /inset\(\s*50%/.test(cs.clipPath)) return false;
      if (cs.overflow === 'hidden') {
        const pr = e.getBoundingClientRect();
        if (pr.width < 2 || pr.height < 2) return false;
      }
    }
    return true;
  }

  function scopeRoots(sel) {
    if (!sel) return [document.body];
    const out = [];
    for (const s of sel.split(',')) { document.querySelectorAll(s.trim()).forEach(e => out.push(e)); }
    return out;
  }
  function inScope(el, roots) { return roots.some(r => r === el || r.contains(el)); }
  function allIn(roots) {
    const out = [];
    for (const r of roots) { out.push(r); r.querySelectorAll('*').forEach(e => out.push(e)); }
    return out;
  }

  const SPEC_ROLES = new Set(['button', 'link', 'checkbox', 'switch', 'tab', 'menuitem', 'option', 'slider']);
  const COMPOSITE = '[role=tablist],[role=menu],[role=menubar],[role=listbox],[role=radiogroup],[role=tree],[role=grid],[role=treegrid],[role=toolbar]';
  const ACTIONABLE = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=link],[role=tab],[role=switch],[role=checkbox],[role=menuitem],[role=option],[role=slider],[role=combobox],[role=radio],[tabindex],[contenteditable=true]';

  function nativeInteractive(el) {
    const t = el.tagName.toLowerCase();
    if (t === 'a' || t === 'area') return el.hasAttribute('href');
    if (['button', 'select', 'textarea', 'summary', 'iframe'].includes(t)) return true;
    if (t === 'input') return el.type !== 'hidden';
    if ((t === 'audio' || t === 'video') && el.hasAttribute('controls')) return true;
    return el.isContentEditable && el.getAttribute('contenteditable') !== null;
  }
  function roleOf(el) { return (el.getAttribute('role') || '').trim().split(/\s+/)[0]; }
  function explicitInteractive(el) {
    const ti = el.getAttribute('tabindex');
    return (ti !== null && parseInt(ti, 10) >= 0) || SPEC_ROLES.has(roleOf(el));
  }
  function clickableLooking(el) {
    const t = el.tagName.toLowerCase();
    if (['label', 'html', 'body'].includes(t)) return false;
    let looks = el.hasAttribute('onclick');
    if (!looks) {
      if (getComputedStyle(el).cursor !== 'pointer') return false;
      const p = el.parentElement;
      if (p && getComputedStyle(p).cursor === 'pointer') return false;   // root of a pointer region only
      looks = true;
    }
    // Wrappers around a real control, and parts of a control, are not separate targets.
    if (el.parentElement && el.parentElement.closest(ACTIONABLE)) return false;
    if (el.querySelector(ACTIONABLE)) return false;
    if (el.closest('label')) return false;
    return looks;
  }
  function interactive(el) {
    if (el.matches(':disabled')) return false;
    return nativeInteractive(el) || explicitInteractive(el) || clickableLooking(el);
  }
  function groupKey(el) {
    if (el.tagName === 'INPUT' && el.type === 'radio' && el.name)
      return 'radio:' + (el.form ? id(el.form) : '') + ':' + el.name;
    const c = el.parentElement && el.parentElement.closest(COMPOSITE);
    return c ? 'composite:' + id(c) : null;
  }
  // Visible interactive elements in scope, as [{id, group}].
  function interactives(sel) {
    const roots = scopeRoots(sel);
    return allIn(roots).filter(e => e.nodeType === 1 && interactive(e) && rendered(e))
      .map(e => ({id: id(e), group: groupKey(e)}));
  }

  function startRecording() {
    window.__reached = new Set();
    if (!window.__recOn) {
      document.addEventListener('focusin', e => {
        const t = (e.composedPath && e.composedPath()[0]) || e.target;
        if (window.__reached) window.__reached.add(t);
      }, true);
      window.__recOn = true;
    }
  }
  function reached(i) {
    const el = byId(i);
    if (!el || !window.__reached) return false;
    for (const r of window.__reached) if (r === el || el.contains(r)) return true;
    return false;
  }

  function textOf(el) {
    let s = el.innerText !== undefined ? el.innerText : el.textContent;
    if (!rendered(el)) s = el.textContent;
    const extra = [el.getAttribute('aria-label'), el.getAttribute('alt'), el.getAttribute('title')];
    if (el.tagName === 'INPUT' && ['button', 'submit', 'reset'].includes(el.type)) extra.push(el.value);
    return norm([s, ...extra].filter(Boolean).join(' '));
  }
  function ownText(el) {
    const parts = [el.getAttribute('aria-label')];
    if (el.tagName === 'INPUT' && ['button', 'submit', 'reset'].includes(el.type)) parts.push(el.value);
    parts.push(el.innerText !== undefined && rendered(el) ? el.innerText : el.textContent);
    return norm(parts.filter(Boolean).join(' '));
  }
  // Deepest elements whose text contains `text` (exact=true: whose own text equals it).
  function findText(text, sel, exact) {
    const t = norm(text);
    const roots = scopeRoots(sel);
    const skip = new Set(['SCRIPT', 'STYLE', 'TEMPLATE', 'NOSCRIPT', 'HEAD', 'TITLE', 'OPTION']);
    let m = allIn(roots).filter(e => e.nodeType === 1 && !skip.has(e.tagName) &&
      (exact ? ownText(e) === t : textOf(e).includes(t)));
    m = m.filter(e => !m.some(o => o !== e && e.contains(o)));
    m.sort((a, b) => (rendered(b) - rendered(a)));
    return m.map(id);
  }
  // The control a user would operate for a text match: nearest actionable ancestor-or-self.
  function actionable(i) {
    const el = byId(i);
    if (!el) return null;
    const c = el.closest(ACTIONABLE);
    if (c) return id(c);
    // A <label> names a control elsewhere.
    const lab = el.closest('label');
    if (lab && lab.control) return id(lab.control);
    return id(el);
  }
  function isActionable(i) { const el = byId(i); return !!(el && el.closest(ACTIONABLE)); }

  // Controls whose label/name/placeholder/id/name attribute mentions `key`.
  function fieldFor(key, sel) {
    const k = norm(key);
    const roots = scopeRoots(sel);
    const fields = allIn(roots).filter(e => e.matches && e.matches('input:not([type=hidden]),select,textarea,[role=combobox],[role=textbox],[role=slider],[role=switch],[role=checkbox]'));
    for (const f of fields) {
      const labels = f.labels ? Array.from(f.labels).map(l => l.textContent) : [];
      const lb = (f.getAttribute('aria-labelledby') || '').split(/\s+/).map(x => document.getElementById(x)).filter(Boolean).map(x => x.textContent);
      const hay = norm([...labels, ...lb, f.getAttribute('aria-label'), f.getAttribute('placeholder'),
                        f.id, f.getAttribute('name')].filter(Boolean).join(' '));
      if (hay.includes(k)) return id(f);
    }
    return null;
  }

  function visibleText() { return norm(document.body.innerText); }

  function rect(i) {
    const el = byId(i);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {x: r.x, y: r.y, w: r.width, h: r.height};
  }
  // Region whose pixels show this element's focus indicator.
  function focusRegion(i) {
    const el = byId(i);
    if (!el) return null;
    el.scrollIntoView({block: 'center', inline: 'center'});
    let target = el;
    const r0 = el.getBoundingClientRect();
    const op = parseFloat(getComputedStyle(el).opacity);
    if (r0.width < 4 || r0.height < 4 || op === 0) {
      target = (el.labels && el.labels[0]) || el.closest('label') || el.parentElement || el;
    }
    const r = target.getBoundingClientRect();
    return {x: r.x - 4, y: r.y - 4, w: r.width + 8, h: r.height + 8};
  }

  function inModal(el) {
    const d = el && el.closest('[role=dialog][aria-modal=true],[role=alertdialog][aria-modal=true],dialog');
    if (!d) return false;
    return d.tagName !== 'DIALOG' || d.matches(':modal');
  }

  function openDialog() {
    const cands = Array.from(document.querySelectorAll('[role=dialog],[role=alertdialog],dialog'));
    return cands.find(d => (d.tagName !== 'DIALOG' || d.open) && rendered(d)) || null;
  }

  function liveRegionOf(el) {
    for (let e = el; e; e = e.parentElement) {
      const role = roleOf(e);
      const live = (e.getAttribute('aria-live') || '').toLowerCase();
      if (['status', 'alert', 'log'].includes(role) || live === 'polite' || live === 'assertive' ||
          e.tagName === 'OUTPUT') return e;
    }
    return null;
  }
  function markLiveRegions() {
    document.querySelectorAll('[role=status],[role=alert],[role=log],[aria-live],output')
      .forEach(e => e.setAttribute('data-se4all-prelive', '1'));
  }

  function headingsOutline() {
    const hs = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6,[role=heading]'))
      .filter(h => rendered(h) || h.closest('[hidden],[aria-hidden=true]') === null && h.getClientRects().length > 0);
    return hs.map(h => {
      const lvl = h.getAttribute('role') === 'heading' ? parseInt(h.getAttribute('aria-level') || '2', 10)
                                                    : parseInt(h.tagName.slice(1), 10);
      return {level: lvl, text: norm(h.textContent).slice(0, 80)};
    });
  }

  function fieldState(i) {
    const f = byId(i);
    if (!f) return null;
    const cs = getComputedStyle(f);
    return {invalidAttr: f.getAttribute('aria-invalid'), cssInvalid: f.matches(':invalid'),
            border: cs.borderColor + '|' + cs.outlineColor + '|' + cs.boxShadow + '|' + cs.backgroundColor,
            describedby: f.getAttribute('aria-describedby'), errormessage: f.getAttribute('aria-errormessage'),
            required: f.required || f.getAttribute('aria-required') === 'true'};
  }
  function formFields(sel) {
    const roots = scopeRoots(sel);
    return allIn(roots).filter(e => e.matches && e.matches(
      'input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=reset]):not([type=image]),select,textarea'))
      .filter(e => rendered(e) || (e.labels && e.labels[0] && rendered(e.labels[0])))
      .map(id);
  }
  // Text programmatically associated with a field: describedby, errormessage, labels.
  function associatedText(i) {
    const f = byId(i);
    const ids = [(f.getAttribute('aria-describedby') || ''), (f.getAttribute('aria-errormessage') || '')]
      .join(' ').split(/\s+/).filter(Boolean);
    const els = ids.map(x => document.getElementById(x)).filter(Boolean);
    if (f.labels) els.push(...f.labels);
    const lb = (f.getAttribute('aria-labelledby') || '').split(/\s+/).map(x => document.getElementById(x)).filter(Boolean);
    els.push(...lb);
    return els.map(e => ({id: id(e), text: norm(e.innerText || e.textContent), seen: seen(e)}));
  }
  // Visible text near a field: its closest field wrapper (does not include other fields).
  function nearbyText(i) {
    const f = byId(i);
    let w = f.parentElement;
    while (w && w !== document.body) {
      const n = w.querySelectorAll('input:not([type=hidden]),select,textarea').length;
      if (n > 1) break;
      if (w.matches('fieldset,li,p,div,section')) {
        const up = w.parentElement;
        if (!up || up.querySelectorAll('input:not([type=hidden]),select,textarea').length > 1) break;
      }
      w = w.parentElement;
    }
    w = w || f.parentElement;
    const out = [];
    w.querySelectorAll('*').forEach(e => {
      if (e === f || e.contains(f)) return;
      if (seen(e)) out.push({id: id(e), text: norm(e.innerText || ''), tag: e.tagName,
                             alt: e.getAttribute('alt') || e.getAttribute('aria-label') || ''});
    });
    return out;
  }

  return {id, byId, path, byPath, active, rendered, seen, interactives, startRecording, reached, findText, actionable,
          isActionable, fieldFor, visibleText, rect, focusRegion, inModal, openDialog, liveRegionOf,
          markLiveRegions, headingsOutline, fieldState, formFields, associatedText, nearbyText, norm,
          roleOf, scopeRoots};
})();
