"""E1-Audit (SPEC 5.4): one audit call per faulty page per model; reported items are matched
one-to-one to the injected faults. Writes runs/<exp>/scores/audit.jsonl.

    python -m harness.e1_audit [--pilot] [--models a,b] [--optional] [--lean]

Matching (greedy, in report order): a report matches the first unmatched fault for which
(a) its success criterion is one of the fault's accepted SCs, and (b) its element resolves
(querySelector on the faulty page) to the target, an ancestor or a descendant, OR its text
contains the target's id, its src, or its tag together with part of its text. Page-level
faults (M06 lang, M12 title, M10 stylesheet) match the element they live on (html, title or
head, a <style> or any focusable control) or a mention of lang/title/focus/outline.
Unmatched reports are false positives (a lower bound on precision).
"""
from __future__ import annotations

import concurrent.futures as cf
import re

from . import common, config, e1_corpus, e1_repair, llm

TEMPLATE = config.prompt("e1_audit.txt")
RETRY_JSON = "Respond with valid JSON only."
SC_RE = re.compile(r"\b([1-4])\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\b")

_RESOLVE_JS = r"""
([elementText, target, op]) => {
  let nodes = [];
  try { nodes = Array.from(document.querySelectorAll(elementText)); } catch (e) { nodes = []; }
  const tid = target.split('>')[0];
  const t = op === 'M06' ? document.documentElement : op === 'M12' ? (document.querySelector('title') || document.head)
          : document.getElementById(tid);
  const related = n => t && (n === t || n.contains(t) || t.contains(n));
  if (op === 'M10') return nodes.some(n => n.matches('style, html, body, a, button, input, select, textarea'));
  if (op === 'M12') return nodes.some(n => n.matches('title, head, html'));
  if (op === 'M06') return nodes.some(n => n === document.documentElement);
  return nodes.slice(0, 50).some(related);
}
"""


def normalize_sc(value) -> str | None:
    m = SC_RE.search(str(value or ""))
    return ".".join(m.groups()) if m else None


def _mentions(element_text: str, fault: dict, tinfo: dict) -> bool:
    e = (element_text or "").lower()
    op = fault["operator"]
    if op == "M06":
        return "<html" in e or "lang" in e or e.strip() == "html"
    if op == "M12":
        return "title" in e or "<head" in e
    if op == "M10":
        return any(w in e for w in ("focus", "outline", "<style", "stylesheet", "css"))
    tid = fault["target_id"].split(">")[0].lower()
    if re.search(rf"(#|id\s*=\s*[\"']?){re.escape(tid)}\b", e):
        return True
    if tinfo.get("src") and tinfo["src"].lower() in e:
        return True
    tag, text = tinfo.get("tag", ""), (tinfo.get("text") or "").lower()
    return bool(tag and text and f"<{tag}" in e and text[:25] in e)


def target_info(renderer, vid: str) -> dict:
    """Tag, text and src of each fault target on the faulty page."""
    gt = e1_repair.ground_truth()[vid]
    with renderer.page(e1_corpus.VARIANTS / f"{vid}.html") as page:
        return {f["target_id"]: page.evaluate("""(i) => { const e = document.getElementById(i.split('>')[0]); if (!e) return {};
            return {tag: e.tagName.toLowerCase(), text: (e.innerText || e.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 60),
                    src: e.getAttribute('src')}; }""", f["target_id"]) for f in gt["faults"]}


def match_reports(renderer, vid: str, reports: list[dict]) -> dict:
    gt = e1_repair.ground_truth()[vid]
    faults = gt["faults"]
    tinfo = target_info(renderer, vid)
    matched = [False] * len(faults)
    unmatched_reports = 0
    with renderer.page(e1_corpus.VARIANTS / f"{vid}.html") as page:
        for rep in reports:
            sc = normalize_sc(rep.get("success_criterion"))
            el = str(rep.get("element") or "")
            hit = None
            for i, f in enumerate(faults):
                if matched[i] or sc not in f["scs"]:
                    continue
                ok = _mentions(el, f, tinfo.get(f["target_id"], {}))
                if not ok:
                    try:
                        ok = bool(page.evaluate(_RESOLVE_JS, [el, f["target_id"], f["operator"]]))
                    except Exception:           # noqa: BLE001 - malformed selector text
                        ok = False
                if ok:
                    hit = i
                    break
            if hit is None:
                unmatched_reports += 1
            else:
                matched[hit] = True
    tp = sum(matched)
    return {"faults": [{"operator": f["operator"], "target_id": f["target_id"], "matched": m} for f, m in zip(faults, matched)],
            "tp": tp, "fp": unmatched_reports, "fn": len(faults) - tp}


def run_one(client, model: str, vid: str) -> dict:
    gt = e1_repair.ground_truth()[vid]
    page = (e1_corpus.VARIANTS / f"{vid}.html").read_text(encoding="utf-8")
    rec = {"model_key": model, "page_id": vid, "base_id": gt["base_id"], "provenance": gt["provenance"],
           "parse_ok": False, "retried": False, "reports": []}
    msgs = [llm.user(TEMPLATE.replace("{FAULTY_PAGE}", page))]
    try:
        r = client.call(model, msgs, item_id=f"audit/{vid}")
    except llm.LLMError as e:
        return {**rec, "status": "error", "error": str(e)[:300], "cost_usd": e.cost_usd, "latency_s": None,
                "call_ids": e.call_ids}
    cost, call_ids, latency = r.cost_usd, list(r.call_ids), r.total_s
    val = common.parse_json(r.text, list) if r.status == "ok" else None
    if val is None and r.status == "ok":
        rec["retried"] = True
        try:
            r2 = client.call(model, msgs + [r.assistant, llm.user(RETRY_JSON)], item_id=f"audit-retry/{vid}")
            cost += r2.cost_usd
            call_ids += r2.call_ids
            latency += r2.total_s
            val = common.parse_json(r2.text, list)
        except llm.LLMError as e:
            cost += e.cost_usd
    reports = [x for x in (val or []) if isinstance(x, dict)]
    rec.update(status=r.status, parse_ok=val is not None, reports=reports, n_reported=len(reports), cost_usd=cost,
               latency_s=latency, call_ids=call_ids, model_returned=r.model_returned)
    return rec


def main(argv=None):
    args = common.cli(__doc__).parse_args(argv)
    exp = common.experiment(args, "e1")
    models = common.select_models(args, "e1", "e1_models")
    gt = e1_repair.ground_truth()
    pages = common.id_list(args.pages) or (e1_repair.pilot_pages(gt) if args.pilot else sorted(gt))
    client = llm.LLM(exp)
    print(f"E1-Audit: {len(models)} models x {len(pages)} pages -> runs/{exp}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
        records = list(ex.map(lambda mv: run_one(client, *mv), [(m, v) for m in models for v in pages]))
    if not args.no_eval:
        jobs = []
        for rec in records:
            key = common.cache_key("audit", rec["page_id"], rec.get("reports", []))
            rec["_k"] = key
            jobs.append((key, "harness.e1_audit.match_reports", (rec["page_id"], rec.get("reports", []))))
        results = common.evaluate_all(jobs, config.RUNS / exp / "evalcache", args.workers)
        for rec in records:
            ev = results[rec.pop("_k")]
            if "eval_error" in ev:
                ev = {"faults": [{"operator": f["operator"], "target_id": f["target_id"], "matched": False}
                                 for f in gt[rec["page_id"]]["faults"]], "tp": 0, "fp": len(rec.get("reports", [])),
                      "fn": len(gt[rec["page_id"]]["faults"]), "eval_error": ev["eval_error"]}
            rec.update(ev)
            rec.setdefault("n_reported", len(rec.get("reports", [])))
    records.sort(key=lambda r: (r["model_key"], r["page_id"]))
    common.write_jsonl(config.RUNS / exp / "scores" / "audit.jsonl", records)
    print(f"wrote {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
