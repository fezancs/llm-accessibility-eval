"""E1-Repair (SPEC 5.3): one repair call per faulty page per model; scoring by resolved
faults, regressions and preservation. Writes runs/<exp>/scores/repair.jsonl.

    python -m harness.e1_repair [--pilot] [--models a,b] [--optional] [--lean]

Preservation compares the repaired page with the BASE page (the ideal repair): visible-text
similarity (normalised Levenshtein of innerText) >= 0.95, every base-page id present, and
DOM tag-sequence similarity >= 0.85. SSIM of full-page screenshots is diagnostic only.
Regressions = axe (rule, node) violations on the repaired page that were not violations
on the faulty page, plus generic probes that passed on the faulty page and fail after repair.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from . import common, config, e1_corpus, llm, probes

TEMPLATE = config.prompt("e1_repair.txt")
TEXT_SIM_MIN, TAG_SIM_MIN = 0.95, 0.85


def ground_truth() -> dict:
    return json.loads((e1_corpus.CORPUS / "ground_truth.json").read_text(encoding="utf-8"))


def pilot_pages(gt: dict) -> list[str]:
    """Pilot: 4 faulty pages, two W3C and two authored, fixed."""
    return ["B01-V1", "B04-V1", "B05-V1", "B09-V1"]


def page_profile(renderer, path_str: str, keep_screenshot: bool = True) -> dict:
    """axe violations, generic probes, text/ids/tags and a full-page screenshot of a page
    (path relative to runs/). Screenshots of repaired pages are deleted after SSIM."""
    path = config.RUNS / path_str
    facts = e1_corpus.page_facts(renderer, path)
    with renderer.page(path) as page:
        dom = page.evaluate("""() => ({text: document.body ? document.body.innerText : '',
            ids: Array.from(document.querySelectorAll('[id]')).map(e => e.id),
            tags: Array.from(document.body ? document.body.querySelectorAll('*') : []).map(e => e.tagName.toLowerCase())})""")
        png = page.screenshot(full_page=True)
    shot = config.RUNS / (path_str + ".png")
    shot.write_bytes(png)
    generic = {}
    for n in probes.GENERIC:
        r = probes.run_probe(n, lambda: renderer.page(path))
        generic[n] = r["passed"] if r["applicable"] else None
    viol = sorted({f"{v['id']}|{n['target']}" for v in facts["axe"]["violations"] for n in v["nodes"]})
    return {"facts": facts, "violations": viol, "generic": generic, "text": dom["text"], "ids": dom["ids"],
            "tags": dom["tags"], "screenshot": str(shot.relative_to(config.RUNS))}


def ssim(a_path: str, b_path: str) -> float | None:
    try:
        from skimage.metrics import structural_similarity
        a = np.asarray(Image.open(config.RUNS / a_path).convert("L"))
        b = np.asarray(Image.open(config.RUNS / b_path).convert("L"))
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        return float(structural_similarity(a[:h, :w], b[:h, :w], data_range=255))
    except Exception:                       # noqa: BLE001 - diagnostic only
        return None


def score_repair(renderer, artifact: str, vid: str, faulty: dict, base: dict) -> dict:
    gt = ground_truth()[vid]
    path = config.RUNS / artifact
    rep = page_profile(renderer, artifact)
    faults = [{"operator": f["operator"], "target_id": f["target_id"],
               "resolved": bool(e1_corpus.resolved(renderer, path, f, rep["facts"]))} for f in gt["faults"]]
    new_viol = sorted(set(rep["violations"]) - set(faulty["violations"]))
    probe_reg = [f"probe:{n}" for n, ok in faulty["generic"].items() if ok is True and rep["generic"].get(n) is False]
    regressions = new_viol + probe_reg
    text_sim = Levenshtein.normalized_similarity(base["text"], rep["text"])
    ids_ok = set(base["ids"]) <= set(rep["ids"])
    tag_sim = Levenshtein.normalized_similarity(base["tags"], rep["tags"])
    preserved = text_sim >= TEXT_SIM_MIN and ids_ok and tag_sim >= TAG_SIM_MIN
    k = len(faults)
    n_res = sum(f["resolved"] for f in faults)
    score = max(0.0, (n_res - len(regressions)) / k) * (1.0 if preserved else 0.0)
    ssim_value = ssim(base["screenshot"], rep["screenshot"])
    (config.RUNS / rep["screenshot"]).unlink(missing_ok=True)       # disk: keep only reference screenshots
    return {"faults": faults, "regressions": len(regressions), "regression_list": regressions[:30],
            "preserved": preserved,
            "preservation": {"text_sim": round(text_sim, 4), "ids_ok": ids_ok, "tag_sim": round(tag_sim, 4),
                             "ssim": ssim_value,
                             "missing_ids": sorted(set(base["ids"]) - set(rep["ids"]))[:20]},
            "score": score, "full_repair": n_res == k and not regressions and preserved, "k": k}


def run_one(client, model: str, vid: str, art_dir) -> dict:
    gt = ground_truth()[vid]
    page = (e1_corpus.VARIANTS / f"{vid}.html").read_text(encoding="utf-8")
    rec = {"model_key": model, "page_id": vid, "base_id": gt["base_id"], "provenance": gt["provenance"], "k": len(gt["faults"])}
    try:
        r = client.call(model, [llm.user(TEMPLATE.replace("{FAULTY_PAGE}", page))], item_id=f"repair/{vid}")
    except llm.LLMError as e:
        return {**rec, "status": "error", "error": str(e)[:300], "html_found": False, "cost_usd": e.cost_usd,
                "latency_s": None, "call_ids": e.call_ids}
    html = common.extract_html(r.text) if r.status == "ok" else None
    prompt_tokens = r.usage["input_tokens"] + r.usage["cached_input_tokens"] + r.usage["cache_write_tokens"]
    rec.update(status=r.status, html_found=html is not None, cost_usd=r.cost_usd, latency_s=r.total_s, ttft_s=r.ttft_s,
               input_tokens=prompt_tokens, output_tokens=r.usage["output_tokens"], page_tokens=prompt_tokens,
               call_ids=r.call_ids, stop_reason=r.stop_reason, model_returned=r.model_returned)
    if html is not None:
        path = art_dir / model / f"{vid}.html"
        common.link_images(path.parent)
        path.write_text(html, encoding="utf-8")
        rec["artifact"] = str(path.relative_to(config.RUNS))
    return rec


def zero_score(rec: dict, gt: dict, why: str | None = None) -> None:
    rec.update(faults=[{"operator": f["operator"], "target_id": f["target_id"], "resolved": False} for f in gt["faults"]],
               regressions=0, regression_list=[], preserved=False, preservation=None, score=0.0, full_repair=False)
    if why:
        rec["eval_error"] = why


def main(argv=None):
    args = common.cli(__doc__).parse_args(argv)
    exp = common.experiment(args, "e1")
    models = common.select_models(args, "e1", "e1_models")
    gt = ground_truth()
    pages = common.id_list(args.pages) or (pilot_pages(gt) if args.pilot else sorted(gt))
    client = llm.LLM(exp)
    art_dir = config.RUNS / exp / "artifacts" / "repair"
    print(f"E1-Repair: {len(models)} models x {len(pages)} pages -> runs/{exp}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
        records = list(ex.map(lambda mv: run_one(client, *mv, art_dir), [(m, v) for m in models for v in pages]))
    if not args.no_eval:
        cache = config.RUNS / exp / "evalcache"
        ref_dir = config.RUNS / exp / "artifacts" / "repair_refs"
        common.link_images(ref_dir)
        prof_jobs = []
        for pid in sorted({gt[v]["base_id"] for v in pages}):
            dst = ref_dir / f"{pid}.html"
            dst.write_bytes((e1_corpus.BASE / f"{pid}.html").read_bytes())
            prof_jobs.append((common.cache_key("profile", dst.read_bytes(), pid), "harness.e1_repair.page_profile",
                              (str(dst.relative_to(config.RUNS)),)))
        for v in pages:
            dst = ref_dir / f"{v}.html"
            dst.write_bytes((e1_corpus.VARIANTS / f"{v}.html").read_bytes())
            prof_jobs.append((common.cache_key("profile", dst.read_bytes(), v), "harness.e1_repair.page_profile",
                              (str(dst.relative_to(config.RUNS)),)))
        profiles = common.evaluate_all(prof_jobs, cache, args.workers)
        by_name = {pathlib_name(j[2][0]): profiles[j[0]] for j in prof_jobs}
        jobs = []
        for rec in records:
            if rec.get("artifact"):
                v = rec["page_id"]
                key = common.cache_key("repair", v, (config.RUNS / rec["artifact"]).read_bytes())
                rec["_k"] = key
                jobs.append((key, "harness.e1_repair.score_repair",
                             (rec["artifact"], v, by_name[v], by_name[gt[v]["base_id"]])))
        results = common.evaluate_all(jobs, cache, args.workers)
        for rec in records:
            ev = results.get(rec.pop("_k", None))
            if ev is None or "eval_error" in ev:
                zero_score(rec, gt[rec["page_id"]], (ev or {}).get("eval_error"))
            else:
                rec.update(ev)
    records.sort(key=lambda r: (r["model_key"], r["page_id"]))
    common.write_jsonl(config.RUNS / exp / "scores" / "repair.jsonl", records)
    print(f"wrote {len(records)} records", flush=True)


def pathlib_name(p: str) -> str:
    return p.rsplit("/", 1)[-1].removesuffix(".html")


if __name__ == "__main__":
    main()
