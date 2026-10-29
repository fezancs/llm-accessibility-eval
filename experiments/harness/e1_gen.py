"""E1-Gen (SPEC 5.1): 20 UI tasks x 2 conditions x 3 samples per model, a self-assessment
turn, and automated scoring. Writes runs/<exp>/scores/gen.jsonl (harness/SCHEMAS.md).

    python -m harness.e1_gen [--pilot] [--models a,b] [--optional] [--lean]
"""
from __future__ import annotations

import concurrent.futures as cf
import pathlib

from . import common, config, functional, llm, oracle

TASKS = config.task_file("e1_gen_tasks.yaml")["tasks"]
TEMPLATES = {"unguided": config.prompt("e1_gen_unguided.txt"), "nudged": config.prompt("e1_gen_nudged.txt")}
SELFASSESS = config.prompt_body("e1_selfassess.txt")
SAMPLES = 3
RETRY_JSON = "Respond with valid JSON only."


def generation_prompt(task: dict, condition: str) -> str:
    return TEMPLATES[condition].replace("{TASK_PROMPT}", task["prompt"].strip())


def run_one(client: llm.LLM, model: str, task: dict, condition: str, sample: int, art_dir: pathlib.Path) -> dict:
    item = f"gen/{task['id']}/{condition}"
    msgs = [llm.user(generation_prompt(task, condition))]
    rec = {"model_key": model, "task_id": task["id"], "condition": condition, "sample_idx": sample,
           "selfassess": {"conforms": None, "parse_ok": False, "retried": False}}
    try:
        r = client.call(model, msgs, item_id=item, sample_idx=sample)
    except llm.LLMError as e:
        return {**rec, "status": "error", "error": str(e)[:300], "html_found": False, "cost_usd": e.cost_usd,
                "latency_s": None, "ttft_s": None, "input_tokens": None, "output_tokens": None,
                "call_ids": e.call_ids}
    html = common.extract_html(r.text) if r.status == "ok" else None
    rec.update(status=r.status, html_found=html is not None, cost_usd=r.cost_usd, latency_s=r.total_s,
               ttft_s=r.ttft_s, stop_reason=r.stop_reason,
               input_tokens=r.usage["input_tokens"] + r.usage["cached_input_tokens"] + r.usage["cache_write_tokens"],
               output_tokens=r.usage["output_tokens"], call_ids=r.call_ids, model_returned=r.model_returned)
    if html is None:
        return rec
    path = art_dir / model / f"{task['id']}_{condition}_s{sample}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    rec["artifact"] = str(path.relative_to(config.RUNS))
    # Self-assessment: second user turn in the same conversation; not counted in cost or latency.
    sa_msgs = msgs + [r.assistant, llm.user(SELFASSESS)]
    sa_cost = 0.0
    try:
        s = client.call(model, sa_msgs, item_id=f"selfassess/{task['id']}/{condition}", sample_idx=sample)
        sa_cost += s.cost_usd
        val = common.parse_json(s.text, dict)
        if not (val and isinstance(val.get("conforms_wcag22_aa"), bool)):
            rec["selfassess"]["retried"] = True
            s2 = client.call(model, sa_msgs + [s.assistant, llm.user(RETRY_JSON)],
                             item_id=f"selfassess-retry/{task['id']}/{condition}", sample_idx=sample)
            sa_cost += s2.cost_usd
            val = common.parse_json(s2.text, dict)
        if val and isinstance(val.get("conforms_wcag22_aa"), bool):
            rec["selfassess"].update(conforms=val["conforms_wcag22_aa"], parse_ok=True,
                                     known_issues=[str(x)[:200] for x in (val.get("known_issues") or [])][:20])
    except llm.LLMError as e:
        rec["selfassess"]["error"] = str(e)[:200]
    rec["selfassess"]["cost_usd"] = sa_cost
    return rec


def score_artifact(renderer, artifact: str, task_id: str) -> dict:
    task = next(t for t in TASKS if t["id"] == task_id)
    path = config.RUNS / artifact
    func = functional.run_checks(renderer, path, task.get("functional", []))
    ev = oracle.evaluate(renderer, path, probe_names=task.get("probes", []), anchors=task.get("anchors") or {})
    ev.pop("probes_detail", None)
    return {"functional_checks": func, "functional": all(c["passed"] for c in func), **ev}


def main(argv=None):
    args = common.cli(__doc__).parse_args(argv)
    exp = common.experiment(args, "e1")
    models = common.select_models(args, "e1", "e1_models")
    wanted = common.id_list(args.tasks)
    tasks = [t for t in TASKS if t["id"] in wanted] if wanted else (TASKS[:3] if args.pilot else TASKS)
    samples = args.samples or (1 if args.pilot else SAMPLES)
    client = llm.LLM(exp)
    art_dir = config.RUNS / exp / "artifacts" / "gen"
    jobs = [(m, t, c, s) for m in models for t in tasks for c in ("unguided", "nudged") for s in range(samples)]
    print(f"E1-Gen: {len(models)} models, {len(jobs)} generations -> runs/{exp}", flush=True)
    records = []
    with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(run_one, client, *j, art_dir) for j in jobs]
        for f in cf.as_completed(futs):
            records.append(f.result())
    if not args.no_eval:
        from . import runmeta
        from .render import Renderer
        with Renderer() as r:
            runmeta.write(r)
        ev_jobs = []
        for rec in records:
            if rec.get("artifact"):
                html = (config.RUNS / rec["artifact"]).read_bytes()
                key = common.cache_key("gen", rec["task_id"], html)
                rec["_eval_key"] = key
                ev_jobs.append((key, "harness.e1_gen.score_artifact", (rec["artifact"], rec["task_id"])))
        results = common.evaluate_all(ev_jobs, config.RUNS / exp / "evalcache", args.workers)
        for rec in records:
            ev = results.get(rec.pop("_eval_key", None)) if rec.get("artifact") else None
            if ev is None or "eval_error" in (ev or {}):
                rec.update(functional=False, a=None, checks=[], report_only=[], n_elements=None,
                           violating_nodes=0, violations_per_100=None, incomplete=0,
                           eval_error=(ev or {}).get("eval_error"))
                continue
            rec.update({k: ev[k] for k in ("functional", "functional_checks", "a", "checks", "report_only",
                                           "n_elements", "violating_nodes", "violations_per_100", "incomplete",
                                           "axe_violations", "moving")})
            rec["probe_details"] = [{"probe": p["probe"], "applicable": p["applicable"], "passed": p["passed"]}
                                    for p in ev["probes"]]
    records.sort(key=lambda r: (r["model_key"], r["task_id"], r["condition"], r["sample_idx"]))
    common.write_jsonl(config.RUNS / exp / "scores" / "gen.jsonl", records)
    print(f"wrote {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
