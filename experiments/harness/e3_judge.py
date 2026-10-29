"""E3 judging (SPEC 7.4): pointwise ratings, known-answer pairs, BLV pairs, generator pairs.

    python -m harness.e3_judge                               # all procedures, judge panel from models.yaml
    python -m harness.e3_judge --procedures pointwise,known
    python -m harness.e3_judge --lean                        # E3c on 80 images x 12 pairs (budget.yaml)
    python -m harness.e3_judge --pilot                       # 10 items per procedure, runs/pilot/
    python -m harness.e3_judge --models claude-sonnet-5-5    # restrict (or replace) the judges

Procedures (every judge, every pair in both orders):
- pointwise: every D_BLV and D_CR item, prompts/e3_judge_pointwise.txt -> {"score": 1-5}.
- known: each item's original description against four degraded versions whose answer is
  known by construction: append (+ the first sentence of the description of another image in
  the same dataset), repeat (the description twice), swap (the description of another image
  in the same dataset), truncate (the first three words; not built when the description has
  three words or fewer). The correct answer is the original.
- blv_pairs: D_BLV descriptions of the same image rated in the same article whose BLV mean
  overall ratings differ by >= 0.5; the higher-rated one is the human-preferred side. (Each
  image's descriptions were rated in different articles; only same-article pairs share the
  context a judge is shown, so only those pairs have a comparable human preference.)
- gen_pairs: for each E3c image, a seeded degree-balanced sample of generator pairs (16 of
  the 28 pairs of 8 generators, each generator in exactly 4), judged on the anonymised alt
  texts. A pair whose either output is missing is dropped and counted.

Prompts: the image first, then the prompt text up to the first candidate (marked as a prompt-
cache breakpoint, so all pairs for the same image and context share a cached prefix), then the
candidates and the answer format. Judges run with purpose="judge" (models.yaml judge_params:
their lowest reasoning setting). An unparseable reply gets one follow-up turn "Respond with
valid JSON only."; if that fails too the verdict is recorded as null.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import itertools
import json
import pathlib
import random
import re
import sys
import threading

from . import budget, config, llm
from .e3_data import blv_pair_groups, load_items
from .e3_generate import (RETRY_TEXT, context_values, fill, image_part, parse_json_object, preflight,
                          prompt_template, sample_images)

PROCEDURES = ["pointwise", "known", "blv_pairs", "gen_pairs"]
DEGRADATIONS = ["append", "repeat", "swap", "truncate"]
GEN_PAIRS_PER_IMAGE = 16
GEN_PAIR_SEED = 7
PILOT_N = 10
BLV_PAIR_MIN_DIFF = 0.5
INNER_WORKERS = 4

# ------------------------------------------------------------------ prompts

def split_template(name: str, marker: str) -> tuple[str, str]:
    t = prompt_template(name)
    i = t.index(marker)
    return t[:i], t[i:]


def judge_messages(item: dict, name: str, marker: str, candidates: dict) -> list[dict]:
    """[image, shared prefix (cache breakpoint), candidates + answer format]."""
    head, tail = split_template(name, marker)
    values = context_values(item["context"]) | candidates
    return [llm.user(image_part(item),
                     {"type": "text", "text": fill(head, values), "cache_breakpoint": True},
                     {"type": "text", "text": fill(tail, values)})]


def pointwise_messages(item: dict, candidate: str) -> list[dict]:
    return judge_messages(item, "e3_judge_pointwise.txt", "Candidate alt text:", {"CANDIDATE": candidate})


def pairwise_messages(item: dict, first: str, second: str) -> list[dict]:
    return judge_messages(item, "e3_judge_pairwise.txt", "Candidate 1:",
                          {"CANDIDATE_1": first, "CANDIDATE_2": second})


# ------------------------------------------------------------------ parsing

def _as_int(v, allowed: set[int]) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and float(v).is_integer() and int(v) in allowed:
        return int(v)
    if isinstance(v, str) and v.strip().isdigit() and int(v.strip()) in allowed:
        return int(v.strip())
    return None


def parse_score(text: str | None) -> int | None:
    obj = parse_json_object(text)
    return _as_int(obj.get("score"), {1, 2, 3, 4, 5}) if obj else None


def parse_better(text: str | None) -> tuple[int, int | None] | None:
    obj = parse_json_object(text)
    if not obj:
        return None
    better = _as_int(obj.get("better"), {1, 2})
    if better is None:
        return None
    return better, _as_int(obj.get("confidence"), {1, 2, 3, 4, 5})


# ------------------------------------------------------------------ degradations

_SENT = re.compile(r"^(.+?[.!?])(?=\s|$)", re.S)


def first_sentence(text: str) -> str:
    t = text.strip()
    m = _SENT.match(t)
    return (m.group(1) if m else t).strip()


def degrade(item: dict, kind: str, pool: list[dict]) -> str | None:
    """Degraded version of item['description'] (None when the degradation does not apply).
    `pool` = the other items of the same dataset; the choice of other item is seeded per item."""
    orig = item["description"].strip()
    rng = random.Random(f"e3-known:{item['item_id']}:{kind}")
    others = sorted((o for o in pool if o["image_id"] != item["image_id"]), key=lambda o: o["item_id"])
    if kind == "repeat":
        return f"{orig} {orig}"
    if kind == "truncate":
        words = orig.split()
        return " ".join(words[:3]) if len(words) > 3 else None
    if kind == "append":
        cands = [o for o in others if first_sentence(o["description"]) and
                 first_sentence(o["description"]).lower() not in orig.lower()]
        if not cands:
            return None
        return f"{orig} {first_sentence(rng.choice(cands)['description'])}"
    if kind == "swap":
        cands = [o for o in others if o["description"].strip().lower() != orig.lower()]
        return rng.choice(cands)["description"].strip() if cands else None
    raise ValueError(kind)


def known_pairs(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """All (item, degradation) pairs; also the ones that do not apply (for the report)."""
    by_ds = collections.defaultdict(list)
    for it in items:
        by_ds[it["dataset"]].append(it)
    pairs, skipped = [], []
    for it in items:
        if not it["retrieved"]:
            continue
        for kind in DEGRADATIONS:
            d = degrade(it, kind, by_ds[it["dataset"]])
            (pairs if d is not None else skipped).append(
                {"item": it, "degradation": kind, "degraded": d})
    return pairs, skipped


# ------------------------------------------------------------------ pair sampling

def sample_pairs(generators: list[str], image_id: str, n_pairs: int, seed: int = GEN_PAIR_SEED) -> list[tuple[str, str]]:
    """n distinct generator pairs for one image, each generator in the same number of pairs
    (+-1 when 2n is not a multiple of the number of generators). Uniform over such graphs:
    configuration model with rejection, seeded per image."""
    g = len(generators)
    all_pairs = list(itertools.combinations(range(g), 2))
    if n_pairs >= len(all_pairs):
        return [(generators[a], generators[b]) for a, b in all_pairs]
    rng = random.Random(f"e3-pairs:{seed}:{image_id}")
    base, extra = divmod(2 * n_pairs, g)
    for _ in range(100_000):
        deg = [base] * g
        for k in rng.sample(range(g), extra):
            deg[k] += 1
        stubs = [k for k in range(g) for _ in range(deg[k])]
        rng.shuffle(stubs)
        edges = set()
        for a, b in zip(stubs[::2], stubs[1::2]):
            e = (min(a, b), max(a, b))
            if a == b or e in edges:
                break
            edges.add(e)
        else:
            return [(generators[a], generators[b]) for a, b in sorted(edges)]
    raise RuntimeError("could not sample a balanced pair set")


# ------------------------------------------------------------------ judging

def judge_once(client: llm.LLM, judge: str, msgs: list[dict], parser, log_id: str) -> dict:
    out = {"value": None, "parse_ok": False, "retried": False, "status": "ok", "cost_usd": 0.0,
           "latency_s": 0.0, "call_ids": []}
    try:
        r = client.call(judge, msgs, item_id=log_id, purpose="judge")
    except llm.LLMError as e:
        out.update(status="error", error=str(e)[:300], cost_usd=e.cost_usd, call_ids=e.call_ids)
        return out
    out.update(cost_usd=r.cost_usd, latency_s=r.total_s, call_ids=list(r.call_ids))
    if r.status == "refusal":
        out["status"] = "refusal"
        return out
    val = parser(r.text)
    if val is None:
        out["retried"] = True
        try:
            r2 = client.call(judge, msgs + [r.assistant, llm.user(RETRY_TEXT)], item_id=log_id + "/retry",
                             purpose="judge")
        except llm.LLMError as e:
            out.update(status="error", error=str(e)[:300], cost_usd=out["cost_usd"] + e.cost_usd)
            return out
        out["cost_usd"] += r2.cost_usd
        out["latency_s"] += r2.total_s
        out["call_ids"] += list(r2.call_ids)
        if r2.status == "refusal":
            out["status"] = "refusal"
            return out
        val = parser(r2.text)
    if val is not None:
        out.update(value=val, parse_ok=True)
    return out


class Sink:
    """Collects records from worker threads; writes a .partial file as it goes."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.records: list[dict] = []
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = open(path.with_suffix(".jsonl.partial"), "w", encoding="utf-8")

    def add(self, rec: dict):
        with self._lock:
            self.records.append(rec)
            self._tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._tmp.flush()

    def finish(self, key):
        self._tmp.close()
        self.records.sort(key=key)
        with open(self.path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.path.with_suffix(".jsonl.partial").unlink(missing_ok=True)


def run_groups(groups: list[list], workers: int) -> None:
    """Each group is a list of thunks sharing a cached prompt prefix: the first runs alone
    (it writes the provider cache), the rest run concurrently. Groups run in parallel."""
    def run_group(jobs):
        jobs[0]()
        if len(jobs) > 1:
            with cf.ThreadPoolExecutor(max_workers=INNER_WORKERS) as inner:
                for f in [inner.submit(j) for j in jobs[1:]]:
                    f.result()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_group, g) for g in groups if g]
        try:
            for f in cf.as_completed(futs):
                f.result()
        except budget.BudgetExceeded:
            ex.shutdown(wait=False, cancel_futures=True)
            raise


def _meta(o: dict) -> dict:
    return {"parse_ok": o["parse_ok"], "retried": o["retried"], "status": o["status"],
            "cost_usd": o["cost_usd"], "latency_s": o["latency_s"], "call_ids": o["call_ids"],
            **({"error": o["error"]} if "error" in o else {})}


def do_pointwise(client, judges, items, sink, workers):
    groups = []
    for j in judges:
        for it in items:
            def job(j=j, it=it):
                o = judge_once(client, j, pointwise_messages(it, it["description"]), parse_score,
                               f"e3/pointwise/{j}/{it['item_id']}")
                sink.add({"judge": j, "dataset": it["dataset"], "item_id": it["item_id"],
                          "image_id": it["image_id"], "score": o["value"], **_meta(o)})
            groups.append([job])
    run_groups(groups, workers)


def do_known(client, judges, pairs, sink, workers):
    by_item = collections.defaultdict(list)
    for p in pairs:
        by_item[p["item"]["item_id"]].append(p)
    groups = []
    for j in judges:
        for item_id, ps in by_item.items():
            jobs = []
            for p in ps:
                it, orig, deg = p["item"], p["item"]["description"].strip(), p["degraded"]
                for original_first in (True, False):
                    first, second = (orig, deg) if original_first else (deg, orig)

                    def job(j=j, it=it, p=p, first=first, second=second, original_first=original_first):
                        o = judge_once(client, j, pairwise_messages(it, first, second), parse_better,
                                       f"e3/known/{j}/{it['item_id']}/{p['degradation']}/{int(original_first)}")
                        better = o["value"][0] if o["value"] else None
                        sink.add({"judge": j, "dataset": it["dataset"], "item_id": it["item_id"],
                                  "image_id": it["image_id"], "degradation": p["degradation"],
                                  "original_first": original_first, "len_first": len(first),
                                  "len_second": len(second), "better": better,
                                  "confidence": o["value"][1] if o["value"] else None,
                                  "y": None if better is None else int(better == 1),
                                  "correct": None if better is None else (better == 1) == original_first,
                                  **_meta(o)})
                    jobs.append(job)
            groups.append(jobs)
    run_groups(groups, workers)


def blv_pairs(items: list[dict]) -> list[dict]:
    out = []
    for (_, _), its in sorted(blv_pair_groups([i for i in items if i["retrieved"]]).items()):
        its = sorted(its, key=lambda i: i["item_id"])
        for a, b in itertools.combinations(its, 2):
            diff = a["human"]["overall_mean"] - b["human"]["overall_mean"]
            if abs(diff) >= BLV_PAIR_MIN_DIFF:
                out.append({"a": a, "b": b, "human_prefers": "a" if diff > 0 else "b"})
    return out


def do_blv_pairs(client, judges, pairs, sink, workers):
    groups = []
    for j in judges:
        by_ctx = collections.defaultdict(list)
        for p in pairs:
            by_ctx[(p["a"]["image_id"], p["a"]["context"]["article_title"])].append(p)
        for _, ps in sorted(by_ctx.items()):
            jobs = []
            for p in ps:
                for a_first in (True, False):
                    def job(j=j, p=p, a_first=a_first):
                        a, b = p["a"], p["b"]
                        first, second = (a, b) if a_first else (b, a)
                        o = judge_once(client, j, pairwise_messages(a, first["description"], second["description"]),
                                       parse_better, f"e3/blv/{j}/{a['item_id']}/{b['item_id']}/{int(a_first)}")
                        better = o["value"][0] if o["value"] else None
                        prefers_a = None if better is None else (better == 1) == a_first
                        sink.add({"judge": j, "image_id": a["image_id"], "item_a": a["item_id"],
                                  "item_b": b["item_id"], "human_prefers": p["human_prefers"], "a_first": a_first,
                                  "len_first": len(first["description"]), "len_second": len(second["description"]),
                                  "better": better, "confidence": o["value"][1] if o["value"] else None,
                                  "y": None if better is None else int(better == 1),
                                  "correct": None if prefers_a is None else prefers_a == (p["human_prefers"] == "a"),
                                  **_meta(o)})
                    jobs.append(job)
            groups.append(jobs)
    run_groups(groups, workers)


def gen_pair_plan(generators: list[str], image_ids: list[str], outputs: dict, n_pairs: int):
    """Sampled pairs per image (fixed in advance) and the pairs dropped for missing outputs."""
    plan, dropped = [], []
    for img in image_ids:
        for a, b in sample_pairs(generators, img, n_pairs):
            if (a, img) in outputs and (b, img) in outputs:
                plan.append((img, a, b))
            else:
                dropped.append({"image_id": img, "a": a, "b": b,
                                "missing": [g for g in (a, b) if (g, img) not in outputs]})
    return plan, dropped


def do_gen_pairs(client, judges, plan, outputs, items_by_image, sink, workers):
    groups = []
    for j in judges:
        by_img = collections.defaultdict(list)
        for img, a, b in plan:
            by_img[img].append((a, b))
        for img, prs in by_img.items():
            it = items_by_image[img]
            jobs = []
            for a, b in prs:
                for a_first in (True, False):
                    def job(j=j, it=it, img=img, a=a, b=b, a_first=a_first):
                        g1, g2 = (a, b) if a_first else (b, a)
                        r1, r2 = outputs[(g1, img)], outputs[(g2, img)]
                        o = judge_once(client, j, pairwise_messages(it, r1["alt_clean"], r2["alt_clean"]),
                                       parse_better, f"e3/gen/{j}/{img}/{r1['opaque_id']}/{r2['opaque_id']}")
                        better = o["value"][0] if o["value"] else None
                        sink.add({"judge": j, "image_id": img, "gen_first": g1, "gen_second": g2,
                                  "id_first": r1["opaque_id"], "id_second": r2["opaque_id"],
                                  "len_first": len(r1["alt_clean"]), "len_second": len(r2["alt_clean"]),
                                  "better": better, "confidence": o["value"][1] if o["value"] else None,
                                  "y": None if better is None else int(better == 1), **_meta(o)})
                    jobs.append(job)
            groups.append(jobs)
    run_groups(groups, workers)


# ------------------------------------------------------------------ CLI

def pilot_subset(items: list[dict], n: int = PILOT_N) -> list[dict]:
    """n items split evenly between the datasets, seeded."""
    out = []
    for k, ds in enumerate(("blv", "cr")):
        pool = sorted((i for i in items if i["dataset"] == ds), key=lambda i: i["item_id"])
        take = n // 2 + (n % 2 if k == 0 else 0)
        out += random.Random(f"e3-pilot:{ds}").sample(pool, min(take, len(pool)))
    return out


def plan_counts(judges: list[str], procs: list[str], pilot: bool, lean: bool) -> dict:
    """Judgments each procedure would request (all generator outputs assumed present)."""
    items = [i for i in load_items() if i["retrieved"]]
    n_j = len(judges)
    out = {"judges": judges}
    if "pointwise" in procs:
        out["pointwise"] = n_j * len(pilot_subset(items) if pilot else items)
    if "known" in procs:
        ids = {i["item_id"] for i in (pilot_subset(items) if pilot else items)}
        pairs, skipped = known_pairs(load_items())
        out["known"] = 2 * n_j * sum(1 for p in pairs if p["item"]["item_id"] in ids)
        out["known_not_applicable"] = dict(collections.Counter(
            s["degradation"] for s in skipped if s["item"]["item_id"] in ids))
    if "blv_pairs" in procs:
        ps = blv_pairs(items)
        out["blv_pairs"] = 2 * n_j * len(ps[:PILOT_N] if pilot else ps)
    if "gen_pairs" in procs:
        gens = config.models_with_role("e3_gen")
        n_img, n_pairs = (PILOT_N if pilot else 120), GEN_PAIRS_PER_IMAGE
        if lean and not pilot:
            lp = config.budget_config()["lean_profile"]
            n_img, n_pairs = int(lp["e3_images"]), int(lp["e3_pairs_per_image"])
        imgs = sample_images(load_items(), n_img)
        out["gen_pairs"] = 2 * n_j * sum(len(sample_pairs(gens, i, n_pairs)) for i in imgs)
    out["total"] = sum(v for k, v in out.items() if isinstance(v, int))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E3 judging")
    ap.add_argument("--procedures", default=",".join(PROCEDURES))
    ap.add_argument("--pilot", action="store_true", help=f"{PILOT_N} items per procedure; runs/pilot/")
    ap.add_argument("--lean", action="store_true", help="E3c images and pairs from budget.yaml lean_profile")
    ap.add_argument("--models", help="comma-separated judge keys (default: the models.yaml judge panel)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plan", action="store_true", help="print the number of judgments; call nothing")
    args = ap.parse_args(argv)
    procs = [p for p in args.procedures.split(",") if p]
    unknown = set(procs) - set(PROCEDURES)
    if unknown:
        ap.error(f"unknown procedures {sorted(unknown)}")

    config.load_env()
    judges = args.models.split(",") if args.models else list(config.models_config()["judges"].values())
    if args.plan:
        print(json.dumps(plan_counts(judges, procs, args.pilot, args.lean), indent=2))
        return 0
    preflight(judges)
    items = [i for i in load_items() if i["retrieved"]]
    out_dir = config.RUNS / ("pilot/e3" if args.pilot else "e3")
    jdir = out_dir / "judgments"
    client = llm.LLM("pilot", budget_key="pilot") if args.pilot else llm.LLM("e3")
    report = {"judges": judges, "procedures": procs}
    try:
        if "pointwise" in procs:
            its = pilot_subset(items) if args.pilot else items
            sink = Sink(jdir / "pointwise.jsonl")
            do_pointwise(client, judges, its, sink, args.workers)
            sink.finish(lambda r: (r["judge"], r["item_id"]))
            report["pointwise"] = len(sink.records)
        if "known" in procs:
            its = pilot_subset(items) if args.pilot else items
            pool = load_items()            # degradations may borrow text from any item of the dataset
            ids = {i["item_id"] for i in its}
            pairs, skipped = known_pairs(pool)
            pairs = [p for p in pairs if p["item"]["item_id"] in ids]
            sink = Sink(jdir / "known_pairs.jsonl")
            do_known(client, judges, pairs, sink, args.workers)
            sink.finish(lambda r: (r["judge"], r["item_id"], r["degradation"], not r["original_first"]))
            report["known"] = {"judgments": len(sink.records), "pairs": len(pairs),
                               "not_applicable": collections.Counter(
                                   s["degradation"] for s in skipped if s["item"]["item_id"] in ids)}
        if "blv_pairs" in procs:
            ps = blv_pairs(items)[:PILOT_N] if args.pilot else blv_pairs(items)
            if not ps:
                report["blv_pairs"] = "skipped: D_BLV has no same-article description pairs differing by >= 0.5"
            else:
                sink = Sink(jdir / "blv_pairs.jsonl")
                do_blv_pairs(client, judges, ps, sink, args.workers)
                sink.finish(lambda r: (r["judge"], r["item_a"], r["item_b"], not r["a_first"]))
                report["blv_pairs"] = {"pairs": len(ps), "judgments": len(sink.records)}
        if "gen_pairs" in procs:
            gen_file = out_dir / "generated.jsonl"
            meta_file = out_dir / "generated_meta.json"
            if not gen_file.exists():
                raise SystemExit(f"{gen_file} not found: run harness.e3_generate first")
            gens = [json.loads(line) for line in open(gen_file, encoding="utf-8") if line.strip()]
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            generators = meta.get("generators") or config.models_with_role("e3_gen")
            n_img, n_pairs = 120, GEN_PAIRS_PER_IMAGE
            if args.lean:
                lp = config.budget_config()["lean_profile"]
                n_img, n_pairs = int(lp["e3_images"]), int(lp["e3_pairs_per_image"])
            if args.pilot:
                n_img = PILOT_N
            # The images the generators wrote for (a nested prefix of the seeded sample).
            if meta.get("images"):
                n_img = min(n_img, len(meta["images"]))
            image_ids = sample_images(load_items(), n_img)
            if meta.get("images") and meta["images"][:n_img] != image_ids:
                raise SystemExit("generated_meta.json images differ from the seeded sample; regenerate")
            outputs = {(r["generator"], r["image_id"]): r for r in gens
                       if r["status"] == "ok" and r.get("alt_clean")}
            plan, dropped = gen_pair_plan(generators, image_ids, outputs, n_pairs)
            by_image = {i["image_id"]: i for i in items if i["dataset"] == "cr"}
            sink = Sink(jdir / "gen_pairs.jsonl")
            do_gen_pairs(client, judges, plan, outputs, by_image, sink, args.workers)
            sink.finish(lambda r: (r["judge"], r["image_id"], r["gen_first"], r["gen_second"]))
            report["gen_pairs"] = {"images": len(image_ids), "pairs_per_image": n_pairs,
                                   "pairs_judged": len(plan), "pairs_dropped": len(dropped),
                                   "judgments": len(sink.records)}
            (jdir / "gen_pairs_dropped.json").write_text(json.dumps(dropped, indent=2), encoding="utf-8")
    except budget.BudgetExceeded as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "run_report.json").write_text(json.dumps(report, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
