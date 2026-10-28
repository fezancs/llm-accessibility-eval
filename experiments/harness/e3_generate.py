"""E3c alt-text generation (SPEC 7.2-7.3).

    python -m harness.e3_generate                  # 120 D_CR images x every core e3_gen model
    python -m harness.e3_generate --lean           # 80 images (budget.yaml lean_profile)
    python -m harness.e3_generate --pilot          # 10 images, logs and outputs under runs/pilot/
    python -m harness.e3_generate --models a,b     # restrict generators
    python -m harness.e3_generate --leakage        # SPEC 7.3 leakage check on generated.jsonl

Each generator writes alt text for each sampled image with prompts/e3_alt_generate.txt (the
image first, then the prompt text). The reply must be JSON {"alt": ...}; an unparseable reply
gets one follow-up turn "Respond with valid JSON only." before it is recorded as a failure.

Anonymisation (SPEC 7.3), in this order, each rule logged per generator when it changes
the text: markdown (emphasis, code, headings, list bullets, links), quotes (matching
quotes around the whole text), alt_text_prefix (leading "Alt text:" / "Alt-text:"),
image_of / picture_of (leading "Image of" / "Picture of", with an optional article), then
whitespace is collapsed and the first letter capitalised. Each output gets an opaque id.

The run is resumable: every call is cached, and the output file is rewritten from the cache.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import random
import re
import sys

from . import budget, config, llm
from .e3_data import load_items

SAMPLE_SEED = 7
N_IMAGES = 120
PILOT_IMAGES = 10
RETRY_TEXT = "Respond with valid JSON only."
PROMPT_FILE = "e3_alt_generate.txt"


# ------------------------------------------------------------------ sampling and prompts

def sample_images(items: list[dict], n: int = N_IMAGES, seed: int = SAMPLE_SEED) -> list[str]:
    """The E3c image sample: n retrieved D_CR images, seeded. Smaller samples (lean, pilot)
    are prefixes of the full sample, so they are nested."""
    pool = sorted({i["image_id"] for i in items if i["dataset"] == "cr" and i["retrieved"]})
    full = random.Random(seed).sample(pool, min(N_IMAGES, len(pool)))
    return full[:n]


def prompt_template(name: str) -> str:
    """Prompt text after the '[image]' marker line (the image part is sent first)."""
    body = config.prompt_body(name)
    first, rest = body.split("\n", 1)
    if first.strip() != "[image]":
        raise ValueError(f"{name}: expected the first line to be [image]")
    return rest.lstrip("\n")


_PLACEHOLDER = re.compile(r"\{(ARTICLE_TITLE|SECTION_TITLE|CONTEXT_EXCERPT|CAPTION_OR_NONE|"
                          r"CANDIDATE_1|CANDIDATE_2|CANDIDATE)\}")


def fill(template: str, values: dict) -> str:
    """Replace every known placeholder in one pass, so inserted text can never be re-substituted
    (str.format is unusable: the prompts contain literal JSON braces)."""
    def rep(m):
        key = m.group(1)
        if key not in values:
            raise KeyError(f"no value for {{{key}}}")
        return values[key]
    return _PLACEHOLDER.sub(rep, template)


def context_values(ctx: dict) -> dict:
    return {"ARTICLE_TITLE": ctx.get("article_title") or "none",
            "SECTION_TITLE": ctx.get("section_title") or "none",
            "CONTEXT_EXCERPT": ctx.get("excerpt") or "none",
            "CAPTION_OR_NONE": ctx.get("caption") or "none"}


def fill_context(template: str, context: dict) -> str:
    return fill(template, context_values(context))


def image_part(item: dict) -> dict:
    return {"type": "image", "path": str(config.ROOT / item["image_path"]),
            "media_type": item.get("media_type") or "image/jpeg"}


def generation_messages(item: dict) -> list[dict]:
    text = fill_context(prompt_template(PROMPT_FILE), item["context"])
    return [llm.user(image_part(item), text)]


# ------------------------------------------------------------------ parsing

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_json_object(text: str | None) -> dict | None:
    """The JSON object in a reply: the whole reply, a fenced block, or the first {...} span."""
    if not text:
        return None
    s = _FENCE.sub("", text.strip()).strip()
    for candidate in (s, *re.findall(r"\{.*?\}", s, re.S), *re.findall(r"\{.*\}", s, re.S)):
        try:
            val = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(val, dict):
            return val
    return None


def parse_alt(text: str | None) -> str | None:
    obj = parse_json_object(text)
    if obj is None or not isinstance(obj.get("alt"), str):
        return None
    return obj["alt"]


# ------------------------------------------------------------------ anonymisation

_QUOTES = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("«", "»"), ("`", "`")]
_MD_RULES = [
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),          # [text](url) -> text
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.M), ""),           # headings
    (re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M), ""),    # list bullets
    (re.compile(r"(\*\*|__)(.+?)\1", re.S), r"\2"),          # bold
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", re.S), r"\1"),   # *italic*
    (re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", re.S), r"\1"),     # _italic_
    (re.compile(r"`([^`]+)`"), r"\1"),                       # inline code
]
_PREFIX_ALT = re.compile(r"^\s*alt[\s-]?text\s*:\s*", re.I)
_PREFIX_IMAGE = re.compile(r"^\s*(?:(?:an?|the)\s+)?image\s+of\s+", re.I)
_PREFIX_PICTURE = re.compile(r"^\s*(?:(?:an?|the)\s+)?picture\s+of\s+", re.I)
RULES = ["markdown", "quotes", "alt_text_prefix", "image_of", "picture_of"]


def _free_quote(text: str, ch: str) -> bool:
    """True if `ch` occurs in `text` other than as an apostrophe inside a word (dog's)."""
    return any(not (0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum())
               for i, c in enumerate(text) if c == ch)


def _strip_quotes(s: str) -> str:
    """Remove quotes wrapping the whole text, but never quotes that open or close inside it."""
    s = s.strip()
    changed = True
    while changed and len(s) >= 2:
        changed = False
        for a, b in _QUOTES:
            if s.startswith(a) and s.endswith(b):
                inner = s[len(a):-len(b)].strip()
                if inner and not _free_quote(inner, a) and not _free_quote(inner, b):
                    s, changed = inner, True
                    break
    return s


def anonymise(alt: str) -> tuple[str, list[str]]:
    """Normalise one alt text; return (clean text, rules that changed it)."""
    fired = []
    s = alt.strip()
    t = s
    for pat, rep in _MD_RULES:
        t = pat.sub(rep, t)
    if t != s:
        fired.append("markdown")
        s = t.strip()
    t = _strip_quotes(s)
    if t != s:
        fired.append("quotes")
        s = t
    t = _PREFIX_ALT.sub("", s)
    if t != s:
        fired.append("alt_text_prefix")
        s = t.strip()
        t = _strip_quotes(s)
        if t != s:
            fired.append("quotes")
            s = t
    for name, pat in (("image_of", _PREFIX_IMAGE), ("picture_of", _PREFIX_PICTURE)):
        t = pat.sub("", s)
        if t != s:
            fired.append(name)
            s = t
    s = re.sub(r"\s+", " ", s).strip()
    if s:
        s = s[0].upper() + s[1:]
    return s, fired


def opaque_id(generator: str, image_id: str) -> str:
    """Random-looking, reproducible identifier that does not reveal the generator."""
    return "%016x" % random.Random(f"e3-opaque:{image_id}:{generator}").getrandbits(64)


# ------------------------------------------------------------------ running

def preflight(models: list[str]) -> None:
    """Refuse to start if a model is unverified or has no price (no call is made)."""
    import datetime as dt
    for k in models:
        m = config.model(k)
        llm._check_available(m)
        if m["provider"] == "openrouter" and not m.get("openrouter_provider"):
            raise llm.ModelUnavailable(f"{k}: openrouter_provider (pinned host) not set in models.yaml")
        budget.price_table(m, {}, dt.datetime.now(dt.timezone.utc))


def generate_one(client: llm.LLM, generator: str, item: dict) -> dict:
    msgs = generation_messages(item)
    rec = {"generator": generator, "image_id": item["image_id"], "item_id": item["item_id"],
           "status": "ok", "alt_raw": None, "alt_clean": None, "transforms": [], "parse_ok": False,
           "retried": False, "opaque_id": opaque_id(generator, item["image_id"]), "length": None,
           "cost_usd": 0.0, "latency_s": 0.0, "ttft_s": None, "call_ids": []}
    try:
        r = client.call(generator, msgs, item_id=f"e3/alt/{item['image_id']}")
    except llm.LLMError as e:
        rec.update(status="error", error=str(e)[:300], cost_usd=e.cost_usd, call_ids=e.call_ids)
        return rec
    rec.update(cost_usd=r.cost_usd, latency_s=r.total_s, ttft_s=r.ttft_s, call_ids=list(r.call_ids),
               model_returned=r.model_returned)
    if r.status == "refusal":
        rec["status"] = "refusal"
        return rec
    alt = parse_alt(r.text)
    if alt is None:
        rec["retried"] = True
        try:
            r2 = client.call(generator, msgs + [r.assistant, llm.user(RETRY_TEXT)],
                             item_id=f"e3/alt/{item['image_id']}/retry")
        except llm.LLMError as e:
            rec.update(status="error", error=str(e)[:300], cost_usd=rec["cost_usd"] + e.cost_usd)
            return rec
        rec["cost_usd"] += r2.cost_usd
        rec["latency_s"] += r2.total_s
        rec["call_ids"] += list(r2.call_ids)
        alt = parse_alt(r2.text) if r2.status == "ok" else None
        if r2.status == "refusal":
            rec["status"] = "refusal"
            return rec
    if alt is None:
        rec["status"] = "parse_error"
        return rec
    clean_alt, fired = anonymise(alt)
    rec.update(parse_ok=True, alt_raw=alt, alt_clean=clean_alt, transforms=fired, length=len(clean_alt))
    if not clean_alt:
        rec["status"] = "empty"
    return rec


def run(generators: list[str], items_by_image: dict, image_ids: list[str], client: llm.LLM,
        out: pathlib.Path, workers: int = 16) -> list[dict]:
    jobs = [(g, items_by_image[i]) for i in image_ids for g in generators]
    records = []
    tmp = out.with_suffix(".jsonl.partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(generate_one, client, g, it): (g, it["image_id"]) for g, it in jobs}
        try:
            for fut in cf.as_completed(futs):
                rec = fut.result()
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
        except budget.BudgetExceeded:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    order = {g: k for k, g in enumerate(generators)}
    pos = {i: k for k, i in enumerate(image_ids)}
    records.sort(key=lambda r: (pos[r["image_id"]], order[r["generator"]]))
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.unlink(missing_ok=True)
    return records


def transform_counts(records: list[dict]) -> dict:
    out = collections.defaultdict(lambda: {r: 0 for r in RULES} | {"n": 0})
    for r in records:
        if r.get("parse_ok"):
            out[r["generator"]]["n"] += 1
            for t in r["transforms"]:
                out[r["generator"]][t] += 1
    return dict(out)


# ------------------------------------------------------------------ leakage check (SPEC 7.3)

def leakage_check(records: list[dict], seed: int = 0) -> dict:
    """5-fold logistic regression on character 3-5-grams predicting the generator's family
    from anonymised alt text, against the majority-class baseline."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    recs = [r for r in records if r.get("alt_clean")]
    texts = [r["alt_clean"] for r in recs]
    fams = [config.family(r["generator"]) for r in recs]
    counts = collections.Counter(fams)
    majority = max(counts.values()) / len(fams) if fams else None
    n_splits = min(5, min(counts.values())) if counts else 0
    if len(counts) < 2 or n_splits < 2:
        return {"n": len(recs), "families": dict(counts), "accuracy": None, "majority_baseline": majority,
                "note": "too few families or examples for 5-fold cross-validation"}
    pipe = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
                         LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(pipe, texts, fams, cv=cv)
    acc = float(np.mean(np.array(pred) == np.array(fams)))
    return {"n": len(recs), "families": dict(counts), "folds": n_splits, "accuracy": round(acc, 4),
            "majority_baseline": round(majority, 4)}


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E3c alt-text generation")
    ap.add_argument("--pilot", action="store_true", help=f"{PILOT_IMAGES} images; logs under runs/pilot/")
    ap.add_argument("--lean", action="store_true", help="lean_profile e3_images from budget.yaml")
    ap.add_argument("--models", help="comma-separated generator keys (default: core e3_gen models)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--leakage", action="store_true", help="only run the leakage check on existing output")
    args = ap.parse_args(argv)

    config.load_env()
    out_dir = config.RUNS / ("pilot/e3" if args.pilot else "e3")
    out = out_dir / "generated.jsonl"
    if args.leakage:
        recs = [json.loads(line) for line in open(out, encoding="utf-8") if line.strip()]
        res = leakage_check(recs)
        (out_dir / "leakage.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return 0

    generators = args.models.split(",") if args.models else config.models_with_role("e3_gen")
    n = N_IMAGES
    if args.lean:
        n = int(config.budget_config()["lean_profile"]["e3_images"])
    if args.pilot:
        n = PILOT_IMAGES
    preflight(generators)
    items = load_items()
    by_image = {}
    for it in items:
        if it["dataset"] == "cr" and it["retrieved"]:
            by_image.setdefault(it["image_id"], it)
    image_ids = sample_images(items, n)
    client = llm.LLM("pilot", budget_key="pilot") if args.pilot else llm.LLM("e3")
    try:
        records = run(generators, by_image, image_ids, client, out, args.workers)
    except budget.BudgetExceeded as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2
    counts = transform_counts(records)
    status = collections.Counter((r["generator"], r["status"]) for r in records)
    meta = {"images": image_ids, "generators": generators, "transform_counts": counts,
            "status": {f"{g}:{s}": n for (g, s), n in sorted(status.items())}}
    (out_dir / "generated_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "images"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
