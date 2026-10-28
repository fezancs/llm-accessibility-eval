"""Shared plumbing for the experiment scripts: model selection by profile, parsing model
output, JSONL writing, and a process pool that evaluates rendered artifacts with one
Chromium per worker and an on-disk result cache (so evaluation is resumable too)."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import re

from . import config

EVAL_VERSION = "2026-10-28.1"     # bump when scoring code changes, so cached evaluations are redone
PILOT_MODELS = ["claude-haiku-5-5", "gpt-6-luna", "qwen3-8-27b"]


def cli(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--pilot", action="store_true", help="CLAUDE.md Phase 2 pilot (logs to runs/pilot)")
    ap.add_argument("--lean", action="store_true", help="lean profile from budget.yaml")
    ap.add_argument("--optional", action="store_true", help="also run optional-tier models")
    ap.add_argument("--models", help="comma-separated model keys (overrides the profile)")
    ap.add_argument("--workers", type=int, default=6, help="browser worker processes for evaluation")
    ap.add_argument("--threads", type=int, default=16, help="concurrent API requests")
    ap.add_argument("--no-eval", action="store_true", help="only make the API calls")
    ap.add_argument("--tasks", help="E1-Gen: comma-separated task ids (default: all, or the pilot subset)")
    ap.add_argument("--samples", type=int, help="E1-Gen: samples per task and condition (default 3; pilot 1)")
    ap.add_argument("--pages", help="E1-Repair/Audit: comma-separated faulty page ids (default: all 48)")
    return ap


def id_list(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def experiment(args, name: str) -> str:
    return "pilot" if args.pilot else name


def select_models(args, role: str, lean_key: str | None = None) -> list[str]:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    if args.pilot:
        return [m for m in PILOT_MODELS if role in config.model(m)["roles"]]
    if args.lean or config.budget_config().get("profile") == "lean":
        return list(config.budget_config()["lean_profile"][lean_key]) if lean_key else config.models_with_role(role)
    return config.models_with_role(role, include_optional=args.optional)


_HTML_RE = re.compile(r"```[ \t]*html[^\n]*\n(.*?)```", re.S | re.I)


def extract_html(text: str) -> str | None:
    """The first ```html block (SPEC 5.1); None if there is no complete block."""
    m = _HTML_RE.search(text or "")
    return m.group(1).strip() + "\n" if m else None


def parse_json(text: str, expect: type):
    """Parse a JSON value of type `expect` from a reply, tolerating a ```json fence."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    for candidate in (t, _outermost(t, expect)):
        if candidate is None:
            continue
        try:
            val = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(val, expect):
            return val
    return None


def _outermost(t: str, expect: type) -> str | None:
    o, c = ("[", "]") if expect is list else ("{", "}")
    i, j = t.find(o), t.rfind(c)
    return t[i:j + 1] if i >= 0 and j > i else None


def write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def link_images(folder: pathlib.Path) -> None:
    """Artifacts render relative image paths from the corpus image folder."""
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / "images"
    if not link.exists():
        link.symlink_to(config.DATA / "corpus" / "base" / "images", target_is_directory=True)


# --------------------------------------------------------------------------- evaluation pool

_RENDERER = None


def _init_worker():
    global _RENDERER
    from .render import Renderer
    _RENDERER = Renderer(config.ROOT)


def _run_job(job):
    fn, cache_path, args = job
    import importlib
    mod, name = fn.rsplit(".", 1)
    out = getattr(importlib.import_module(mod), name)(_RENDERER, *args)
    tmp = pathlib.Path(str(cache_path) + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cache_path)
    return out


def cache_key(*parts) -> str:
    h = hashlib.sha256(EVAL_VERSION.encode())
    for p in parts:
        h.update(b"\x00")
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()


def evaluate_all(jobs: list[tuple[str, str, tuple]], cache_dir: pathlib.Path, workers: int) -> dict[str, dict]:
    """jobs: (key, "module.function", args). The function is called as f(renderer, *args)
    in a worker process; results are cached under cache_dir/<key>.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    results, todo = {}, []
    for key, fn, args in jobs:
        p = cache_dir / f"{key}.json"
        if p.exists():
            results[key] = json.loads(p.read_text(encoding="utf-8"))
        else:
            todo.append((key, (fn, p, args)))
    if todo:
        with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = {ex.submit(_run_job, job): key for key, job in todo}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                key = futs[fut]
                try:
                    results[key] = fut.result()
                except Exception as exc:        # noqa: BLE001 - one broken artifact must not stop the run
                    results[key] = {"eval_error": f"{type(exc).__name__}: {exc}"}
                if i % 10 == 0 or i == len(todo):
                    print(f"  evaluated {i}/{len(todo)}", flush=True)
    return results
