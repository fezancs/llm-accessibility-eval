#!/usr/bin/env python3
"""Extra values for the local-run paper: serving speed and GPU memory, plus E1 diagnostics.

    .venv/bin/python local/extra_values.py [--out results/local-run-2026-11/results/extra_values.tex]

Hardware (study experiments e1, e2, e3; successful calls of the local models):
- decode-tps-<m>: median output_tokens / (total_s - ttft_s) over calls with >= 64 output tokens;
- prompt-tps-<m>: median uncached prompt tokens / ttft_s over calls with >= 1,000 uncached
  prompt tokens. The chat adapter logs input_tokens = prompt tokens minus those llama.cpp served
  from its prompt cache, and TTFT covers only the uncached ones, so this is the rate at which the
  server processed new prompt tokens. (Requiring a nearly uncached prompt left no Ministral call:
  its chat template's ~534-token system prefix is always cached.) Caveat: every E3 prompt of the
  vision models carries an image, so their TTFT includes image encoding; Gemma 3 4B and
  Ministral 3 8B, which run E3 only, are therefore not comparable with the text prompts of
  Qwen3 8B and (mostly E1) Qwen2.5-VL 7B;
- vram-mib-<m>: median VRAM in use (all processes, desktop included) over the 10 s samples of
  runs/local_pipeline/gpu_usage.csv (local/gpu_monitor.py) while the model was working.
E1 diagnostics (first attempt of each request; JSON retries excluded):
- capped-<family>-<m>: share of requests whose answer stopped at the output-token cap
  (families: gen, selfassess, repair, audit);
- unchanged-repair-<m>: repairs whose HTML equals the faulty page (whitespace-normalised),
  as a share of repairs that returned an HTML block (-count, -n);
- functional-gen-<m>: share of E1-Gen samples (both conditions) that pass the task's
  functional checks (-count, -n).
Values are \\setres{key}{value} lines, as in analysis/export_latex.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics

EXP = pathlib.Path(__file__).resolve().parent.parent
RUNS = EXP / "runs"
STUDY = ("e1", "e2", "e3")
FAMILIES = ("gen", "selfassess", "repair", "audit")


def norm(html: str) -> str:
    return re.sub(r"\s+", " ", html).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=pathlib.Path,
                    default=EXP / "results" / "local-run-2026-11" / "results" / "extra_values.tex")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load((EXP / "config" / "models.yaml").read_text(encoding="utf-8"))
    alias = {m["provider_model_id"]: m["key"] for m in cfg["models"] if m.get("provider") == "local"}

    decode, prompt, first = {}, {}, {}
    for exp in STUDY:
        p = RUNS / exp / "calls.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("provider") != "local" or r.get("status") != "ok":
                continue
            m = r["model_key"]
            out, total, ttft = r.get("output_tokens") or 0, r.get("total_s"), r.get("ttft_s")
            if out >= 64 and total and ttft is not None and total - ttft > 0:
                decode.setdefault(m, []).append(out / (total - ttft))
            n_new = r.get("input_tokens") or 0          # prompt tokens not served from the prompt cache
            if n_new >= 1000 and ttft:
                prompt.setdefault(m, []).append(n_new / ttft)
            if exp == "e1":
                fam = r["item_id"].split("/")[0]
                if fam in FAMILIES and not r["item_id"].endswith("/retry"):
                    first.setdefault((m, fam), {})[(r["item_id"], r.get("sample_idx"))] = r.get("stop_reason")

    vram = {}
    gpu = RUNS / "local_pipeline" / "gpu_usage.csv"
    if gpu.exists():
        for row in csv.DictReader(open(gpu, encoding="utf-8")):
            k = alias.get(row.get("model", ""))
            if k and row.get("state") in ("generating", "reading prompt") and row.get("vram_used_mib"):
                vram.setdefault(k, []).append(int(row["vram_used_mib"]))

    functional = {}                       # E1-Gen samples (both conditions) that pass the task's checks
    gen = RUNS / "e1" / "scores" / "gen.jsonl"
    if gen.exists():
        for line in open(gen, encoding="utf-8"):
            r = json.loads(line)
            f = functional.setdefault(r["model_key"], [0, 0])
            f[1] += 1
            f[0] += bool(r.get("functional"))

    unchanged = {}
    rep = RUNS / "e1" / "scores" / "repair.jsonl"
    if rep.exists():
        for line in open(rep, encoding="utf-8"):
            r = json.loads(line)
            art = r.get("artifact")
            if not r.get("html_found") or not art:
                continue
            path = pathlib.Path(art) if pathlib.Path(art).is_absolute() else RUNS / art
            src = EXP / "data" / "corpus" / "variants" / f"{r['page_id']}.html"
            if path.exists() and src.exists():
                u = unchanged.setdefault(r["model_key"], [0, 0])
                u[1] += 1
                u[0] += norm(path.read_text(encoding="utf-8")) == norm(src.read_text(encoding="utf-8"))

    lines = ["% Generated by experiments/local/extra_values.py. Do not edit by hand."]
    for m in sorted(set(decode) | set(prompt) | set(vram) | {k for k, _ in first} | set(unchanged)):
        if m in decode:
            lines += [f"\\setres{{decode-tps-{m}}}{{{statistics.median(decode[m]):.1f}}}",
                      f"\\setres{{decode-tps-{m}-n}}{{{len(decode[m])}}}"]
        if m in prompt:
            lines += [f"\\setres{{prompt-tps-{m}}}{{{statistics.median(prompt[m]):.0f}}}",
                      f"\\setres{{prompt-tps-{m}-n}}{{{len(prompt[m])}}}"]
        if m in vram:
            lines += [f"\\setres{{vram-mib-{m}}}{{{statistics.median(vram[m]):.0f}}}",
                      f"\\setres{{vram-mib-{m}-max}}{{{max(vram[m])}}}"]
        for fam in FAMILIES:
            reqs = first.get((m, fam))
            if reqs:
                capped = sum(1 for s in reqs.values() if s == "max_tokens")
                lines += [f"\\setres{{capped-{fam}-{m}}}{{{capped / len(reqs):.2f}}}",
                          f"\\setres{{capped-{fam}-{m}-count}}{{{capped}}}",
                          f"\\setres{{capped-{fam}-{m}-n}}{{{len(reqs)}}}"]
        if m in functional and functional[m][1]:
            f_ok, f_n = functional[m]
            lines += [f"\\setres{{functional-gen-{m}}}{{{f_ok / f_n:.2f}}}",
                      f"\\setres{{functional-gen-{m}-count}}{{{f_ok}}}",
                      f"\\setres{{functional-gen-{m}-n}}{{{f_n}}}"]
        if m in unchanged and unchanged[m][1]:
            u, n = unchanged[m]
            lines += [f"\\setres{{unchanged-repair-{m}}}{{{u / n:.2f}}}",
                      f"\\setres{{unchanged-repair-{m}-count}}{{{u}}}",
                      f"\\setres{{unchanged-repair-{m}-n}}{{{n}}}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
