"""Phase 2 pilot report (CLAUDE.md): checks and cost projection from the pilot logs.

    python -m harness.pilot_report          # writes runs/pilot_cost_projection.md

Reads runs/pilot/calls.jsonl and the pilot score files. Reports parse-failure rates,
tool-call validity per model (SPEC 12: > 20 % malformed -> exclude from E2), latency
capture, prompt-cache hits and cost per call, then projects the cost of the full and lean
profiles per experiment and model against the caps in budget.yaml.

Projection method (stated in the report): per unit of work (one generation, one repair,
one audit, one E2 step per arm, one judgment, one alt text) the pilot gives billed cost and
token counts. Pilot models are projected from their own measured cost. Other models are
projected from the mean token counts of the pilot models of the same vendor if one ran,
otherwise of all pilot models, priced at the model's own models.yaml prices; reasoning-heavy
models may use more tokens, so treat those rows as rough. E2 steps get longer as a session
grows: session and padded costs are extrapolated from positions 1-6 with a linear fit in
position, which assumes the history grows at the pilot's rate.
"""
from __future__ import annotations

import collections
import json
import statistics

import numpy as np

from . import budget, config

PILOT = config.RUNS / "pilot"
OUT = config.RUNS / "pilot_cost_projection.md"


def _jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def component(item_id: str) -> str:
    """Map a call's item_id to the unit of work it belongs to."""
    if item_id.startswith("gen/"):
        return "e1_gen"
    if item_id.startswith("selfassess"):
        return "e1_selfassess"
    if item_id.startswith("repair/"):
        return "e1_repair"
    if item_id.startswith("audit"):
        return "e1_audit"
    if item_id.startswith("e2/padding/"):
        return "e2_padding_transcript"
    if item_id.startswith("e2/"):
        for arm in ("session", "fresh", "padded"):
            if f"/{arm}/" in item_id:
                return f"e2_{arm}"
        return "e2_other"
    if item_id.startswith("e3/alt/"):
        return "e3_alt"
    for proc in ("pointwise", "known", "blv", "gen"):
        if item_id.startswith(f"e3/{proc}/"):
            return f"e3_{proc}"
    return "other"


def unit_key(rec: dict) -> str:
    """Calls belonging to one unit of work (retries and follow-ups share it)."""
    iid = rec["item_id"]
    if iid.startswith("e2/"):
        return iid.split("/call")[0]                    # one step in one arm
    return iid.removesuffix("/retry") + f"#{rec.get('sample_idx', 0)}"


def unit_stats(calls: list[dict]) -> dict:
    """Per (component, model): number of units and mean cost/tokens per unit."""
    units = collections.defaultdict(lambda: collections.defaultdict(float))
    for c in calls:
        u = units[(component(c["item_id"]), c["model_key"], unit_key(c))]
        u["cost"] += c.get("cost_usd") or 0.0
        for k in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens"):
            u[k] += c.get(k) or 0
    out = collections.defaultdict(lambda: {"n": 0, "cost": 0.0, "input_tokens": 0.0, "cached_input_tokens": 0.0,
                                           "cache_write_tokens": 0.0, "output_tokens": 0.0})
    for (comp, model, _), u in units.items():
        s = out[(comp, model)]
        s["n"] += 1
        for k in ("cost", "input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens"):
            s[k] += u[k]
    for s in out.values():
        for k in ("cost", "input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens"):
            s[k] /= max(s["n"], 1)
    return dict(out)


def priced(model: str, tokens: dict) -> float | None:
    try:
        usage = {k: tokens.get(k, 0) for k in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")}
        import datetime as dt
        return budget.cost_usd(model, usage, dt.datetime(2026, 11, 2, 12, tzinfo=dt.timezone.utc))[0]
    except budget.PriceMissing:
        return None


def per_unit_cost(stats: dict, comp: str, model: str) -> tuple[float | None, str]:
    if (comp, model) in stats and stats[(comp, model)]["n"]:
        return stats[(comp, model)]["cost"], "measured"
    fam = config.model(model)["family"]
    donors = [s for (c, m), s in stats.items() if c == comp and config.model(m)["family"] == fam] or \
             [s for (c, m), s in stats.items() if c == comp]
    if not donors:
        return None, "no pilot data"
    tokens = {k: statistics.mean(d[k] for d in donors) for k in ("input_tokens", "cached_input_tokens",
                                                                  "cache_write_tokens", "output_tokens")}
    c = priced(model, tokens)
    if c is None:
        return None, "price missing"
    same_vendor = any(config.model(m)["family"] == fam for (cc, m) in stats if cc == comp)
    return c, "token-projected (same vendor)" if same_vendor else "token-projected (all pilot models)"


def e2_growth(steps: list[dict]) -> dict:
    """Per (model, arm): linear fit of step cost on position from the pilot (positions 1-6)."""
    fits = {}
    by = collections.defaultdict(list)
    for s in steps:
        if s.get("cost_usd") is not None:
            by[(s["model_key"], s["arm"])].append((s["position"], s["cost_usd"]))
    for k, pts in by.items():
        x, y = np.array([p for p, _ in pts], float), np.array([c for _, c in pts], float)
        if len(set(x)) >= 2:
            b, a = np.polyfit(x, y, 1)
            fits[k] = (float(a), float(max(b, 0.0)))
        else:
            fits[k] = (float(y.mean()), 0.0)
    return fits


def main():
    calls = _jsonl(PILOT / "calls.jsonl")
    if not calls:
        raise SystemExit("no pilot calls logged (runs/pilot/calls.jsonl); run the pilot first")
    caps = config.budget_config()["caps"]
    lean = config.budget_config()["lean_profile"]
    stats = unit_stats([c for c in calls if c.get("status") != "error" or c.get("cost_usd")])
    lines = ["# Pilot report and cost projection", "",
             f"Pilot calls (attempts): {len(calls)}; billed: ${sum(c.get('cost_usd') or 0 for c in calls):.2f} "
             f"(cap ${caps['pilot']}). Price snapshot: {config.models_config().get('snapshot_date')}.", "",
             "## Checks", ""]
    # -- parse failures
    gen, aud, rep = (_jsonl(PILOT / "scores" / n) for n in ("gen.jsonl", "audit.jsonl", "repair.jsonl"))
    lines.append("| Model | E1 no html block | self-assess unparsed | repair no html | audit unparsed | API errors (attempts) |")
    lines.append("|---|---|---|---|---|---|")
    for m in sorted({c["model_key"] for c in calls}):
        g = [r for r in gen if r["model_key"] == m]
        a = [r for r in aud if r["model_key"] == m]
        rp = [r for r in rep if r["model_key"] == m]
        err = sum(1 for c in calls if c["model_key"] == m and c.get("status") == "error")
        frac = lambda xs, f: f"{sum(f(x) for x in xs)}/{len(xs)}" if xs else "--"
        lines.append(f"| {m} | {frac(g, lambda r: not r.get('html_found'))} | "
                     f"{frac([r for r in g if r.get('html_found')], lambda r: not r['selfassess']['parse_ok'])} | "
                     f"{frac(rp, lambda r: not r.get('html_found'))} | {frac(a, lambda r: not r.get('parse_ok'))} | {err} |")
    # -- tool calls
    steps = _jsonl(PILOT / "scores" / "steps.jsonl")
    lines += ["", "E2 tool-call validity (session and fresh arms; SPEC 12: more than 20 % malformed -> exclude from E2):", "",
              "| Model | tool calls | malformed | rate | rule |", "|---|---|---|---|---|"]
    for m in sorted({s["model_key"] for s in steps}):
        tc = sum(s.get("tool_calls") or 0 for s in steps if s["model_key"] == m)
        bad = sum(s.get("malformed_tool_calls") or 0 for s in steps if s["model_key"] == m)
        rate = bad / tc if tc else None
        lines.append(f"| {m} | {tc} | {bad} | {'' if rate is None else f'{rate:.1%}'} | "
                     f"{'EXCLUDE (ask the author)' if rate is not None and rate > 0.2 else 'ok'} |")
    # -- latency and caching
    ok = [c for c in calls if c.get("status") == "ok"]
    lines += ["", f"Latency captured: total_s on {sum(c.get('total_s') is not None for c in ok)}/{len(ok)} successful calls; "
              f"ttft_s on {sum(c.get('ttft_s') is not None for c in ok)}/{len(ok)}.", "",
              "| Provider | prompt tokens | cached share | cache writes |", "|---|---|---|---|"]
    for p in sorted({c["provider"] for c in ok}):
        cs = [c for c in ok if c["provider"] == p]
        tot = sum((c.get("input_tokens") or 0) + (c.get("cached_input_tokens") or 0) + (c.get("cache_write_tokens") or 0) for c in cs)
        cached = sum(c.get("cached_input_tokens") or 0 for c in cs)
        lines.append(f"| {p} | {tot} | {cached / tot:.1%} | {sum(c.get('cache_write_tokens') or 0 for c in cs)} |" if tot else f"| {p} | 0 | -- | -- |")
    lines += ["", "Mean cost per call by component and model:", "", "| Component | Model | units | $ per unit |", "|---|---|---|---|"]
    for (comp, m), s in sorted(stats.items()):
        lines.append(f"| {comp} | {m} | {s['n']} | {s['cost']:.4f} |")

    # -- projection
    def project(profile: str) -> tuple[list[str], dict]:
        totals = collections.defaultdict(float)
        rows = []
        e1_models = lean["e1_models"] if profile == "lean" else config.models_with_role("e1")
        e2_models = lean["e2_models"] if profile == "lean" else config.models_with_role("e2")
        e2_runs = lean["e2_runs"] if profile == "lean" else 3
        for m in e1_models:
            for comp, n in (("e1_gen", 120), ("e1_selfassess", 120), ("e1_repair", 48), ("e1_audit", 48)):
                c, how = per_unit_cost(stats, comp, m)
                if c is not None:
                    totals["e1"] += c * n
                rows.append(f"| E1 | {m} | {comp} | {n} | {'--' if c is None else f'{c * n:.2f}'} | {how} |")
        fits = e2_growth(steps)
        for m in e2_models:
            for arm, positions in (("session", range(1, 25)), ("fresh", range(1, 25)), ("padded", (8, 12, 16, 20, 24))):
                fit = fits.get((m, "session" if arm == "padded" else arm))
                if fit is not None:
                    c, how = sum(fit[0] + fit[1] * p for p in positions) * e2_runs, "measured, linear in position"
                else:
                    unit, how = per_unit_cost(stats, f"e2_{'session' if arm == 'padded' else arm}", m)
                    c = None if unit is None else unit * len(list(positions)) * e2_runs * 2.0
                    how += "; x2 for longer sessions" if unit is not None else ""
                if c is not None:
                    totals["e2"] += c
                rows.append(f"| E2 | {m} | {arm} x{e2_runs} runs | {len(list(positions)) * e2_runs} | "
                            f"{'--' if c is None else f'{c:.2f}'} | {how} |")
        try:
            from .e3_judge import plan_counts
            judges = list(config.models_config()["judges"].values())
            plan = plan_counts(judges, ["pointwise", "known", "blv", "gen"], False, profile == "lean")
        except Exception as exc:                # noqa: BLE001
            plan, rows = {}, rows + [f"| E3 | -- | plan unavailable: {exc} | | | |"]
        comp_of = {"pointwise": "e3_pointwise", "known_pairs": "e3_known", "known": "e3_known",
                   "blv_pairs": "e3_blv", "blv": "e3_blv", "gen_pairs": "e3_gen", "gen": "e3_gen"}
        for proc, n_total in plan.items():
            comp = comp_of.get(proc)
            if comp is None or not isinstance(n_total, (int, float)):
                continue
            judges = list(config.models_config()["judges"].values())
            for j in judges:
                c, how = per_unit_cost(stats, comp, j)
                n = n_total / len(judges)
                if c is not None:
                    totals["e3"] += c * n
                rows.append(f"| E3 | {j} | {comp} | {n:.0f} | {'--' if c is None else f'{c * n:.2f}'} | {how} |")
        n_img = lean["e3_images"] if profile == "lean" else 120
        for g in config.models_with_role("e3_gen"):
            c, how = per_unit_cost(stats, "e3_alt", g)
            if c is not None:
                totals["e3"] += c * n_img
            rows.append(f"| E3 | {g} | e3_alt | {n_img} | {'--' if c is None else f'{c * n_img:.2f}'} | {how} |")
        return rows, totals

    for profile in ("full", "lean"):
        rows, totals = project(profile)
        lines += ["", f"## Projection: {profile} profile", "",
                  "| Exp. | Model | Unit | n | Projected $ | Basis |", "|---|---|---|---|---|---|", *rows, "",
                  "| Experiment | Projected $ | Cap $ |", "|---|---|---|"]
        cap_key = {"e1": "e1", "e2": "e2", "e3": "e3"}
        for e in ("e1", "e2", "e3"):
            lines.append(f"| {e.upper()} | {totals[e]:.2f} | {caps[cap_key[e]]} |")
        lines.append(f"| Total | {sum(totals.values()):.2f} | {sum(caps[k] for k in ('e1', 'e2', 'e3')) + caps['contingency']} (incl. contingency) |")
    lines += ["", "Rows marked token-projected are estimates for models the pilot did not run; check them before approving.",
              "Pilot data are not used in the paper (CLAUDE.md Phase 2)."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
