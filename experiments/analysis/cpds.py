"""Capability-per-Dollar-Second and Pareto analysis (SPEC 8).

For model m and task family f in {gen (nudged), rep}:
  CS = S_gen (nudged) or S_rep (mean instance score, as in metrics.py)
  C  = mean billed USD per task instance (record cost_usd: all attempts of the request)
  L  = median latency per task instance (record latency_s)
  CPDS^(a,b) = CS / ((1 + log10(1 + C/c0))^a * (1 + log10(1 + L/l0))^b), c0 = $0.01, l0 = 1 s.

Interpretations:
- Scenario -> task family: batch remediation (1, 0) on repair; interactive assistant (0.5, 1)
  on generation (Table III); balanced (1, 1) is reported for both families.
- Pareto dominance uses all three objectives (CS up, C down, L down) as SPEC 8 states; the
  batch-remediation frontier is that Pareto set on the repair family. pi_m is the share of
  B resamples (task instances resampled within each model, independently) in which m is
  non-dominated.
- Instances are the records that enter CS (gen: nudged condition). Records without a
  latency are kept for CS and C and skipped for the median latency.
- Rank stability: Kendall tau-b between the CPDS ranking at the default (c0, l0) and at each
  other grid point, per scenario.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .bootstrap import B_DEFAULT, SEED, seed_for
from .metrics import gen_table

C0, L0 = 0.01, 1.0
SCENARIOS = {"batch": (1.0, 0.0, "rep"), "interactive": (0.5, 1.0, "gen"),
             "balanced-rep": (1.0, 1.0, "rep"), "balanced-gen": (1.0, 1.0, "gen")}
GRID_C0 = (0.001, 0.01, 0.1)
GRID_L0 = (0.5, 1.0, 5.0)


def cpds_value(cs: float, c: float, l: float, alpha: float, beta: float, c0: float = C0, l0: float = L0) -> float:
    return cs / ((1 + np.log10(1 + c / c0)) ** alpha * (1 + np.log10(1 + l / l0)) ** beta)


def instances(gen: pd.DataFrame, rep: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    if not gen.empty:
        g = gen_table(gen)
        g = g[g["condition"] == "nudged"]
        out["gen"] = pd.DataFrame({"model_key": g["model_key"], "score": g["score"],
                                   "cost": pd.to_numeric(g.get("cost_usd"), errors="coerce"),
                                   "latency": pd.to_numeric(g.get("latency_s"), errors="coerce")})
    if not rep.empty:
        out["rep"] = pd.DataFrame({"model_key": rep["model_key"],
                                   "score": pd.to_numeric(rep.get("score"), errors="coerce").fillna(0.0),
                                   "cost": pd.to_numeric(rep.get("cost_usd"), errors="coerce"),
                                   "latency": pd.to_numeric(rep.get("latency_s"), errors="coerce"),
                                   "page_tokens": pd.to_numeric(rep.get("page_tokens"), errors="coerce")})
    return out


def summarize(inst: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for m, d in inst.groupby("model_key"):
        cost = d["cost"].dropna()
        lat = d["latency"].dropna()
        out[m] = {"cs": float(d["score"].mean()), "c": float(cost.mean()) if len(cost) else None,
                  "l": float(lat.median()) if len(lat) else None, "n": int(len(d)),
                  "n_cost": int(len(cost)), "n_latency": int(len(lat))}
        if "page_tokens" in d:
            pt = d["page_tokens"].dropna()
            out[m]["page_tokens"] = float(pt.mean()) if len(pt) else None
    return out


def non_dominated(points: dict[str, tuple[float, float, float]]) -> set[str]:
    """Models not dominated on (CS up, C down, L down)."""
    keep = set()
    for m, (cs, c, l) in points.items():
        dominated = False
        for o, (cs2, c2, l2) in points.items():
            if o == m:
                continue
            if cs2 >= cs and c2 <= c and l2 <= l and (cs2 > cs or c2 < c or l2 < l):
                dominated = True
                break
        if not dominated:
            keep.add(m)
    return keep


def pareto_pi(inst: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED, label: str = "pareto") -> dict[str, float]:
    """Pareto-membership probability per model; resamples instances within each model."""
    models = {}
    for m, d in inst.groupby("model_key"):
        d = d.dropna(subset=["cost", "latency"])
        if len(d):
            models[m] = (d["score"].to_numpy(float), d["cost"].to_numpy(float), d["latency"].to_numpy(float))
    if not models:
        return {}
    rng = np.random.default_rng(seed_for(label, seed))
    reps = {}
    for m, (s, c, l) in models.items():
        idx = rng.integers(0, len(s), size=(b, len(s)))
        reps[m] = (s[idx].mean(axis=1), c[idx].mean(axis=1), np.median(l[idx], axis=1))
    hits = {m: 0 for m in models}
    for i in range(b):
        pts = {m: (r[0][i], r[1][i], r[2][i]) for m, r in reps.items()}
        for m in non_dominated(pts):
            hits[m] += 1
    return {m: hits[m] / b for m in models}


def scores(summ: dict[str, dict], alpha: float, beta: float, c0: float = C0, l0: float = L0) -> dict[str, float]:
    out = {}
    for m, s in summ.items():
        if s["c"] is None or (beta > 0 and s["l"] is None):
            continue
        out[m] = float(cpds_value(s["cs"], s["c"], s["l"] or 0.0, alpha, beta, c0, l0))
    return out


def sensitivity(summs: dict[str, dict]) -> dict[str, dict]:
    """Kendall tau-b of each grid point's ranking against the default (c0=0.01, l0=1)."""
    out = {}
    for name, (alpha, beta, fam) in SCENARIOS.items():
        if fam not in summs:
            continue
        base = scores(summs[fam], alpha, beta)
        models = sorted(base)
        if len(models) < 3:
            continue
        taus = {}
        for c0, l0 in itertools.product(GRID_C0, GRID_L0):
            alt = scores(summs[fam], alpha, beta, c0, l0)
            common = [m for m in models if m in alt]
            tau = stats.kendalltau([base[m] for m in common], [alt[m] for m in common]).statistic
            taus[f"{c0:g}/{l0:g}"] = float(tau)
        vals = list(taus.values())
        out[name] = {"taus": taus, "min": float(np.min(vals)), "median": float(np.median(vals))}
    return out


def analyse(gen: pd.DataFrame, rep: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    inst = instances(gen, rep)
    summs = {fam: summarize(df) for fam, df in inst.items()}
    out = {"summary": summs, "cpds": {}, "pi": {}, "frontier": {}, "sensitivity": sensitivity(summs),
           "affordability": {}}
    for name, (alpha, beta, fam) in SCENARIOS.items():
        if fam in summs:
            out["cpds"][name] = scores(summs[fam], alpha, beta)
    for fam, df in inst.items():
        out["pi"][fam] = pareto_pi(df, b, seed, f"pareto-{fam}")
        pts = {m: (s["cs"], s["c"], s["l"]) for m, s in summs[fam].items() if s["c"] is not None and s["l"] is not None}
        out["frontier"][fam] = sorted(non_dominated(pts)) if pts else []
    if "rep" in summs:
        for m, s in summs["rep"].items():
            if s["c"] is not None:
                out["affordability"][m] = {"cost_1000_pages": 1000 * s["c"], "page_tokens": s.get("page_tokens")}
    return out
