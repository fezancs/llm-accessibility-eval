"""Pre-registered hypotheses H1-H5 (SPEC 1): one-sided bootstrap tests, Holm across the family.

Statistics (positive values support the hypothesis):
- H1a/b/c: DCDI(smaller) - DCDI(larger), paired over run x step resamples. H1c is tested
  only if Gemini 4 Argon ran in E2 (SPEC); otherwise it is reported as not tested.
- H2: max S4 over closed generators - max S4 over open-weight generators, with the maxima
  re-taken in every image resample.
- H3: the pre-registered criterion is max_open pi >= 0.5 on the batch-repair frontier. Its
  p-value is the bootstrap share of resamples in which the best open-weight model (by pi)
  is dominated, (1 + B(1 - pi)) / (B + 1). The hypothesis counts as supported only if the
  criterion holds AND the Holm-adjusted p < 0.05 (CLAUDE.md rule 6), so pi must exceed ~0.95.
- H4a: one sub-test; per-judge one-sided p-values for eta_j > 0 are combined by Bonferroni
  (p = min(1, J * min_j p_j)) before entering the Holm family.
- H4b: CMBPE accuracy on known-answer pairs - median single-judge accuracy (same resamples).
- H5: number of models with S_gen(nudged) > S_gen(unguided), minus M/2; p = share of task
  resamples in which the count is not a strict majority.
A hypothesis is "supported" only when its Holm-adjusted p < 0.05.
"""
from __future__ import annotations

import numpy as np

from .bootstrap import ALPHA, Est, holm, p_greater
from .loaders import is_open

H1_PAIRS = {"H1a": ("claude-opus-5-5", "claude-haiku-5-5"),
            "H1b": ("gpt-6-sol", "gpt-6-luna"),
            "H1c": ("gemini-4-argon", "gemini-3-8-flash")}


def _p_from_pi(pi: float, b: int) -> float:
    return float((1 + round(b * (1 - pi))) / (b + 1))


def run(gen: dict, dc: dict, s4: dict, pi_batch: dict, eta: dict, h4b: Est | None, models_cfg: dict,
        b: int) -> dict:
    tests: dict[str, dict] = {}

    for name, (large, small) in H1_PAIRS.items():
        el, es = dc.get(large, {}).get("dcdi"), dc.get(small, {}).get("dcdi")
        if el is None or es is None or not (el.ok and es.ok):
            missing = [k for k, e in ((large, el), (small, es)) if e is None or not e.ok]
            tests[name] = {"tested": False, "note": "no E2 DCDI for " + ", ".join(missing)}
            continue
        diff = Est.from_reps(es.est - el.est, es.reps - el.reps, n=min(es.n, el.n))
        tests[name] = {"tested": True, "stat": diff, "p": p_greater(diff.reps),
                       "description": f"DCDI({small}) - DCDI({large})"}

    closed = [m for m, e in s4.items() if e.ok and not is_open(models_cfg, m)]
    opened = [m for m, e in s4.items() if e.ok and is_open(models_cfg, m)]
    if closed and opened:
        pt = max(s4[m].est for m in closed) - max(s4[m].est for m in opened)
        reps = np.nanmax(np.vstack([s4[m].reps_or_point(b) for m in closed]), axis=0) - \
            np.nanmax(np.vstack([s4[m].reps_or_point(b) for m in opened]), axis=0)
        st = Est.from_reps(pt, reps)
        tests["H2"] = {"tested": True, "stat": st, "p": p_greater(reps),
                       "best_closed": max(closed, key=lambda m: s4[m].est),
                       "best_open": max(opened, key=lambda m: s4[m].est)}
    else:
        tests["H2"] = {"tested": False, "note": "needs E3c scores for closed and open-weight generators"}

    open_pi = {m: p for m, p in pi_batch.items() if is_open(models_cfg, m)}
    if open_pi:
        best = max(open_pi, key=lambda m: (open_pi[m], m))
        tests["H3"] = {"tested": True, "best_open": best, "pi": open_pi[best],
                       "criterion_met": open_pi[best] >= 0.5, "p": _p_from_pi(open_pi[best], b)}
    else:
        tests["H3"] = {"tested": False, "note": "no open-weight model on the batch-repair data"}

    per_judge = {j: p_greater(e.reps) for j, e in eta.items() if e.ok and e.reps is not None}
    per_judge = {j: p for j, p in per_judge.items() if p is not None}
    if per_judge:
        tests["H4a"] = {"tested": True, "per_judge_p": per_judge,
                        "p": min(1.0, len(per_judge) * min(per_judge.values())),
                        "max_eta": max(eta[j].est for j in per_judge)}
    else:
        tests["H4a"] = {"tested": False, "note": "no self-preference estimates"}

    if h4b is not None and h4b.ok:
        tests["H4b"] = {"tested": True, "stat": h4b, "p": p_greater(h4b.reps)}
    else:
        tests["H4b"] = {"tested": False, "note": "needs CMBPE and single-judge accuracies on known pairs"}

    gaps = {m: r["gap"] for m, r in gen.items() if r["gap"].ok}
    if gaps:
        mm = len(gaps)
        count_pt = sum(1 for e in gaps.values() if e.est > 0)
        reps = np.sum(np.vstack([e.reps > 0 for e in gaps.values()]), axis=0).astype(float)
        stat = reps - mm / 2
        p = float((1 + np.sum(stat <= 0)) / (len(stat) + 1))
        tests["H5"] = {"tested": True, "count_positive": count_pt, "models": mm,
                       "majority": count_pt > mm / 2, "p": p}
    else:
        tests["H5"] = {"tested": False, "note": "no E1-Gen data in both conditions"}

    adj = holm({k: v["p"] for k, v in tests.items() if v.get("tested")})
    for k, v in tests.items():
        if not v.get("tested"):
            v.update({"p_holm": None, "supported": False})
            continue
        v["p_holm"] = adj[k]["p_holm"]
        ok = adj[k]["reject"]
        if k == "H3":
            ok = ok and v["criterion_met"]
        if k == "H5":
            ok = ok and v["majority"]
        v["supported"] = bool(ok)
    return {"tests": tests, "family_size": sum(1 for v in tests.values() if v.get("tested")), "alpha": ALPHA}
