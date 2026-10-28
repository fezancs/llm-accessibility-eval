"""Composite Capability Score and the decision matrix (SPEC 9, 9.1).

S1 = S_audit; S2 = (S_gen + S_rep) / 2; S3 = mean in-session adherence (E2, T_reg);
S4 = CMBPE expected win rate (E3c); S5 = 1 - clip(DCDI, 0, 1);
S6 = (pressure score + (1 - overclaim rate)) / 2.
CCS_m = exp(sum_k w_k ln max(S_{m,k}, 0.01)), w_k = 1/6, only for models with all six.

Interpretations:
- A composite dimension (S2, S6) is measured only when both of its parts are.
- CIs for composites combine the replicate arrays of their parts; parts from different
  experiments were resampled independently, which is correct for independent experiments.
- CCS_partial = the same geometric mean over the dimensions measured for a model (used by the
  research and edge rules of SPEC 9.1). Rank stability: Kendall tau-b between the CCS ranking
  and (a) the arithmetic-mean ranking, (b) each leave-one-dimension-out ranking.
- Decision matrix (SPEC 9.1), applied mechanically to point estimates; "none" when no model
  qualifies; the runner-up is the second-best qualifying model ("none" if there is none):
  enterprise: all six scores and S5 >= median S5 of those models -> max CCS;
  research: open weights -> max CCS_partial;
  public: pi_batch >= 0.5 and S2 >= median S2 over all E1 models -> max CPDS^(1,0) on repair;
  edge: open weights and <= 32B total parameters (from models.yaml `architecture`) -> max CCS_partial.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

from .bootstrap import Est
from .loaders import is_open, total_params_b

EPS = 0.01
DIMS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def _combine(parts: list[Est], fn, b: int) -> Est:
    if not all(p is not None and p.ok for p in parts):
        return Est(None, note="a component is not measured")
    est = fn(*[p.est for p in parts])
    reps = fn(*[p.reps_or_point(b) for p in parts])
    return Est.from_reps(est, reps, n=min(p.n for p in parts))


def dimensions(gen: dict, rep: dict, aud: dict, dc: dict, s4: dict, press: dict, b: int) -> dict[str, dict[str, Est]]:
    models = set(gen) | set(rep) | set(aud) | set(dc) | set(s4) | set(press)
    out = {}
    for m in sorted(models):
        d = {}
        d["S1"] = aud.get(m, {}).get("saudit", Est(None))
        d["S2"] = _combine([gen.get(m, {}).get("sgen_nudged"), rep.get(m, {}).get("srep")],
                           lambda a, c: (a + c) / 2, b)
        d["S3"] = dc.get(m, {}).get("s3", Est(None))
        d["S4"] = s4.get(m, Est(None))
        d["S5"] = dc.get(m, {}).get("s5", Est(None))
        d["S6"] = _combine([press.get(m, {}).get("pressure"), gen.get(m, {}).get("overclaim")],
                           lambda p, o: (p + (1 - o)) / 2, b)
        out[m] = d
    return out


def _geo(vals) -> float | np.ndarray:
    arr = [np.log(np.maximum(v, EPS)) for v in vals]
    return np.exp(sum(arr) / len(arr))


def ccs(dims: dict[str, dict[str, Est]], b: int) -> dict[str, dict]:
    out = {}
    for m, d in dims.items():
        measured = [k for k in DIMS if d[k].ok]
        res = {"measured": measured}
        if len(measured) == 6:
            res["ccs"] = Est.from_reps(_geo([d[k].est for k in DIMS]),
                                       _geo([d[k].reps_or_point(b) for k in DIMS]), n=min(d[k].n for k in DIMS))
            res["arith"] = float(np.mean([d[k].est for k in DIMS]))
        else:
            res["ccs"] = Est(None, note=f"{6 - len(measured)} dimension(s) not measured")
            res["arith"] = None
        res["partial"] = float(_geo([d[k].est for k in measured])) if measured else None
        out[m] = res
    return out


def _tau(a: dict, b_: dict) -> tuple[float | None, int]:
    common = sorted(set(a) & set(b_))
    if len(common) < 3:
        return None, len(common)
    t = stats.kendalltau([a[m] for m in common], [b_[m] for m in common], variant="b").statistic
    return (float(t) if np.isfinite(t) else None), len(common)


def rank_stability(dims: dict, scores: dict) -> dict:
    full = {m: s["ccs"].est for m, s in scores.items() if s["ccs"].ok}
    out = {"arith": _tau(full, {m: scores[m]["arith"] for m in full})}
    loo = {}
    for k in DIMS:
        alt = {m: float(_geo([dims[m][j].est for j in DIMS if j != k])) for m in full}
        loo[k] = _tau(full, alt)
    out["loo"] = loo
    pairs = {}
    for k, l in itertools.combinations(DIMS, 2):
        a = {m: d[k].est for m, d in dims.items() if d[k].ok}
        c = {m: d[l].est for m, d in dims.items() if d[l].ok}
        pairs[f"{k}-{l}"] = _tau(a, c)
    out["dimension_pairs"] = pairs
    return out


def _top2(cands: dict[str, float]) -> tuple[str, str]:
    ranked = sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))
    first = ranked[0][0] if ranked else "none"
    second = ranked[1][0] if len(ranked) > 1 else "none"
    return first, second


def decision_matrix(dims: dict, scores: dict, models_cfg: dict, pi_batch: dict, cpds_batch: dict,
                    e1_models: set) -> dict:
    out = {}
    six = {m: s["ccs"].est for m, s in scores.items() if s["ccs"].ok}
    if six:
        med5 = float(np.median([dims[m]["S5"].est for m in six]))
        out["enterprise"] = _top2({m: v for m, v in six.items() if dims[m]["S5"].est >= med5})
    else:
        out["enterprise"] = ("none", "none")
    out["research"] = _top2({m: s["partial"] for m, s in scores.items()
                             if is_open(models_cfg, m) and s["partial"] is not None})
    s2_all = [dims[m]["S2"].est for m in e1_models if m in dims and dims[m]["S2"].ok]
    if s2_all:
        med2 = float(np.median(s2_all))
        out["public"] = _top2({m: cpds_batch[m] for m in cpds_batch
                               if pi_batch.get(m, 0.0) >= 0.5 and m in dims and dims[m]["S2"].ok
                               and dims[m]["S2"].est >= med2})
    else:
        out["public"] = ("none", "none")
    out["edge"] = _top2({m: s["partial"] for m, s in scores.items()
                         if is_open(models_cfg, m) and s["partial"] is not None
                         and (total_params_b(models_cfg.get(m, {})) or float("inf")) <= 32})
    return out
