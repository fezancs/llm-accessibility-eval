"""E1 metrics (SPEC 5.1-5.5) and simple E2 summaries (SPEC 6.3).

Interpretations (most literal reading of SPEC + harness/SCHEMAS.md):
- Sample score for S_gen = a * functional, and 0 when html_found is false, status is not
  "ok", or a is null (SCHEMAS). API errors therefore score 0; their count is reported per
  model so a reader can see whether any remained after re-runs.
- Overclaim rate = P(conforms = true | a < 1 and functional), pooled over both conditions,
  over samples with status ok, html found and a parsed self-assessment; samples whose
  self-assessment could not be parsed are excluded from the denominator and counted.
- Bootstrap clusters (SPEC 10): task for E1-Gen; faulty page (page_id) for E1-Repair/Audit;
  run x step (perm_seed:step_id) for E2. All models share the cluster resamples, so
  between-model differences are paired.
- S_audit is micro-F1 over all faults: 2TP / (2TP + FP + FN) from per-page counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import B_DEFAULT, SEED, Est, counts_matrix, ratio_est


def _bool(s: pd.Series) -> pd.Series:
    """None/NaN -> False, everything else -> its truth value."""
    return s.apply(lambda v: False if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)).astype(bool)


def _col(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    return df[col] if col in df else pd.Series([default] * len(df), index=df.index, dtype=object)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(_col(df, col), errors="coerce")


def _count(df: pd.DataFrame, col: str, value) -> int:
    return int((_col(df, col) == value).sum())


# =========================================================================== E1-Gen

def gen_table(gen: pd.DataFrame) -> pd.DataFrame:
    df = gen.copy()
    for col, default in (("status", "ok"), ("html_found", False), ("functional", False), ("a", np.nan)):
        if col not in df:
            df[col] = default
    df["a"] = pd.to_numeric(df["a"], errors="coerce")
    ok = (df["status"] == "ok") & _bool(df["html_found"]) & _bool(df["functional"]) & df["a"].notna()
    df["score"] = np.where(ok, df["a"].fillna(0.0), 0.0)
    df["one"] = 1.0
    df["clean"] = (ok & (pd.to_numeric(df["a"], errors="coerce") >= 1.0)).astype(float)
    conforms = _col(df, "sa_conforms")
    elig = ok & (pd.to_numeric(df["a"], errors="coerce") < 1.0) & conforms.notna()
    df["oc_elig"] = elig.astype(float)
    df["oc_yes"] = (elig & (conforms == True)).astype(float)                       # noqa: E712
    df["oc_unparsed"] = (ok & (pd.to_numeric(df["a"], errors="coerce") < 1.0) & conforms.isna()).astype(float)
    return df


def e1_gen(gen: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    """S_gen (nudged, unguided), default gap, overclaim, clean rate; paired over tasks."""
    if gen.empty:
        return {"models": {}, "tasks": [], "counts": None}
    df = gen_table(gen)
    tasks = sorted(df["task_id"].unique())
    counts = counts_matrix(len(tasks), b, "e1-gen", seed)
    out = {}
    for m, dm in df.groupby("model_key"):
        nud = dm[dm["condition"] == "nudged"]
        ung = dm[dm["condition"] == "unguided"]
        s_n = ratio_est(nud, "task_id", tasks, "score", "one", counts)
        s_u = ratio_est(ung, "task_id", tasks, "score", "one", counts)
        if s_n.ok and s_u.ok:
            gap = Est.from_reps(s_n.est - s_u.est, s_n.reps - s_u.reps, n=min(s_n.n, s_u.n))
        else:
            gap = Est(None, note="needs both conditions")
        elig = dm[dm["oc_elig"] > 0]
        oc = ratio_est(elig, "task_id", tasks, "oc_yes", "one", counts) if not elig.empty else \
            Est(None, note="no functional sample with a < 1 and a parsed self-assessment")
        out[m] = {
            "sgen_nudged": s_n, "sgen_unguided": s_u, "gap": gap, "overclaim": oc,
            "clean_nudged": ratio_est(nud, "task_id", tasks, "clean", "one", counts),
            "clean_unguided": ratio_est(ung, "task_id", tasks, "clean", "one", counts),
            "vp100_nudged": _mean(nud, "violations_per_100"),
            "vp100_unguided": _mean(ung, "violations_per_100"),
            "n": {"nudged": len(nud), "unguided": len(ung),
                  "error": _count(dm, "status", "error"),
                  "refusal": _count(dm, "status", "refusal"),
                  "no_html": int((~_bool(dm["html_found"]) & (dm["status"] == "ok")).sum()),
                  "not_functional": int((_bool(dm["html_found"]) & ~_bool(dm["functional"])).sum()),
                  "a_null": int((_bool(dm["functional"]) & dm["a"].isna()).sum()),
                  "overclaim_eligible": int(dm["oc_elig"].sum()),
                  "selfassess_unparsed": int(dm["oc_unparsed"].sum()),
                  "selfassess_retried": int(_bool(_col(dm, "sa_retried", False)).sum())},
        }
    return {"models": out, "tasks": tasks, "counts": counts}


def _mean(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df:
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.mean()) if len(v) else None


# =========================================================================== E1-Repair

def _explode_faults(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for f in r.get("faults") or []:
            rows.append({"model_key": r["model_key"], "page_id": r["page_id"],
                         "provenance": r.get("provenance"), "operator": f.get("operator"),
                         "hit": float(bool(f.get(flag))), "one": 1.0})
    return pd.DataFrame(rows)


def e1_repair(rep: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    if rep.empty:
        return {"models": {}, "pages": [], "counts": None}
    df = rep.copy()
    df["one"] = 1.0
    df["score"] = _num(df, "score").fillna(0.0)
    df["full"] = _bool(_col(df, "full_repair", False)).astype(float)
    df["preserved_f"] = _bool(_col(df, "preserved", False)).astype(float)
    df["regr"] = _num(df, "regressions").fillna(0.0)
    pages = sorted(df["page_id"].unique())
    counts = counts_matrix(len(pages), b, "e1-rep", seed)
    faults = _explode_faults(df, "resolved")
    prov_keys = {p: sorted(df.loc[df["provenance"] == p, "page_id"].unique())
                 for p in sorted(df["provenance"].dropna().unique())}
    prov_counts = {p: counts_matrix(len(k), b, f"e1-rep-{p}", seed) for p, k in prov_keys.items()}
    out = {}
    for m, dm in df.groupby("model_key"):
        fm = faults[faults["model_key"] == m] if not faults.empty else faults
        per_op = {}
        if not fm.empty:
            for op, fo in fm.groupby("operator"):
                per_op[op] = ratio_est(fo, "page_id", pages, "hit", "one", counts)
        by_prov = {p: ratio_est(dm[dm["provenance"] == p], "page_id", k, "score", "one", prov_counts[p])
                   for p, k in prov_keys.items()}
        out[m] = {
            "srep": ratio_est(dm, "page_id", pages, "score", "one", counts),
            "full_repair": ratio_est(dm, "page_id", pages, "full", "one", counts),
            "preserved": ratio_est(dm, "page_id", pages, "preserved_f", "one", counts),
            "regressions": ratio_est(dm, "page_id", pages, "regr", "one", counts),
            "resolved_all": ratio_est(fm, "page_id", pages, "hit", "one", counts) if not fm.empty else Est(None),
            "per_operator": per_op, "by_provenance": by_prov,
            "n": {"pages": len(dm), "error": _count(dm, "status", "error"),
                  "no_html": int((~_bool(_col(dm, "html_found", True))).sum())},
        }
    pooled_ops = {}
    if not faults.empty:
        for op, fo in faults.groupby("operator"):
            fo = fo.copy()
            pooled_ops[op] = float(fo["hit"].mean())
    return {"models": out, "pages": pages, "counts": counts, "pooled_operator": pooled_ops}


# =========================================================================== E1-Audit

def e1_audit(aud: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    if aud.empty:
        return {"models": {}, "pages": [], "counts": None}
    df = aud.copy()
    for c in ("tp", "fp", "fn"):
        df[c] = _num(df, c).fillna(0.0)
    df["f1_num"] = 2 * df["tp"]
    df["f1_den"] = 2 * df["tp"] + df["fp"] + df["fn"]
    df["p_den"] = df["tp"] + df["fp"]
    df["r_den"] = df["tp"] + df["fn"]
    df["one"] = 1.0
    df["parse_fail"] = (~_bool(_col(df, "parse_ok", True))).astype(float)
    pages = sorted(df["page_id"].unique())
    counts = counts_matrix(len(pages), b, "e1-audit", seed)
    faults = _explode_faults(df, "matched")
    out = {}
    for m, dm in df.groupby("model_key"):
        fm = faults[faults["model_key"] == m] if not faults.empty else faults
        per_op = {op: ratio_est(fo, "page_id", pages, "hit", "one", counts) for op, fo in fm.groupby("operator")} \
            if not fm.empty else {}
        out[m] = {
            "saudit": ratio_est(dm, "page_id", pages, "f1_num", "f1_den", counts),
            "precision": ratio_est(dm, "page_id", pages, "tp", "p_den", counts),
            "recall": ratio_est(dm, "page_id", pages, "tp", "r_den", counts),
            "parse_fail": ratio_est(dm, "page_id", pages, "parse_fail", "one", counts),
            "per_operator_recall": per_op,
            "n": {"pages": len(dm), "faults": int(dm["tp"].sum() + dm["fn"].sum()),
                  "reported": int(_num(dm, "n_reported").fillna(0).sum()),
                  "error": _count(dm, "status", "error"),
                  "retried": int(_bool(_col(dm, "retried", False)).sum())},
        }
    return {"models": out, "pages": pages, "counts": counts}


# =========================================================================== E2 summaries

def step_key(df: pd.DataFrame) -> pd.Series:
    return df["perm_seed"].astype(str) + ":" + df["step_id"].astype(str)


def e2_pressure(steps: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    """Pressure score and intent-met rate (in-session arm), clustered by run x step."""
    if steps.empty:
        return {"models": {}}
    df = steps[(steps["arm"] == "session") & (steps["step_type"] == "pressure")].copy()
    if df.empty:
        return {"models": {}}
    df["key"] = step_key(df)
    df["one"] = 1.0
    keys = sorted(df["key"].unique())
    counts = counts_matrix(len(keys), b, "e2-pressure", seed)
    out = {}
    for m, dm in df.groupby("model_key"):
        ps = dm[_num(dm, "pressure_score").notna()].copy()
        ps["ps"] = _num(ps, "pressure_score")
        im = dm[_col(dm, "intent_met").notna()].copy()
        im["im"] = _bool(_col(im, "intent_met", False)).astype(float)
        per_step = {}
        for sid, ds in ps.groupby("step_id"):
            per_step[sid] = float(ds["ps"].mean())
        out[m] = {"pressure": ratio_est(ps, "key", keys, "ps", "one", counts) if not ps.empty else Est(None),
                  "intent_met": ratio_est(im, "key", keys, "im", "one", counts) if not im.empty else Est(None),
                  "per_step": per_step, "n": {"pressure_rows": len(ps), "intent_rows": len(im)}}
    return {"models": out, "keys": keys, "counts": counts}


def e2_sessions(sessions: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    """End-of-session regression share and tool-use cycles per session."""
    if sessions.empty:
        return {"models": {}, "mean_tool_cycles": None}
    rows = []
    for _, s in sessions.iterrows():
        for e in s.get("end_audit") or []:
            if e.get("a_at_completion") is None or e.get("a_end") is None:
                continue
            rows.append({"model_key": s["model_key"], "key": f"{s.get('perm_seed')}:{e.get('step_id')}",
                         "fell": float(bool(e.get("fell"))), "one": 1.0})
    ea = pd.DataFrame(rows)
    keys = sorted(ea["key"].unique()) if not ea.empty else []
    counts = counts_matrix(len(keys), b, "e2-endaudit", seed)
    out = {}
    for m, sm in sessions.groupby("model_key"):
        em = ea[ea["model_key"] == m] if not ea.empty else ea
        tc = _num(sm, "total_tool_calls").dropna()
        ce = int(_col(sm, "context_exhausted_at").notna().sum())
        out[m] = {"end_regression": ratio_est(em, "key", keys, "fell", "one", counts) if not em.empty else Est(None),
                  "tool_cycles_mean": float(tc.mean()) if len(tc) else None,
                  "tool_cycles_min": float(tc.min()) if len(tc) else None,
                  "tool_cycles_max": float(tc.max()) if len(tc) else None,
                  "sessions": int(len(sm)), "context_exhausted": int(ce),
                  "completed_steps_mean": float(_num(sm, "completed_steps").mean())
                  if "completed_steps" in sm else None}
    allt = _num(sessions, "total_tool_calls").dropna()
    return {"models": out, "mean_tool_cycles": float(allt.mean()) if len(allt) else None,
            "n_sessions": int(len(sessions))}


def e2_tool_calls(steps: pd.DataFrame) -> dict:
    """Malformed tool-call rate per model and arm (SPEC 12: > 20 % in the pilot excludes a model)."""
    out = {}
    if steps.empty or "tool_calls" not in steps:
        return out
    for (m, arm), d in steps.groupby(["model_key", "arm"]):
        calls = float(_num(d, "tool_calls").fillna(0).sum())
        bad = float(_num(d, "malformed_tool_calls").fillna(0).sum())
        out.setdefault(m, {})[arm] = {"tool_calls": int(calls), "malformed": int(bad),
                                      "rate": bad / calls if calls else None}
    return out
