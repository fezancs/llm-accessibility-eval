"""Dynamic Contextual Drift Index and its companions (SPEC 6.3-6.4).

DCDI_m = sum_{r, t in T_reg} (a0 - a) / sum a0, where a is in-session adherence, a0 the
fresh-control adherence, and T_reg the foundation and regular steps (pressure excluded)
whose functional check passed in BOTH arms and whose adherence is defined in both.

Interpretations:
- Bootstrap cluster = run x step, keyed by perm_seed:step_id; the three permutations are the
  same for every model, so DCDI differences between models (H1) are paired.
- The length/history decomposition uses the padded checkpoints (positions 8, 12, 16, 20, 24)
  restricted to T_reg steps that are functional (and have defined adherence) in all three
  arms, so DCDI_K = DCDI_len + DCDI_hist holds exactly, in the estimate and in every replicate.
  Padded rows with length_matched = False (realised length outside Lambda +/- 10 %) are
  excluded and counted per model (drops: padded_length_unmatched).
- Drift rate beta: OLS of d = a0 - a on log2(Lambda / 8192) with step fixed effects over
  T_reg rows (Lambda = input tokens of the step's first in-session call), cluster-robust SEs
  by run; the CI uses t with G - 1 degrees of freedom (G = number of runs).
- S_C3 = mean in-session adherence over T_reg; S_C5 = 1 - clip(DCDI, 0, 1).
- The drift figure uses the same T_reg rows, so the gap between the lines is what DCDI sums.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .bootstrap import B_DEFAULT, SEED, Est, cluster_sums, counts_matrix, ratio_reps

LAMBDA0 = 8192


def _truth(s: pd.Series) -> pd.Series:
    """None/NaN -> False, everything else -> its truth value."""
    return s.apply(lambda v: False if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)).astype(bool)

CHECKPOINTS = (8, 12, 16, 20, 24)
ID_COLS = ["model_key", "run", "perm_seed", "position", "step_id", "step_type"]


def _arm(steps: pd.DataFrame, arm: str, suffix: str) -> pd.DataFrame:
    d = steps[steps["arm"] == arm].copy()
    keep = ID_COLS + ["a", "functional", "Lambda", "status", "length_matched"]
    for c in keep:
        if c not in d:
            d[c] = np.nan
    d = d[keep].rename(columns={"a": f"a{suffix}", "functional": f"func{suffix}", "Lambda": f"Lambda{suffix}",
                                "status": f"status{suffix}", "length_matched": f"lm{suffix}"})
    return d.drop_duplicates(subset=ID_COLS, keep="last")


def _not_false(s: pd.Series) -> pd.Series:
    """False only where the value is an explicit boolean False (missing counts as True)."""
    return s.apply(lambda v: not (isinstance(v, (bool, np.bool_)) and not v)).astype(bool)


def paired(steps: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, run, position) with session, fresh and (if any) padded arms."""
    if steps.empty:
        return pd.DataFrame()
    s = _arm(steps, "session", "_s")
    f = _arm(steps, "fresh", "_f")
    p = _arm(steps, "padded", "_p")
    df = s.merge(f, on=ID_COLS, how="outer").merge(p, on=ID_COLS, how="left")
    df["key"] = df["perm_seed"].astype(str) + ":" + df["step_id"].astype(str)
    for arm in ("s", "f", "p"):
        df[f"ok_{arm}"] = (_truth(df[f"func_{arm}"]) & df[f"a_{arm}"].notna())
    # A padded control whose realised length missed Lambda by more than 10 % (after the one
    # rebuild) is not a length-matched control (SPEC 6.2): it is excluded and counted.
    df["ok_p"] = df["ok_p"] & _not_false(df["lm_p"])
    df["reg"] = df["step_type"].isin(["foundation", "regular"])
    df["treg"] = df["reg"] & df["ok_s"] & df["ok_f"]
    return df


def drops(pairs: pd.DataFrame) -> dict:
    """Functional drops per model and arm among foundation/regular (run, step) pairs."""
    out = {}
    if pairs.empty:
        return out
    reg = pairs[pairs["reg"]]
    for m, d in reg.groupby("model_key"):
        has_s, has_f = d["a_s"].notna() | d["func_s"].notna(), d["a_f"].notna() | d["func_f"].notna()
        fs = _truth(d["func_s"])
        ff = _truth(d["func_f"])
        out[m] = {"pairs": int(len(d)), "kept": int(d["treg"].sum()),
                  "missing_session": int((~has_s).sum()), "missing_fresh": int((~has_f).sum()),
                  "session_not_functional": int((has_s & ~fs).sum()),
                  "fresh_not_functional": int((has_f & ~ff).sum()),
                  "both_not_functional": int((has_s & has_f & ~fs & ~ff).sum()),
                  "no_applicable_checks": int((fs & ff & (d["a_s"].isna() | d["a_f"].isna())).sum())}
        ck = d[d["position"].isin(CHECKPOINTS)]
        has_p = ck["a_p"].notna() | ck["func_p"].notna()
        fp = _truth(ck["func_p"])
        out[m].update({"checkpoint_pairs": int(len(ck)), "missing_padded": int((~has_p).sum()),
                       "padded_not_functional": int((has_p & ~fp).sum()),
                       "padded_length_unmatched": int((has_p & ~_not_false(ck["lm_p"])).sum())})
    return out


def dcdi(pairs: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    """DCDI, S_C3 and S_C5 per model, with paired replicates over run x step."""
    if pairs.empty or not pairs["treg"].any():
        return {"models": {}, "keys": [], "counts": None}
    t = pairs[pairs["treg"]].copy()
    t["num"] = t["a_f"] - t["a_s"]
    t["den"] = t["a_f"]
    t["one"] = 1.0
    keys = sorted(pairs.loc[pairs["reg"], "key"].unique())
    counts = counts_matrix(len(keys), b, "e2-reg", seed)
    out = {}
    for m, d in t.groupby("model_key"):
        s = cluster_sums(d, "key", keys, ["num", "den", "a_s", "one"])
        if s["den"].sum() <= 0:
            out[m] = {"dcdi": Est(None, note="fresh-control adherence sums to 0"), "s3": Est(None), "s5": Est(None)}
            continue
        est = s["num"].sum() / s["den"].sum()
        reps = ratio_reps(counts, s["num"], s["den"])
        e_dcdi = Est.from_reps(est, reps, n=len(d))
        s3_reps = ratio_reps(counts, s["a_s"], s["one"])
        e_s3 = Est.from_reps(s["a_s"].sum() / s["one"].sum(), s3_reps, n=len(d))
        e_s5 = Est.from_reps(1 - min(max(est, 0.0), 1.0), 1 - np.clip(reps, 0.0, 1.0), n=len(d))
        out[m] = {"dcdi": e_dcdi, "s3": e_s3, "s5": e_s5,
                  "fresh_mean": float(d["a_f"].mean()), "session_mean": float(d["a_s"].mean())}
    return {"models": out, "keys": keys, "counts": counts}


def decomposition(pairs: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    """DCDI_K = DCDI_len + DCDI_hist at the padded checkpoints."""
    if pairs.empty:
        return {"models": {}}
    c = pairs[pairs["treg"] & pairs["position"].isin(CHECKPOINTS) & pairs["ok_p"]].copy()
    if c.empty:
        return {"models": {}}
    c["tot"] = c["a_f"] - c["a_s"]
    c["len"] = c["a_f"] - c["a_p"]
    c["hist"] = c["a_p"] - c["a_s"]
    keys = sorted(pairs.loc[pairs["reg"] & pairs["position"].isin(CHECKPOINTS), "key"].unique())
    counts = counts_matrix(len(keys), b, "e2-ckpt", seed)
    out = {}
    for m, d in c.groupby("model_key"):
        s = cluster_sums(d, "key", keys, ["tot", "len", "hist", "a_f"])
        den = s["a_f"].sum()
        if den <= 0:
            continue
        res = {}
        for part in ("tot", "len", "hist"):
            res[part] = Est.from_reps(s[part].sum() / den, ratio_reps(counts, s[part], s["a_f"]), n=len(d))
        out[m] = res
    return {"models": out}


def drift_rate(pairs: pd.DataFrame) -> dict:
    """beta_m: adherence lost per doubling of context, step fixed effects, cluster-robust by run."""
    out = {}
    if pairs.empty:
        return out
    t = pairs[pairs["treg"] & pairs["Lambda_s"].notna() & (pd.to_numeric(pairs["Lambda_s"], errors="coerce") > 0)].copy()
    for m, d in t.groupby("model_key"):
        d = d.copy()
        d["d"] = d["a_f"] - d["a_s"]
        d["x"] = np.log2(pd.to_numeric(d["Lambda_s"]) / LAMBDA0)
        g = d["run"].nunique()
        if g < 2 or d["x"].nunique() < 2 or len(d) <= d["step_id"].nunique() + 1:
            out[m] = {"beta": None, "note": "not estimable (too few runs or no variation in context length)",
                      "n": int(len(d)), "groups": int(g)}
            continue
        fit = smf.ols("d ~ x + C(step_id)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d["run"])[0]}, use_t=True)
        beta = float(fit.params["x"])
        se = float(fit.bse["x"])
        tcrit = stats.t.ppf(0.975, g - 1)
        out[m] = {"beta": beta, "se": se, "lo": beta - tcrit * se, "hi": beta + tcrit * se,
                  "n": int(len(d)), "groups": int(g), "df": int(g - 1)}
    return out


def curves(pairs: pd.DataFrame, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    """Mean in-session and fresh adherence by session position over T_reg, with bootstrap bands."""
    out = {}
    if pairs.empty:
        return out
    t = pairs[pairs["treg"]]
    for m, d in t.groupby("model_key"):
        rows = []
        for pos, dp in d.groupby("position"):
            n = len(dp)
            rng = np.random.default_rng((seed + int(pos) * 7919 + sum(map(ord, m))) % 2 ** 32)
            idx = rng.integers(0, n, size=(b, n))
            a_s = dp["a_s"].to_numpy(float)
            a_f = dp["a_f"].to_numpy(float)
            rs, rf = a_s[idx].mean(axis=1), a_f[idx].mean(axis=1)
            rows.append({"position": int(pos), "n": n, "session": float(a_s.mean()), "fresh": float(a_f.mean()),
                         "session_lo": float(np.percentile(rs, 2.5)), "session_hi": float(np.percentile(rs, 97.5)),
                         "fresh_lo": float(np.percentile(rf, 2.5)), "fresh_hi": float(np.percentile(rf, 97.5))})
        out[m] = pd.DataFrame(rows).sort_values("position")
    return out
