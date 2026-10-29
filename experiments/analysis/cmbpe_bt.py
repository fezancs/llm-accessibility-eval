"""Cross-Model Blind Peer Evaluation: Bradley-Terry aggregation with judge biases (SPEC 7.3-7.6).

Model (SPEC 7.5): P(y = 1) = sigmoid(theta_a - theta_b + delta_j + gamma_j * dl), y = 1 when the
first-shown candidate wins, dl = (len_a - len_b) / sd(len). Penalised maximum likelihood with
ridge lambda = 1e-3 on theta (penalty lambda * sum(theta^2) added to the summed negative
log-likelihood); the ridge makes the optimum satisfy sum(theta) = 0, and theta is re-centred
afterwards. Fitted by Newton-Raphson with step halving.

Interpretations fixed here:
- A 1e-6 ridge on delta/gamma/eta is a numerical safeguard only (a judge that always picks
  the first candidate would otherwise send delta to infinity); it does not move estimates
  in non-degenerate data.
- sd(len) is the sample SD of character lengths over the unique candidates of the data set.
- Recusal (aggregation only): drop (j, a, b) if fam(j) = fam(a) or fam(j) = fam(b).
- eta_j: refit on all of judge j's E3c verdicts (own-family pairs included) with theta fixed
  at the panel estimate; delta_j, gamma_j and eta_j are re-estimated jointly.
- Image-cluster bootstrap (B = 2000): rows are re-weighted by their image's resample count,
  which is identical to duplicating the image's rows; every replicate refits the panel and
  each eta_j, so S_C4, theta, delta, gamma and eta all have replicate arrays.
- Known-answer pairs (E3b) and BLV pairs (E3a): with a free quality per candidate, the common
  level of gamma is not identified within a pair (it trades off exactly against theta_a -
  theta_b). CMBPE therefore applies the judge biases (delta_j, gamma_j) estimated on the E3c
  generator comparisons, standardising lengths with the E3c sd(len), and estimates each
  pair's theta difference from all judges and both orders; the verdict is its sign. If E3c
  has no data, delta_j is estimated on E3b itself and gamma_j is set to 0 (stated in output).
- Single judge: one order per pair, drawn once per pair with a seeded RNG and shared by all
  judges and the naive panel; a missing verdict in that order is excluded (counted).
- Naive panel: majority of the four judges in the chosen order; ties and all-missing broken
  by a seeded coin. Its PC compares the majority verdicts of the two orders; its gamma is
  fitted (one delta, one gamma) to its chosen-order verdicts on E3c.
- Position consistency (Table IV) is computed on E3c pairs when present, else on E3b; both
  are reported in the summary.
- Krippendorff's alpha (nominal) on order-averaged verdicts: per (pair, judge) the mean over
  available orders of 1[canonical candidate preferred], categories {0, 0.5, 1}.
- Pointwise: CMBPE panel score = mean over judges of per-judge z-scores (within data set);
  naive panel = mean raw score; length-only = description length in characters.
- Leakage check: pooled out-of-fold accuracy of a 5-fold (stratified, seeded) logistic
  regression on character 3-5-gram TF-IDF features, with a Wilson 95% interval, against the
  majority-class share.
"""
from __future__ import annotations

import dataclasses

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from .bootstrap import B_DEFAULT, SEED, Est, counts_matrix, ratio_reps, seed_for
from .loaders import family

RIDGE = 1e-3
TINY = 1e-6


# ------------------------------------------------------------------ core penalised logistic fit

def logit_fit(X: np.ndarray, y: np.ndarray, pen: np.ndarray, w: np.ndarray | None = None,
              offset: np.ndarray | None = None, beta0: np.ndarray | None = None,
              max_iter: int = 100) -> tuple[np.ndarray, bool]:
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    off = np.zeros(n) if offset is None else offset
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float)

    def nll(bt):
        eta = X @ bt + off
        return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) + np.sum(pen * bt * bt))

    f = nll(beta)
    for _ in range(max_iter):
        eta = X @ beta + off
        mu = expit(eta)
        g = X.T @ (w * (mu - y)) + 2 * pen * beta
        hw = w * mu * (1 - mu)
        H = (X * hw[:, None]).T @ X + np.diag(2 * pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while True:
            nb = beta - t * step
            nf = nll(nb)
            if nf <= f + 1e-12 or t < 1e-10:
                break
            t /= 2
        moved = np.max(np.abs(nb - beta)) if p else 0.0
        beta, f_old, f = nb, f, nf
        if moved < 1e-9 or abs(f_old - f) < 1e-12 * (1 + abs(f)):
            return beta, True
    return beta, False


# ------------------------------------------------------------------ Bradley-Terry panel

@dataclasses.dataclass
class Design:
    cands: list[str]
    judges: list[str]
    X: np.ndarray
    y: np.ndarray
    pen: np.ndarray
    judge_idx: np.ndarray


def design(df: pd.DataFrame, cands: list[str], judges: list[str], sd_len: float,
           a_col: str = "first", b_col: str = "second") -> Design:
    G, J = len(cands), len(judges)
    ci = {c: i for i, c in enumerate(cands)}
    ji = {j: i for i, j in enumerate(judges)}
    n = len(df)
    X = np.zeros((n, G + 2 * J))
    ia = df[a_col].map(ci).to_numpy()
    ib = df[b_col].map(ci).to_numpy()
    jj = df["judge"].map(ji).to_numpy()
    rows = np.arange(n)
    X[rows, ia] += 1.0
    X[rows, ib] -= 1.0
    X[rows, G + jj] = 1.0
    dl = (df["len_first"].to_numpy(float) - df["len_second"].to_numpy(float)) / sd_len
    X[rows, G + J + jj] = dl
    pen = np.r_[np.full(G, RIDGE), np.full(2 * J, TINY)]
    return Design(cands, judges, X, df["y"].to_numpy(float), pen, jj)


def unpack(d: Design, beta: np.ndarray) -> dict:
    G, J = len(d.cands), len(d.judges)
    theta = beta[:G] - beta[:G].mean()
    return {"theta": dict(zip(d.cands, theta)), "delta": dict(zip(d.judges, beta[G:G + J])),
            "gamma": dict(zip(d.judges, beta[G + J:]))}


def win_rates(theta: dict[str, float]) -> dict[str, float]:
    keys = list(theta)
    out = {}
    for m in keys:
        others = [g for g in keys if g != m]
        out[m] = float(np.mean([expit(theta[m] - theta[g]) for g in others])) if others else None
    return out


def sd_of_lengths(df: pd.DataFrame, id_a: str, id_b: str) -> float:
    lens = pd.concat([df[[id_a, "len_first"]].rename(columns={id_a: "id", "len_first": "len"}),
                      df[[id_b, "len_second"]].rename(columns={id_b: "id", "len_second": "len"})])
    lens = lens.drop_duplicates("id")["len"].astype(float)
    sd = float(lens.std(ddof=1)) if len(lens) > 1 else 0.0
    return sd if sd > 0 else 1.0


def eta_design(dj: pd.DataFrame, theta: dict, fam_j: str, fam_of: dict, sd_len: float):
    off = dj["first"].map(theta).to_numpy(float) - dj["second"].map(theta).to_numpy(float)
    own = (dj["first"].map(fam_of) == fam_j).astype(float).to_numpy() - \
          (dj["second"].map(fam_of) == fam_j).astype(float).to_numpy()
    dl = (dj["len_first"].to_numpy(float) - dj["len_second"].to_numpy(float)) / sd_len
    X = np.column_stack([np.ones(len(dj)), dl, own])
    return X, off, own


def e3c(gen_pairs: pd.DataFrame, models: dict, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    """Panel fit with recusal, S_C4, judge biases, self-preference, PC, alpha; bootstrap by image."""
    if gen_pairs.empty:
        return {}
    df = gen_pairs.rename(columns={"gen_first": "first", "gen_second": "second"}).copy()
    n_all = len(df)
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(float)
    fam_of = {m: family(models, m) for m in set(df["first"]) | set(df["second"]) | set(df["judge"])}
    df["fam_j"] = df["judge"].map(fam_of)
    recuse = (df["fam_j"] == df["first"].map(fam_of)) | (df["fam_j"] == df["second"].map(fam_of))
    sd_len = sd_of_lengths(df, "id_first", "id_second") if "id_first" in df else sd_of_lengths(df, "first", "second")
    cands = sorted(set(df["first"]) | set(df["second"]))
    judges = sorted(df["judge"].unique())
    agg = df[~recuse].reset_index(drop=True)
    d = design(agg, cands, judges, sd_len)
    beta, conv = logit_fit(d.X, d.y, d.pen)
    point = unpack(d, beta)
    s4 = win_rates(point["theta"])

    # self-preference eta_j (theta fixed at the panel estimate)
    eta_point, eta_rows = {}, {}
    for j in judges:
        dj = df[df["judge"] == j].reset_index(drop=True)
        X, off, own = eta_design(dj, point["theta"], fam_of[j], fam_of, sd_len)
        eta_rows[j] = (dj, X, off)
        if not np.any(own != 0):
            eta_point[j] = None
            continue
        bj, _ = logit_fit(X, dj["y"].to_numpy(float), np.full(3, TINY), offset=off)
        eta_point[j] = float(bj[2])

    # image-cluster bootstrap
    images = sorted(df["image_id"].unique())
    img_i = {im: i for i, im in enumerate(images)}
    counts = counts_matrix(len(images), b, "e3c", seed)
    agg_img = agg["image_id"].map(img_i).to_numpy()
    theta_r = np.full((b, len(cands)), np.nan)
    delta_r = np.full((b, len(judges)), np.nan)
    gamma_r = np.full((b, len(judges)), np.nan)
    s4_r = np.full((b, len(cands)), np.nan)
    eta_r = {j: np.full(b, np.nan) for j in judges}
    for r in range(b):
        w = counts[r, agg_img].astype(float)
        if w.sum() == 0:
            continue
        br, _ = logit_fit(d.X, d.y, d.pen, w=w, beta0=beta)
        u = unpack(d, br)
        theta_r[r] = [u["theta"][c] for c in cands]
        delta_r[r] = [u["delta"][j] for j in judges]
        gamma_r[r] = [u["gamma"][j] for j in judges]
        wr = win_rates(u["theta"])
        s4_r[r] = [wr[c] for c in cands]
        for j in judges:
            if eta_point[j] is None:
                continue
            dj, _, _ = eta_rows[j]
            X, off, own = eta_design(dj, u["theta"], fam_of[j], fam_of, sd_len)
            wj = counts[r, dj["image_id"].map(img_i).to_numpy()].astype(float)
            if wj.sum() == 0 or not np.any(own[wj > 0] != 0):
                continue
            bj, _ = logit_fit(X, dj["y"].to_numpy(float), np.full(3, TINY), w=wj, offset=off)
            eta_r[j][r] = bj[2]

    out = {"converged": conv, "sd_len": sd_len, "n_judgments": n_all, "n_valid": int(len(df)),
           "n_recused": int(recuse.sum()), "n_aggregated": int(len(agg)), "n_images": len(images),
           "theta": {c: Est.from_reps(point["theta"][c], theta_r[:, i], n=len(agg)) for i, c in enumerate(cands)},
           "s4": {c: Est.from_reps(s4[c], s4_r[:, i], n=len(agg)) for i, c in enumerate(cands)},
           "delta": {j: Est.from_reps(point["delta"][j], delta_r[:, i], n=int((agg["judge"] == j).sum()))
                     for i, j in enumerate(judges)},
           "gamma": {j: Est.from_reps(point["gamma"][j], gamma_r[:, i], n=int((agg["judge"] == j).sum()))
                     for i, j in enumerate(judges)},
           "eta": {j: (Est.from_reps(eta_point[j], eta_r[j], n=int((df["judge"] == j).sum()))
                       if eta_point[j] is not None else Est(None, note="no own-family candidates"))
                   for j in judges},
           "missing_by_judge": {j: int(((gen_pairs["judge"] == j) & gen_pairs["y"].isna()).sum()) for j in judges},
           "first_pref_rate": {j: float(df.loc[df["judge"] == j, "y"].mean()) for j in judges},
           "point": point, "judges": judges, "cands": cands, "fam_of": fam_of}
    pairs = pair_table(df, "first", "second", "image_id")
    out["pc"] = position_consistency(pairs, judges, images, b, seed, "e3c-pc")
    out["kalpha"] = kalpha(pairs, judges)
    out["naive"] = naive_panel_pairs(df, pairs, cands, judges, sd_len, seed)
    return out


# ------------------------------------------------------------------ pair utilities

def pair_table(df: pd.DataFrame, a: str, b: str, img: str) -> pd.DataFrame:
    """Rows keyed by unordered pair with the canonical candidate; pref_canon = 1 if it won."""
    d = df.copy()
    d["cand_lo"] = np.where(d[a].astype(str) <= d[b].astype(str), d[a], d[b])
    d["cand_hi"] = np.where(d[a].astype(str) <= d[b].astype(str), d[b], d[a])
    d["pair"] = d[img].astype(str) + "|" + d["cand_lo"].astype(str) + "|" + d["cand_hi"].astype(str)
    d["canon_first"] = d[a] == d["cand_lo"]
    d["pref_canon"] = np.where(d["canon_first"], d["y"], 1 - d["y"])
    d["image"] = d[img]
    return d


def position_consistency(pairs: pd.DataFrame, judges: list[str], images: list, b: int, seed: int, label: str) -> dict:
    out = {}
    counts = counts_matrix(len(images), b, label, seed)
    for j in judges:
        dj = pairs[(pairs["judge"] == j) & pairs["pref_canon"].notna()]
        g = dj.groupby("pair").agg(n=("pref_canon", "size"), s=("pref_canon", "sum"), image=("image", "first"))
        g = g[g["n"] == 2]
        if g.empty:
            out[j] = Est(None, note="no pair judged in both orders")
            continue
        g["cons"] = ((g["s"] == 0) | (g["s"] == 2)).astype(float)
        g["one"] = 1.0
        s = g.groupby("image")[["cons", "one"]].sum().reindex(images).fillna(0.0)
        out[j] = Est.from_reps(g["cons"].mean(), ratio_reps(counts, s["cons"].to_numpy(), s["one"].to_numpy()), n=len(g))
    return out


def kalpha(pairs: pd.DataFrame, judges: list[str]) -> float | None:
    g = pairs[pairs["pref_canon"].notna()].groupby(["pair", "judge"])["pref_canon"].mean().unstack("judge")
    g = g.reindex(columns=judges)
    if g.shape[0] < 2:
        return None
    data = g.to_numpy(float).T
    try:
        return float(krippendorff.alpha(reliability_data=data, level_of_measurement="nominal"))
    except (ValueError, ZeroDivisionError):
        return None


def chosen_orders(pair_ids: list[str], seed: int, label: str) -> dict[str, bool]:
    """Seeded choice per pair: True = canonical candidate shown first."""
    rng = np.random.default_rng(seed_for(label, seed))
    ids = sorted(pair_ids)
    picks = rng.integers(0, 2, size=len(ids)).astype(bool)
    return dict(zip(ids, picks))


def naive_panel_pairs(df: pd.DataFrame, pairs: pd.DataFrame, cands: list[str], judges: list[str],
                      sd_len: float, seed: int) -> dict:
    """Naive panel on E3c: PC across orders, and gamma from its chosen-order verdicts."""
    coin = np.random.default_rng(seed_for("naive-coin-e3c", seed))
    p = pairs[pairs["pref_canon"].notna()]
    maj = p.groupby(["pair", "canon_first"])["pref_canon"].mean().unstack("canon_first")
    cons, total = 0, 0
    verdict = {}
    for pid, row in maj.iterrows():
        v = {}
        for order in (True, False):
            if order in row and pd.notna(row[order]):
                v[order] = 1.0 if row[order] > 0.5 else 0.0 if row[order] < 0.5 else float(coin.integers(0, 2))
        verdict[pid] = v
        if len(v) == 2:
            total += 1
            cons += int(v[True] == v[False])
    out = {"pc": cons / total if total else None, "pc_n": total}
    order = chosen_orders(list(maj.index), seed, "order-e3c")
    rows = []
    first_row = p.drop_duplicates(["pair", "canon_first"]).set_index(["pair", "canon_first"])
    for pid, canon_first in order.items():
        if canon_first not in verdict.get(pid, {}):
            continue
        r = first_row.loc[(pid, canon_first)]
        pref = verdict[pid][canon_first]                    # 1 = canonical preferred
        y = pref if canon_first else 1 - pref
        rows.append({"first": r["first"], "second": r["second"], "judge": "panel",
                     "len_first": r["len_first"], "len_second": r["len_second"], "y": y})
    if len(rows) > len(cands) + 2:
        nd = pd.DataFrame(rows)
        d = design(nd, cands, ["panel"], sd_len)
        beta, _ = logit_fit(d.X, d.y, d.pen)
        out["gamma"] = float(unpack(d, beta)["gamma"]["panel"])
        out["delta"] = float(unpack(d, beta)["delta"]["panel"])
    else:
        out["gamma"] = None
    return out


# ------------------------------------------------------------------ known-answer and BLV pairs

def _pair_delta(s: np.ndarray, off: np.ndarray, y: np.ndarray) -> float:
    """Penalised ML of a pair's theta difference with fixed judge offsets (ridge lambda/2 on Delta)."""
    X = s.reshape(-1, 1).astype(float)
    beta, _ = logit_fit(X, y, np.array([RIDGE / 2]), offset=off)
    return float(beta[0])


def accuracy_methods(rows: pd.DataFrame, judges: list[str], biases: dict | None, sd_len_c: float | None,
                     seed: int, label: str, by: str | None = None) -> pd.DataFrame:
    """Per pair: correct (bool or NaN) for each single judge, the naive panel and CMBPE.

    `rows` columns: pair, image, judge, orig_first (bool: preferred-by-truth candidate shown
    first), y, len_first, len_second [, by]. Ground truth: the candidate marked `orig` wins.
    """
    r = rows[rows["y"].notna()].copy()
    r["orig_pref"] = np.where(r["orig_first"], r["y"], 1 - r["y"])
    pair_ids = sorted(rows["pair"].unique())
    order = chosen_orders(pair_ids, seed, f"order-{label}")
    coin = np.random.default_rng(seed_for(f"coin-{label}", seed))
    meta = rows.drop_duplicates("pair").set_index("pair")
    out = []
    grouped = {pid: g for pid, g in r.groupby("pair")}
    for pid in pair_ids:
        g = grouped.get(pid, r.iloc[0:0])
        ch = order[pid]
        rec = {"pair": pid, "image": meta.loc[pid, "image"]}
        if by:
            rec[by] = meta.loc[pid, by]
        in_order = g[g["orig_first"] == ch]
        for j in judges:
            v = in_order.loc[in_order["judge"] == j, "orig_pref"]
            rec[f"single:{j}"] = float(v.iloc[0]) if len(v) else np.nan
        votes = in_order["orig_pref"].to_numpy(float)
        m = votes.mean() if len(votes) else 0.5
        rec["naive"] = 1.0 if m > 0.5 else 0.0 if m < 0.5 else float(coin.integers(0, 2))
        if biases is not None and len(g):
            s = np.where(g["orig_first"], 1.0, -1.0)
            dl = (g["len_first"].to_numpy(float) - g["len_second"].to_numpy(float)) / (sd_len_c or 1.0)
            off = g["judge"].map(biases["delta"]).fillna(0.0).to_numpy(float) + \
                g["judge"].map(biases["gamma"]).fillna(0.0).to_numpy(float) * dl
            delta = _pair_delta(s, off, g["y"].to_numpy(float))
            rec["cmbpe"] = 1.0 if delta > 0 else 0.0 if delta < 0 else float(coin.integers(0, 2))
        else:
            rec["cmbpe"] = np.nan
        out.append(rec)
    return pd.DataFrame(out)


def e3b_biases_fallback(rows: pd.DataFrame, judges: list[str]) -> dict:
    """delta_j from known pairs alone (per-pair theta difference free), gamma_j = 0."""
    r = rows[rows["y"].notna()].reset_index(drop=True)
    pids = sorted(r["pair"].unique())
    pi = {p: i for i, p in enumerate(pids)}
    ji = {j: i for i, j in enumerate(judges)}
    P, J = len(pids), len(judges)
    X = np.zeros((len(r), P + J))
    rows_i = np.arange(len(r))
    X[rows_i, r["pair"].map(pi).to_numpy()] = np.where(r["orig_first"], 1.0, -1.0)
    X[rows_i, P + r["judge"].map(ji).to_numpy()] = 1.0
    pen = np.r_[np.full(P, RIDGE / 2), np.full(J, TINY)]
    beta, _ = logit_fit(X, r["y"].to_numpy(float), pen)
    return {"delta": dict(zip(judges, beta[P:])), "gamma": {j: 0.0 for j in judges}}


def summarize_accuracy(acc: pd.DataFrame, methods: list[str], images: list, b: int, seed: int, label: str,
                       by: str | None = None) -> dict:
    counts = counts_matrix(len(images), b, label, seed)
    img_index = pd.Index(images)

    def one(sub: pd.DataFrame) -> dict:
        res = {}
        for mth in methods:
            v = sub[["image", mth]].dropna()
            if v.empty:
                res[mth] = Est(None, note="no verdicts")
                continue
            v = v.assign(one=1.0)
            s = v.groupby("image")[[mth, "one"]].sum().reindex(img_index).fillna(0.0)
            res[mth] = Est.from_reps(v[mth].mean(), ratio_reps(counts, s[mth].to_numpy(), s["one"].to_numpy()), n=len(v))
        return res

    out = {"all": one(acc)}
    if by:
        for k, sub in acc.groupby(by):
            out[k] = one(sub)
    return out


def known_pairs(kp: pd.DataFrame, judges: list[str], biases: dict | None, sd_len_c: float | None,
                b: int = B_DEFAULT, seed: int = SEED) -> dict:
    if kp.empty:
        return {}
    rows = kp.copy()
    rows["pair"] = rows["dataset"].astype(str) + "|" + rows["item_id"].astype(str) + "|" + rows["degradation"].astype(str)
    rows["image"] = rows["dataset"].astype(str) + "|" + rows["image_id"].astype(str)
    rows["orig_first"] = rows["original_first"].astype(bool)
    rows["y"] = pd.to_numeric(rows["y"], errors="coerce")
    note = "biases from E3c"
    if biases is None:
        biases = e3b_biases_fallback(rows, judges)
        sd_len_c = 1.0
        note = "E3c unavailable: delta from E3b, gamma = 0"
    acc = accuracy_methods(rows, judges, biases, sd_len_c, seed, "e3b", by="degradation")
    methods = [f"single:{j}" for j in judges] + ["naive", "cmbpe"]
    images = sorted(rows["image"].unique())
    res = summarize_accuracy(acc, methods, images, b, seed, "e3b", by="degradation")
    # H4b replicates: CMBPE minus the median single judge, same image resamples
    counts = counts_matrix(len(images), b, "e3b", seed)
    img_index = pd.Index(images)
    rep_single = []
    for j in judges:
        v = acc[["image", f"single:{j}"]].dropna().assign(one=1.0)
        s = v.groupby("image")[[f"single:{j}", "one"]].sum().reindex(img_index).fillna(0.0)
        rep_single.append(ratio_reps(counts, s[f"single:{j}"].to_numpy(), s["one"].to_numpy()))
    rep_single = np.vstack(rep_single) if rep_single else np.zeros((0, b))
    cm = res["all"]["cmbpe"]
    h4b = None
    if cm.ok and rep_single.size:
        med_point = float(np.median([res["all"][f"single:{j}"].est for j in judges if res["all"][f"single:{j}"].ok]))
        h4b = Est.from_reps(cm.est - med_point, cm.reps - np.nanmedian(rep_single, axis=0), n=cm.n)
    pairs = pair_table(rows.assign(first=np.where(rows["orig_first"], "orig", "deg"),
                                   second=np.where(rows["orig_first"], "deg", "orig")),
                       "first", "second", "image")
    pairs["pair"] = rows["pair"].values
    excluded = {j: int(acc[f"single:{j}"].isna().sum()) for j in judges}
    return {"accuracy": res, "h4b": h4b, "note": note, "n_pairs": int(acc.shape[0]),
            "single_missing": excluded, "pc": position_consistency(pairs, judges, images, b, seed, "e3b-pc"),
            "kalpha": kalpha(pairs, judges),
            "missing_by_judge": {j: int(((kp["judge"] == j) & kp["y"].isna()).sum()) for j in judges}}


def blv_pairs(bp: pd.DataFrame, items: pd.DataFrame, judges: list[str], biases: dict | None,
              sd_len_c: float | None, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    if bp.empty:
        return {}
    lens = {}
    if not items.empty:
        lens = dict(zip(items["item_id"].astype(str), items["description"].fillna("").str.len()))
    r = bp.copy()
    r["pair"] = r["image_id"].astype(str) + "|" + r[["item_a", "item_b"]].astype(str).agg("|".join, axis=1)
    r["image"] = r["image_id"].astype(str)
    winner_a = r["human_prefers"] == "a"
    a_first = r["a_first"].astype(bool)
    r["orig_first"] = np.where(winner_a, a_first, ~a_first)
    la = r["item_a"].astype(str).map(lens).astype(float)
    lb = r["item_b"].astype(str).map(lens).astype(float)
    r["len_first"] = np.where(a_first, la, lb)
    r["len_second"] = np.where(a_first, lb, la)
    r["y"] = pd.to_numeric(r["y"], errors="coerce")
    if biases is None:
        return {"note": "E3c unavailable: CMBPE not computed on BLV pairs",
                "accuracy": summarize_accuracy(accuracy_methods(r, judges, None, None, seed, "e3a"),
                                               [f"single:{j}" for j in judges] + ["naive"],
                                               sorted(r["image"].unique()), b, seed, "e3a")}
    acc = accuracy_methods(r, judges, biases, sd_len_c, seed, "e3a")
    methods = [f"single:{j}" for j in judges] + ["naive", "cmbpe"]
    return {"note": "biases from E3c", "n_pairs": int(acc.shape[0]),
            "accuracy": summarize_accuracy(acc, methods, sorted(r["image"].unique()), b, seed, "e3a")}


# ------------------------------------------------------------------ pointwise calibration

def _corr_reps(x: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int, counts: np.ndarray) -> np.ndarray:
    """Weighted Pearson r per replicate from per-cluster moment sums."""
    def csum(v):
        return np.bincount(groups, weights=v, minlength=k)
    m = {n: csum(v) for n, v in (("n", np.ones_like(x)), ("x", x), ("y", y), ("xx", x * x),
                                 ("yy", y * y), ("xy", x * y))}
    S = {n: counts @ v for n, v in m.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = S["xy"] / S["n"] - (S["x"] / S["n"]) * (S["y"] / S["n"])
        vx = S["xx"] / S["n"] - (S["x"] / S["n"]) ** 2
        vy = S["yy"] / S["n"] - (S["y"] / S["n"]) ** 2
        return cov / np.sqrt(vx * vy)


def _tau_reps(x: np.ndarray, y: np.ndarray, groups: list[np.ndarray], counts: np.ndarray) -> np.ndarray:
    from .bootstrap import resample_rows
    out = np.full(counts.shape[0], np.nan)
    for r in range(counts.shape[0]):
        idx = resample_rows(groups, counts[r])
        if len(idx) > 2:
            out[r] = stats.kendalltau(x[idx], y[idx], variant="b").statistic
    return out


def correlations(scores: pd.DataFrame, b: int, seed: int, label: str) -> dict:
    """scores: columns image, x (method score), h (human mean). Pearson r and Kendall tau-b."""
    s = scores.dropna(subset=["x", "h"])
    if len(s) < 3:
        return {"r": Est(None, note="fewer than 3 items"), "tau": Est(None, note="fewer than 3 items")}
    images = sorted(s["image"].unique())
    gi = s["image"].map({im: i for i, im in enumerate(images)}).to_numpy()
    counts = counts_matrix(len(images), b, label, seed)
    x, h = s["x"].to_numpy(float), s["h"].to_numpy(float)
    r = stats.pearsonr(x, h).statistic if np.std(x) > 0 and np.std(h) > 0 else np.nan
    tau = stats.kendalltau(x, h, variant="b").statistic
    groups = [np.flatnonzero(gi == i) for i in range(len(images))]
    return {"r": Est.from_reps(r if np.isfinite(r) else None, _corr_reps(x, h, gi, len(images), counts), n=len(s)),
            "tau": Est.from_reps(tau if np.isfinite(tau) else None, _tau_reps(x, h, groups, counts), n=len(s))}


def pointwise(pw: pd.DataFrame, items: pd.DataFrame, judges: list[str], b: int = B_DEFAULT,
              seed: int = SEED) -> dict:
    if pw.empty or items.empty or "human" not in items:
        return {}
    it = items.copy()
    it["h"] = [ (hm or {}).get("overall_mean") if isinstance(hm, dict) else None for hm in it["human"]]
    it["length"] = it["description"].fillna("").str.len()
    it["image"] = it["dataset"].astype(str) + "|" + it["image_id"].astype(str)
    p = pw[pw["score"].notna()].copy()
    p["score"] = p["score"].astype(float)
    out = {}
    for ds in sorted(it["dataset"].unique()):
        itd = it[it["dataset"] == ds].set_index("item_id")
        pd_ = p[p["dataset"] == ds]
        res = {}
        wide = pd_.pivot_table(index="item_id", columns="judge", values="score", aggfunc="mean")
        for j in judges:
            if j not in wide:
                res[f"single:{j}"] = {"r": Est(None, note="no scores"), "tau": Est(None, note="no scores")}
                continue
            sc = pd.DataFrame({"x": wide[j]}).join(itd[["h", "image"]], how="inner")
            res[f"single:{j}"] = correlations(sc, b, seed, f"pw-{ds}-{j}")
        if not wide.empty:
            z = (wide - wide.mean()) / wide.std(ddof=0).replace(0, np.nan)
            sc = pd.DataFrame({"x": z.mean(axis=1, skipna=True)}).join(itd[["h", "image"]], how="inner")
            res["cmbpe"] = correlations(sc, b, seed, f"pw-{ds}-cmbpe")
            sc = pd.DataFrame({"x": wide.mean(axis=1, skipna=True)}).join(itd[["h", "image"]], how="inner")
            res["naive"] = correlations(sc, b, seed, f"pw-{ds}-naive")
        sc = itd[["h", "image"]].assign(x=itd["length"].astype(float))
        res["length"] = correlations(sc, b, seed, f"pw-{ds}-length")
        res["n_items"] = int(itd["h"].notna().sum())
        res["missing_by_judge"] = {j: int(((pw["judge"] == j) & (pw["dataset"] == ds) & pw["score"].isna()).sum())
                                   for j in judges}
        out[ds] = res
    # crowd minus BLV correlation per method (independent resamples of the two data sets)
    if "blv" in out and "cr" in out:
        diff = {}
        for mth in [k for k in out["blv"] if isinstance(out["blv"][k], dict) and "r" in out["blv"][k]]:
            a, c = out["blv"][mth]["r"], out["cr"].get(mth, {}).get("r")
            if a is not None and c is not None and a.ok and c.ok:
                diff[mth] = Est.from_reps(c.est - a.est, c.reps - a.reps, n=min(a.n, c.n))
        out["crowd_minus_blv"] = diff
    return out


# ------------------------------------------------------------------ leakage classifier

def leakage(generated: pd.DataFrame, models: dict, seed: int = SEED) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline

    if generated.empty:
        return {}
    g = generated[(generated.get("status", "ok") == "ok") & generated["alt_clean"].notna()].copy()
    g = g[g["alt_clean"].astype(str).str.len() > 0]
    if g.empty:
        return {}
    g["fam"] = g["generator"].map(lambda m: family(models, m))
    vc = g["fam"].value_counts()
    folds = int(min(5, vc.min())) if len(vc) > 1 else 0
    majority = float(vc.max() / vc.sum())
    if folds < 2:
        return {"accuracy": None, "majority": majority, "n": int(len(g)), "note": "too few texts per family"}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed % (2 ** 31))
    X, y = g["alt_clean"].astype(str).to_numpy(), g["fam"].to_numpy()
    correct = 0
    for tr, te in skf.split(X, y):
        clf = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True),
                            LogisticRegression(max_iter=5000))
        clf.fit(X[tr], y[tr])
        correct += int((clf.predict(X[te]) == y[te]).sum())
    n = len(g)
    acc = correct / n
    z = 1.959963984540054
    centre = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"accuracy": acc, "lo": centre - half, "hi": centre + half, "majority": majority, "n": n,
            "folds": folds, "families": {k: int(v) for k, v in vc.items()}}


def transforms_report(generated: pd.DataFrame) -> dict:
    """How often each anonymisation rule fired, per generator (SPEC 7.3)."""
    if generated.empty or "transforms" not in generated:
        return {}
    out = {}
    for m, d in generated.groupby("generator"):
        c: dict[str, int] = {}
        for t in d["transforms"]:
            for rule in t or []:
                c[rule] = c.get(rule, 0) + 1
        out[m] = {"n": int(len(d)), "fired": c}
    return out


def analyse(inputs, b: int = B_DEFAULT, seed: int = SEED) -> dict:
    judges = list(dict.fromkeys(inputs.judges.values()))          # panel order as in models.yaml
    out = {"judges": judges}
    out["e3c"] = e3c(inputs.gen_pairs, inputs.models, b, seed) if not inputs.gen_pairs.empty else {}
    biases = None
    sd_c = None
    if out["e3c"]:
        biases = {"delta": {j: e.est for j, e in out["e3c"]["delta"].items()},
                  "gamma": {j: e.est for j, e in out["e3c"]["gamma"].items()}}
        sd_c = out["e3c"]["sd_len"]
        judges = judges + [j for j in out["e3c"]["judges"] if j not in judges]
        out["judges"] = judges
    out["known"] = known_pairs(inputs.known_pairs, judges, biases, sd_c, b, seed)
    out["blv_pairs"] = blv_pairs(inputs.blv_pairs, inputs.items, judges, biases, sd_c, b, seed)
    out["pointwise"] = pointwise(inputs.pointwise, inputs.items, judges, b, seed)
    out["leakage"] = leakage(inputs.generated, inputs.models, seed)
    out["transforms"] = transforms_report(inputs.generated)
    return out
