"""Percentile cluster bootstrap, one-sided bootstrap p-values and Holm correction (SPEC 10).

Resampling K clusters with replacement is represented by a B x K matrix of multinomial
counts. Every statistic in this study that is a ratio of cluster sums (means, DCDI,
micro-F1, accuracies) is then `counts @ num / counts @ den`, which is exact and fast.
All models that share a cluster set (tasks, pages, run x step, images) use the SAME count
matrix, so model differences are paired. Seeds derive from a base seed and a label, so a
statistic's replicates do not depend on the order in which statistics are computed.

Interpretations fixed here:
- One-sided p-value for H1: stat > 0 is (1 + #{replicates <= 0}) / (B_finite + 1).
- Replicates where a statistic is undefined (empty resample) are dropped from its CI and
  p-value; their share is reported.
"""
from __future__ import annotations

import dataclasses
import zlib

import numpy as np

SEED = 20271113
B_DEFAULT = 2000
ALPHA = 0.05


def seed_for(label: str, base: int = SEED) -> int:
    return (base + zlib.crc32(label.encode("utf-8"))) % (2 ** 32)


def counts_matrix(k: int, b: int = B_DEFAULT, label: str = "", base: int = SEED) -> np.ndarray:
    """B x K counts of each cluster in B resamples of K clusters with replacement."""
    if k == 0:
        return np.zeros((b, 0), dtype=np.int64)
    rng = np.random.default_rng(seed_for(label, base))
    return rng.multinomial(k, np.full(k, 1.0 / k), size=b)


def ratio_reps(counts: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    n = counts @ np.asarray(num, dtype=float)
    d = counts @ np.asarray(den, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)


def ci(reps, alpha: float = ALPHA) -> tuple[float | None, float | None]:
    r = np.asarray(reps, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None, None
    return float(np.percentile(r, 100 * alpha / 2)), float(np.percentile(r, 100 * (1 - alpha / 2)))


def p_greater(reps) -> float | None:
    """One-sided bootstrap p-value for H1: statistic > 0."""
    r = np.asarray(reps, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    return float((1 + np.sum(r <= 0)) / (r.size + 1))


def holm(pvals: dict[str, float | None], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down adjustment over the non-missing p-values."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {k: {"p": None, "p_holm": None, "reject": False} for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = {"p": p, "p_holm": running, "reject": running < alpha}
    return out


@dataclasses.dataclass
class Est:
    """A point estimate with its percentile CI, n, and the replicates (for derived stats)."""
    est: float | None
    lo: float | None = None
    hi: float | None = None
    n: int = 0
    reps: np.ndarray | None = dataclasses.field(default=None, repr=False)
    note: str = ""

    @classmethod
    def from_reps(cls, est, reps, n: int = 0, note: str = "") -> "Est":
        if est is None or (isinstance(est, float) and not np.isfinite(est)):
            return cls(None, None, None, n, None, note or "undefined")
        lo, hi = ci(reps) if reps is not None else (None, None)
        return cls(float(est), lo, hi, n, None if reps is None else np.asarray(reps, dtype=float), note)

    @property
    def ok(self) -> bool:
        return self.est is not None

    def reps_or_point(self, b: int) -> np.ndarray:
        if self.reps is not None:
            return self.reps
        return np.full(b, np.nan if self.est is None else self.est)


def cluster_sums(df, key_col: str, keys: list, value_cols: list[str]) -> dict[str, np.ndarray]:
    """Per-cluster sums of `value_cols`, aligned to `keys` (missing clusters sum to 0)."""
    out = {}
    if df.empty:
        return {c: np.zeros(len(keys)) for c in value_cols}
    g = df.groupby(key_col)[value_cols].sum()
    g = g.reindex(keys).fillna(0.0)
    for c in value_cols:
        out[c] = g[c].to_numpy(dtype=float)
    return out


def ratio_est(df, key_col: str, keys: list, num_col: str, den_col: str, counts: np.ndarray) -> Est:
    """Ratio of sums over clusters, with replicates from `counts` (B x len(keys))."""
    if df.empty or df[den_col].sum() == 0:
        return Est(None, n=0, note="no data")
    s = cluster_sums(df, key_col, keys, [num_col, den_col])
    est = s[num_col].sum() / s[den_col].sum()
    reps = ratio_reps(counts, s[num_col], s[den_col])
    return Est.from_reps(est, reps, n=int(round(df[den_col].sum())))


def resample_rows(groups: list[np.ndarray], counts_row: np.ndarray) -> np.ndarray:
    """Row indices of one resample: each cluster's rows repeated by its count."""
    picks = [np.tile(g, c) for g, c in zip(groups, counts_row) if c]
    return np.concatenate(picks) if picks else np.array([], dtype=int)
