"""Bootstrap, one-sided p-values and Holm (synthetic inputs only)."""
import numpy as np
import pandas as pd
import pytest

from analysis import bootstrap as bs


def test_counts_rows_sum_to_k_and_are_seeded():
    c1 = bs.counts_matrix(7, 50, "x")
    c2 = bs.counts_matrix(7, 50, "x")
    assert c1.shape == (50, 7)
    assert (c1.sum(axis=1) == 7).all()
    assert (c1 == c2).all()
    assert not (c1 == bs.counts_matrix(7, 50, "y")).all()


def test_ratio_reps_by_hand():
    counts = np.array([[2, 0, 1], [1, 1, 1]])
    num, den = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 4.0])
    r = bs.ratio_reps(counts, num, den)
    assert r[0] == pytest.approx((2 * 1 + 3) / (2 * 2 + 4))
    assert r[1] == pytest.approx(6 / 8)


def test_ratio_est_matches_plain_mean():
    df = pd.DataFrame({"k": ["a", "a", "b", "c"], "x": [1.0, 0.0, 1.0, 1.0], "one": 1.0})
    keys = ["a", "b", "c"]
    e = bs.ratio_est(df, "k", keys, "x", "one", bs.counts_matrix(3, 200, "t"))
    assert e.est == pytest.approx(0.75)
    assert e.n == 4
    assert 0 <= e.lo <= e.est <= e.hi <= 1


def test_p_greater():
    assert bs.p_greater(np.ones(99)) == pytest.approx(1 / 100)
    assert bs.p_greater(-np.ones(99)) == pytest.approx(1.0)
    assert bs.p_greater([np.nan, np.nan]) is None


def test_holm_by_hand():
    out = bs.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    # sorted: a .01*3 = .03; c .03*2 = .06; b max(.06, .04*1) = .06
    assert out["a"]["p_holm"] == pytest.approx(0.03) and out["a"]["reject"]
    assert out["c"]["p_holm"] == pytest.approx(0.06) and not out["c"]["reject"]
    assert out["b"]["p_holm"] == pytest.approx(0.06) and not out["b"]["reject"]
    assert out["d"]["p_holm"] is None


def test_cluster_bootstrap_coverage_sanity():
    """95% percentile CIs of a clustered mean cover the truth in roughly 95% of simulations."""
    rng = np.random.default_rng(1)
    hits, sims = 0, 150
    for s in range(sims):
        k = 30
        effects = rng.normal(0, 0.5, k)
        rows = [(i, 1.0 + effects[i] + rng.normal(0, 0.5)) for i in range(k) for _ in range(4)]
        df = pd.DataFrame(rows, columns=["k", "x"]).assign(one=1.0)
        e = bs.ratio_est(df, "k", list(range(k)), "x", "one", bs.counts_matrix(k, 400, f"cov{s}"))
        hits += e.lo <= 1.0 <= e.hi
    assert 0.85 <= hits / sims <= 0.99
