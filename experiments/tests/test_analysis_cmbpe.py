"""CMBPE Bradley-Terry aggregation and judge baselines (synthetic inputs only)."""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from analysis import cmbpe_bt as cb

MODELS = {
    "g-a1": {"family": "fa"}, "g-a2": {"family": "fa"}, "g-b1": {"family": "fb"}, "g-b2": {"family": "fb"},
    "g-c1": {"family": "fc"}, "g-d1": {"family": "fd"},
    "j-a": {"family": "fa"}, "j-b": {"family": "fb"}, "j-c": {"family": "fc"}, "j-e": {"family": "fe"},
}
THETA = {"g-a1": 0.9, "g-a2": 0.3, "g-b1": 0.0, "g-b2": -0.2, "g-c1": -0.4, "g-d1": -0.6}
DELTA = {"j-a": 0.2, "j-b": -0.3, "j-c": 0.6, "j-e": 0.0}
GAMMA = {"j-a": 0.5, "j-b": 0.0, "j-c": -0.4, "j-e": 0.2}


def simulate(n_images=250, eta=None, seed=0):
    eta = eta or {}
    rng = np.random.default_rng(seed)
    th = {g: v - np.mean(list(THETA.values())) for g, v in THETA.items()}
    rows = []
    gens = list(THETA)
    for im in range(n_images):
        lens = {g: int(rng.integers(40, 160)) for g in gens}
        for a, b in itertools.combinations(gens, 2):
            for j in DELTA:
                for first, second in ((a, b), (b, a)):
                    dl = (lens[first] - lens[second]) / 34.6
                    own = (MODELS[first]["family"] == MODELS[j]["family"]) - (MODELS[second]["family"] == MODELS[j]["family"])
                    p = expit(th[first] - th[second] + DELTA[j] + GAMMA[j] * dl + eta.get(j, 0.0) * own)
                    rows.append({"judge": j, "image_id": f"i{im}", "gen_first": first, "gen_second": second,
                                 "id_first": f"i{im}-{first}", "id_second": f"i{im}-{second}",
                                 "len_first": lens[first], "len_second": lens[second], "y": int(rng.random() < p)})
    return pd.DataFrame(rows), th


@pytest.fixture(scope="module")
def fitted():
    df, th = simulate(eta={"j-a": 1.2})
    return cb.e3c(df, MODELS, b=20, seed=1), df, th


def test_bt_recovers_theta_delta_gamma(fitted):
    out, df, th = fitted
    sd = cb.sd_of_lengths(df.rename(columns={"gen_first": "first", "gen_second": "second"}), "id_first", "id_second")
    assert sd == pytest.approx(34.6, rel=0.15)                      # uniform(40, 160) has SD ~34.6
    for g, v in th.items():
        assert out["theta"][g].est == pytest.approx(v, abs=0.12)
    assert sum(e.est for e in out["theta"].values()) == pytest.approx(0.0, abs=1e-9)
    for j in DELTA:
        assert out["delta"][j].est == pytest.approx(DELTA[j], abs=0.12)
        assert out["gamma"][j].est == pytest.approx(GAMMA[j] * sd / 34.6, abs=0.12)
    assert out["converged"]


def test_recusal_count(fitted):
    out, df, _ = fitted
    fam = {k: v["family"] for k, v in MODELS.items()}
    expected = int(((df["judge"].map(fam) == df["gen_first"].map(fam)) |
                    (df["judge"].map(fam) == df["gen_second"].map(fam))).sum())
    assert out["n_recused"] == expected
    assert out["n_aggregated"] == len(df) - expected


def test_self_preference_sign(fitted):
    out, _, _ = fitted
    assert out["eta"]["j-a"].est > 0.8
    assert abs(out["eta"]["j-b"].est) < 0.35
    assert abs(out["eta"]["j-c"].est) < 0.35
    assert not out["eta"]["j-e"].ok                                 # family fe has no generator


def test_win_rates_by_hand():
    w = cb.win_rates({"a": 1.0, "b": 0.0, "c": -1.0})
    assert w["a"] == pytest.approx((expit(1) + expit(2)) / 2)
    assert w["c"] == pytest.approx((expit(-2) + expit(-1)) / 2)


def test_position_consistency_extremes():
    rows = []
    for im in range(5):
        for first, second in (("x", "y"), ("y", "x")):
            rows.append({"judge": "first-picker", "image_id": im, "first": first, "second": second, "y": 1})
            rows.append({"judge": "x-lover", "image_id": im, "first": first, "second": second, "y": int(first == "x")})
    pairs = cb.pair_table(pd.DataFrame(rows), "first", "second", "image_id")
    pc = cb.position_consistency(pairs, ["first-picker", "x-lover"], list(range(5)), 50, 0, "t")
    assert pc["first-picker"].est == 0.0
    assert pc["x-lover"].est == 1.0


def known_rows(judge_rule, n_items=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for deg in ("append", "truncate"):
            for j in ("j1", "j2", "j3", "j4"):
                for of in (True, False):
                    rows.append({"judge": j, "dataset": "blv", "item_id": i, "image_id": i // 2, "degradation": deg,
                                 "original_first": of, "len_first": 50 if of else 70, "len_second": 70 if of else 50,
                                 "y": judge_rule(j, of, rng)})
    return pd.DataFrame(rows)


def test_known_pairs_perfect_and_position_biased_judges():
    rule = lambda j, of, rng: 1 if j == "j4" else int(of)          # j4 always picks the first shown
    out = cb.known_pairs(known_rows(rule), ["j1", "j2", "j3", "j4"], None, None, b=50, seed=0)
    acc = out["accuracy"]["all"]
    assert acc["single:j1"].est == 1.0
    assert 0.3 < acc["single:j4"].est < 0.7                        # right only when the original came first
    assert acc["naive"].est == 1.0                                  # 3 of 4 judges are right in every order
    assert acc["cmbpe"].est == 1.0
    assert out["note"].startswith("E3c unavailable")
    assert set(out["accuracy"]) == {"all", "append", "truncate"}
    assert out["pc"]["j1"].est == 1.0 and out["pc"]["j4"].est == 0.0
    assert out["h4b"] is not None and out["h4b"].est == pytest.approx(0.0)


def test_cmbpe_uses_supplied_biases():
    rule = lambda j, of, rng: int(of) if rng.random() < 0.9 else 1 - int(of)
    kp = known_rows(rule, n_items=30, seed=3)
    biases = {"delta": {j: 0.0 for j in ("j1", "j2", "j3", "j4")}, "gamma": {j: 0.0 for j in ("j1", "j2", "j3", "j4")}}
    out = cb.known_pairs(kp, ["j1", "j2", "j3", "j4"], biases, 20.0, b=50, seed=0)
    assert out["note"] == "biases from E3c"
    assert out["accuracy"]["all"]["cmbpe"].est > 0.9


def test_pointwise_correlations():
    items = pd.DataFrame([{"item_id": f"i{k}", "dataset": "blv", "image_id": k, "description": "w" * (10 + k),
                           "human": {"overall_mean": float(k)}} for k in range(20)])
    pw = pd.DataFrame([{"judge": j, "dataset": "blv", "item_id": f"i{k}", "image_id": k, "score": float(k)}
                       for k in range(20) for j in ("j1", "j2")])
    out = cb.pointwise(pw, items, ["j1", "j2"], b=50, seed=0)["blv"]
    assert out["single:j1"]["r"].est == pytest.approx(1.0)
    assert out["single:j1"]["tau"].est == pytest.approx(1.0)
    assert out["cmbpe"]["r"].est == pytest.approx(1.0)
    assert out["naive"]["r"].est == pytest.approx(1.0)
    assert out["length"]["r"].est == pytest.approx(1.0)            # length rises with k by construction


def test_leakage_classifier_beats_majority_when_text_leaks():
    rows = []
    for k in range(40):
        rows.append({"generator": "g-a1", "status": "ok", "alt_clean": f"Certainly the item {k} shows"})
        rows.append({"generator": "g-b1", "status": "ok", "alt_clean": f"Item {k} displayed, notably"})
    out = cb.leakage(pd.DataFrame(rows), MODELS, seed=0)
    assert out["majority"] == pytest.approx(0.5)
    assert out["accuracy"] > 0.9 and out["lo"] > 0.5


def test_kalpha_perfect_agreement():
    rows = []
    for im in range(6):
        for first, second in (("x", "y"), ("y", "x")):
            for j in ("j1", "j2", "j3"):
                pref_x = im % 2 == 0
                rows.append({"judge": j, "image_id": im, "first": first, "second": second,
                             "y": int((first == "x") == pref_x)})
    pairs = cb.pair_table(pd.DataFrame(rows), "first", "second", "image_id")
    assert cb.kalpha(pairs, ["j1", "j2", "j3"]) == pytest.approx(1.0)
