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


def _shared_content_texts(n_images=60, seed=0):
    """Three families describe each image with the SAME words: no style signal at all."""
    rng = np.random.default_rng(seed)
    vocab = ["lamp", "river", "bicycle", "garden", "violin", "harbor", "candle", "meadow", "tower", "kettle",
             "forest", "pillow", "ladder", "window", "saddle", "basket", "mirror", "bridge", "statue", "carpet"]
    rows = []
    for k in range(n_images):
        words = " ".join(rng.choice(vocab, size=4, replace=False)) + f" scene{k:02d}x"
        for g in ("g-a1", "g-b1", "g-c1"):
            rows.append({"generator": g, "image_id": f"img{k}", "status": "ok", "alt_clean": f"A photo of {words}"})
    return pd.DataFrame(rows)


def test_leakage_folds_are_grouped_by_image():
    """Ungrouped folds put an image's other-family texts in training, so the classifier picks a
    wrong family and scores far below chance; grouped folds give chance when there is no style."""
    df = _shared_content_texts()
    out = cb.leakage(df, MODELS, seed=0)
    assert out["groups"] == 60 and out["folds"] == 5
    assert out["majority"] == pytest.approx(1 / 3)
    assert out["lo"] <= out["majority"] <= out["hi"]                # chance, not below it
    per_row = cb.leakage(df.drop(columns="image_id"), MODELS, seed=0)  # fallback: one group per text
    assert per_row["groups"] == len(df) and per_row["accuracy"] < 0.1   # the bias the grouping removes


def test_e3c_disconnected_generators_are_not_identified():
    """One judge (j-a, family fa) recuses every pair with g-a1/g-a2, so those two generators have no
    comparison left: theta, S4 and eta are 'not identified', not the ridge's 0 / 0.5 [0.5, 0.5]."""
    df, _ = simulate(n_images=12, seed=4)
    one = cb.e3c(df[df["judge"] == "j-a"], MODELS, b=10, seed=1)
    assert one["connected"] is False
    for g in THETA:
        assert not one["s4"][g].ok and not one["theta"][g].ok and "not connected" in one["s4"][g].note
    assert not one["eta"]["j-a"].ok and "not connected" in one["eta"]["j-a"].note
    assert one["s4_diff"] and not any(e.ok for e in one["s4_diff"].values())   # no S4 difference key either
    full = cb.e3c(df, MODELS, b=10, seed=1)
    assert full["connected"] is True and all(e.ok for e in full["s4"].values())


def test_e3c_counts_pairs_and_skips_all_missing_verdicts():
    df, _ = simulate(n_images=3, seed=5)
    out = cb.e3c(df, MODELS, b=5, seed=1)
    assert out["n_pairs"] == 3 * 15                                 # 3 images x C(6, 2) unordered pairs
    assert cb.e3c(df.assign(y=np.nan), MODELS, b=5, seed=1) == {}   # nothing valid: no fit, no crash


def test_s4_differences_are_paired_and_alphabetical(fitted):
    out, _, _ = fitted
    assert list(out["s4_diff"]) == list(itertools.combinations(sorted(THETA), 2))     # first minus second
    for (a, c), e in out["s4_diff"].items():
        assert e.est == pytest.approx(out["s4"][a].est - out["s4"][c].est)
        np.testing.assert_allclose(e.reps, out["s4"][a].reps - out["s4"][c].reps)    # same image resamples
        assert e.n == out["s4"][a].n
    big = out["s4_diff"][("g-a1", "g-d1")]                                          # theta 0.9 vs -0.6
    assert big.est > 0 and big.lo > 0


def test_kalpha_replicates_equal_the_package_on_duplicated_images(fitted):
    from analysis.bootstrap import counts_matrix
    df, _ = simulate(n_images=15, seed=6)
    pairs = cb.pair_table(df.rename(columns={"gen_first": "first", "gen_second": "second"}), "first", "second",
                          "image_id")
    judges, images = sorted(DELTA), sorted(df["image_id"].unique())
    counts = np.vstack([np.ones(len(images), dtype=int), counts_matrix(len(images), 3, "t", 0)])
    e = cb.kalpha_est(pairs, judges, images, counts)
    assert e.est == cb.kalpha(pairs, judges) and e.n == 15 * 15                   # every pair has 4 judges
    assert e.reps[0] == pytest.approx(e.est, abs=1e-12)                            # each image once
    for r in range(1, 4):                                                          # images duplicated by count
        dup = pd.concat([pairs[pairs["image"] == im].assign(pair=lambda x, k=k: x["pair"] + f"#{k}")
                         for im, c in zip(images, counts[r]) for k in range(c)])
        assert e.reps[r] == pytest.approx(cb.kalpha(dup, judges), abs=1e-12)
    out = fitted[0]
    assert out["kalpha_est"].est == out["kalpha"] and out["kalpha_est"].lo <= out["kalpha"] <= out["kalpha_est"].hi


def test_known_pairs_cmbpe_minus_naive_uses_the_accuracy_resamples():
    # j3 and j4 always pick the first shown: the naive majority ties (coin) whenever the original is
    # second, while CMBPE absorbs their position bias.
    rule = lambda j, of, rng: 1 if j in ("j3", "j4") else (int(of) if rng.random() < 0.9 else 1 - int(of))
    out = cb.known_pairs(known_rows(rule, n_items=40, seed=2), ["j1", "j2", "j3", "j4"], None, None, b=50, seed=0)
    d = out["cmbpe_minus_naive"]
    assert set(d) == {"all", "append", "truncate"}
    for grp, e in d.items():
        cm, nv = out["accuracy"][grp]["cmbpe"], out["accuracy"][grp]["naive"]
        assert e.est == pytest.approx(cm.est - nv.est) and e.n == min(cm.n, nv.n)
        np.testing.assert_allclose(e.reps, cm.reps - nv.reps)
    assert d["all"].est > 0 and d["all"].lo > 0
    assert out["kalpha_est"].est == out["kalpha"]


def test_pointwise_length_minus_panel_and_common_items():
    from scipy import stats
    rng = np.random.default_rng(2)
    k = 40
    items = pd.DataFrame([{"item_id": f"i{i}", "dataset": "blv", "image_id": i // 2,
                           "description": "w" * int(rng.integers(10, 200)),
                           "human": {"overall_mean": float(rng.normal())}} for i in range(k)])
    pw = pd.DataFrame([{"judge": j, "dataset": "blv", "item_id": f"i{i}", "image_id": i // 2,
                        "score": np.nan if (j == "j3" and i % 4 == 0) else float(rng.integers(1, 6))}
                       for i in range(k) for j in ("j1", "j2", "j3")])
    out = cb.pointwise(pw, items, ["j1", "j2", "j3"], b=50, seed=0)["blv"]
    lm = out["length_minus"]
    assert set(lm) == {"cmbpe", "naive"}
    for m in ("cmbpe", "naive"):
        assert lm[m].est == pytest.approx(out["length"]["r"].est - out[m]["r"].est) and lm[m].n == k
        assert lm[m].lo is not None
    # the difference is taken on the length baseline's own resamples
    it = items.set_index("item_id")
    h = it["human"].map(lambda d: d["overall_mean"])
    sc_len = pd.DataFrame({"x": it["description"].str.len().astype(float), "h": h,
                           "image": "blv|" + it["image_id"].astype(str)})
    np.testing.assert_allclose(cb.paired_r({"length": sc_len}, 50, 0, "pw-blv-length")["length"].reps,
                               out["length"]["r"].reps)
    # common items: j3 skipped every fourth item; scores keep their whole-data-set definitions
    common = [f"i{i}" for i in range(k) if i % 4]
    wide = pw.pivot_table(index="item_id", columns="judge", values="score")
    z = ((wide - wide.mean()) / wide.std(ddof=0)).mean(axis=1)
    expect = {"single:j1": wide["j1"], "single:j3": wide["j3"], "cmbpe": z, "naive": wide.mean(axis=1),
              "length": sc_len["x"]}
    c = out["common"]
    assert set(c) == {"single:j1", "single:j2", "single:j3", "cmbpe", "naive", "length"}
    for m, x in expect.items():
        assert c[m].n == len(common)
        assert c[m].est == pytest.approx(stats.pearsonr(x.loc[common], h.loc[common]).statistic), m
    assert c["single:j3"].est == pytest.approx(out["single:j3"]["r"].est)          # j3 scored exactly these


def test_alt_lengths_count_only_texts_that_enter_e3c():
    g = pd.DataFrame([{"generator": "a", "status": "ok", "alt_clean": "abcd"},
                      {"generator": "a", "status": "ok", "alt_clean": "ab"},
                      {"generator": "a", "status": "error", "alt_clean": "abcdefghij"},
                      {"generator": "a", "status": "ok", "alt_clean": ""},
                      {"generator": "b", "status": "ok", "alt_clean": None},
                      {"generator": "b", "status": "ok", "alt_clean": "xyz"}])
    assert cb.alt_lengths(g) == {"a": 3.0, "b": 3.0}
    assert cb.alt_lengths(pd.DataFrame()) == {}
