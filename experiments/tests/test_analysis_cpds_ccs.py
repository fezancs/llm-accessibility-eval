"""CPDS, Pareto membership, CCS, the decision matrix and the hypothesis family (synthetic inputs only)."""
import numpy as np
import pandas as pd
import pytest

from analysis import ccs, cpds, hypotheses
from analysis.bootstrap import Est
from analysis.loaders import total_params_b


def test_cpds_values_by_hand():
    pen = 1 + np.log10(2)                                           # C = c0, L = l0 -> 1.30103
    assert cpds.cpds_value(0.8, 0.01, 1.0, 1, 0) == pytest.approx(0.8 / pen)
    assert cpds.cpds_value(0.8, 0.01, 1.0, 0.5, 1) == pytest.approx(0.8 / pen ** 1.5)
    assert cpds.cpds_value(0.8, 0.01, 1.0, 1, 1) == pytest.approx(0.8 / pen ** 2)
    assert cpds.cpds_value(0.8, 0.0, 0.0, 1, 1) == pytest.approx(0.8)       # penalties are 1 at zero
    # one decade beyond the reference adds about one to the penalty: C = 0.1 -> 1 + log10(11)
    assert cpds.cpds_value(1.0, 0.1, 0.0, 1, 0) == pytest.approx(1 / (1 + np.log10(11)))


def test_non_dominated_by_hand():
    pts = {"a": (0.9, 1.0, 10), "b": (0.8, 0.1, 5), "c": (0.7, 0.2, 6), "d": (0.9, 1.0, 10)}
    # c is dominated by b; a and d tie (neither dominates the other)
    assert cpds.non_dominated(pts) == {"a", "b", "d"}


def test_pareto_pi_extremes():
    rng = np.random.default_rng(0)
    rows = []
    for m, (s, c, l) in {"best": (0.9, 0.01, 2.0), "worst": (0.2, 1.0, 50.0), "mid": (0.6, 0.02, 1.0)}.items():
        for _ in range(48):
            rows.append({"model_key": m, "score": s + rng.normal(0, 0.01), "cost": c * (1 + rng.normal(0, 0.01)),
                         "latency": l * (1 + rng.normal(0, 0.01))})
    pi = cpds.pareto_pi(pd.DataFrame(rows), b=200)
    assert pi["best"] == 1.0 and pi["worst"] == 0.0
    assert pi["mid"] == 1.0                                          # faster than "best", so not dominated


def test_analyse_scenarios_and_affordability():
    gen = pd.DataFrame([{"model_key": "m", "task_id": "T1", "condition": "nudged", "status": "ok",
                         "html_found": True, "functional": True, "a": 0.5, "cost_usd": 0.01, "latency_s": 1.0}])
    rep = pd.DataFrame([{"model_key": "m", "score": 0.8, "cost_usd": 0.01, "latency_s": 1.0, "page_tokens": 5000}])
    out = cpds.analyse(gen, rep, b=20)
    assert out["cpds"]["batch"]["m"] == pytest.approx(0.8 / (1 + np.log10(2)))
    assert out["cpds"]["interactive"]["m"] == pytest.approx(0.5 / (1 + np.log10(2)) ** 1.5)
    assert out["affordability"]["m"]["cost_1000_pages"] == pytest.approx(10.0)
    assert out["affordability"]["m"]["page_tokens"] == pytest.approx(5000)


def E(v, reps=None):
    return Est(v, v, v, 10, None if reps is None else np.asarray(reps, float))


def test_ccs_geometric_mean_floor_and_partial():
    dims = {"x": {k: E(0.5) for k in ccs.DIMS}, "y": {**{k: E(0.5) for k in ccs.DIMS}, "S4": E(0.0)},
            "z": {**{k: E(0.5) for k in ccs.DIMS}, "S3": Est(None)}}
    out = ccs.ccs(dims, b=5)
    assert out["x"]["ccs"].est == pytest.approx(0.5)
    assert out["y"]["ccs"].est == pytest.approx(np.exp((5 * np.log(0.5) + np.log(0.01)) / 6))
    assert not out["z"]["ccs"].ok and out["z"]["partial"] == pytest.approx(0.5)
    assert out["x"]["arith"] == pytest.approx(0.5)


def test_dimensions_compose_s2_s6():
    gen = {"m": {"sgen_nudged": E(0.6, [0.6] * 5), "overclaim": E(0.4, [0.4] * 5)}}
    rep = {"m": {"srep": E(0.8, [0.8] * 5)}}
    press = {"m": {"pressure": E(0.5, [0.5] * 5)}}
    d = ccs.dimensions(gen, rep, {}, {}, {}, press, b=5)["m"]
    assert d["S2"].est == pytest.approx(0.7)
    assert d["S6"].est == pytest.approx((0.5 + 0.6) / 2)
    assert not d["S1"].ok and not d["S3"].ok


def test_total_params_parsing():
    assert total_params_b({"architecture": "sparse MoE, 1.6T total / 49B active"}) == pytest.approx(1600)
    assert total_params_b({"architecture": "dense 27B (verify)"}) == pytest.approx(27)
    assert total_params_b({"architecture": "MoE, 400B total / 17B active, 128 experts"}) == pytest.approx(400)
    assert total_params_b({"architecture": "undisclosed"}) is None


def test_decision_matrix_rules():
    cfg = {"open27": {"access": "open", "architecture": "dense 27B"},
           "open400": {"access": "open", "architecture": "MoE, 400B total / 17B active"},
           "closedA": {"access": "closed", "architecture": "undisclosed"},
           "closedB": {"access": "closed", "architecture": "undisclosed"}}
    def six(v, s5, s2):
        return {**{k: E(v) for k in ccs.DIMS}, "S5": E(s5), "S2": E(s2)}
    dims = {"closedA": six(0.9, 0.2, 0.9), "closedB": six(0.8, 0.9, 0.8), "open400": six(0.6, 0.8, 0.7),
            "open27": {**{k: E(0.5) for k in ccs.DIMS}, "S3": Est(None), "S5": Est(None), "S2": E(0.4)}}
    scores = ccs.ccs(dims, b=5)
    pi = {"closedA": 0.2, "closedB": 0.9, "open400": 0.6, "open27": 0.9}
    batch = {"closedA": 0.5, "closedB": 0.4, "open400": 0.45, "open27": 0.6}
    out = ccs.decision_matrix(dims, scores, cfg, pi, batch, set(dims))
    # median S5 of the three six-score models is 0.8: closedA (0.2) fails, closedB wins over open400
    assert out["enterprise"] == ("closedB", "open400")
    assert out["research"] == ("open400", "open27")
    # median S2 over E1 models = median(0.9, 0.8, 0.7, 0.4) = 0.75; pi >= 0.5 -> closedB only
    assert out["public"] == ("closedB", "none")
    assert out["edge"] == ("open27", "none")


def test_hypothesis_family_and_holm():
    b = 400
    rng = np.random.default_rng(0)
    big = Est.from_reps(0.05, rng.normal(0.05, 0.01, b))
    small = Est.from_reps(0.20, big.reps + 0.15)                     # paired: small drifts 0.15 more
    dc = {"claude-opus-5-5": {"dcdi": big}, "claude-haiku-5-5": {"dcdi": small}}
    gen = {"a": {"gap": Est.from_reps(0.1, rng.normal(0.1, 0.01, b))},
           "b": {"gap": Est.from_reps(0.1, rng.normal(0.1, 0.01, b))},
           "c": {"gap": Est.from_reps(-0.1, rng.normal(-0.1, 0.01, b))}}
    cfg = {"claude-opus-5-5": {"access": "closed"}, "qwen3-8-27b": {"access": "open"}}
    s4 = {"claude-opus-5-5": Est.from_reps(0.7, rng.normal(0.7, 0.01, b)),
          "qwen3-8-27b": Est.from_reps(0.4, rng.normal(0.4, 0.01, b))}
    eta = {"j1": Est.from_reps(0.5, rng.normal(0.5, 0.1, b)), "j2": Est.from_reps(0.0, rng.normal(0.0, 0.1, b))}
    out = hypotheses.run(gen, dc, s4, {"qwen3-8-27b": 0.6}, eta, None, cfg, b)["tests"]
    assert out["H1a"]["supported"] and out["H1a"]["p"] < 0.01
    assert not out["H1b"]["tested"] and not out["H1c"]["tested"]
    assert out["H2"]["supported"]
    assert out["H3"]["criterion_met"] and not out["H3"]["supported"]      # pi = 0.6: p = 0.4 after Holm
    assert out["H4a"]["supported"] and out["H4a"]["p"] == pytest.approx(2 * out["H4a"]["per_judge_p"]["j1"])
    assert not out["H4b"]["tested"]
    assert out["H5"]["count_positive"] == 2 and out["H5"]["supported"]
