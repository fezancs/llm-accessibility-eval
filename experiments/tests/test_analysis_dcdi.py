"""DCDI, decomposition and drift rate against hand-computed values (synthetic inputs only)."""
import numpy as np
import pandas as pd
import pytest

from analysis import dcdi


def row(m, run, pos, sid, typ, arm, a, func=True, lam=8192):
    return {"model_key": m, "run": run, "perm_seed": [101, 202, 303][run], "position": pos, "step_id": sid,
            "step_type": typ, "arm": arm, "a": a, "functional": func, "Lambda": lam, "status": "ok"}


def test_dcdi_by_hand_with_exclusions():
    rs = []
    # run 0: R01 (1.0 fresh, 0.8 session), R02 (0.5, 0.5); P01 pressure (excluded); R03 session not functional
    rs += [row("m", 0, 5, "R01", "regular", "fresh", 1.0), row("m", 0, 5, "R01", "regular", "session", 0.8)]
    rs += [row("m", 0, 6, "R02", "regular", "fresh", 0.5), row("m", 0, 6, "R02", "regular", "session", 0.5)]
    rs += [row("m", 0, 7, "P01", "pressure", "fresh", 1.0), row("m", 0, 7, "P01", "pressure", "session", 0.0)]
    rs += [row("m", 0, 8, "R03", "regular", "fresh", 1.0), row("m", 0, 8, "R03", "regular", "session", 0.0, func=False)]
    # foundation step counts
    rs += [row("m", 0, 1, "F01", "foundation", "fresh", 0.9), row("m", 0, 1, "F01", "foundation", "session", 0.9)]
    pairs = dcdi.paired(pd.DataFrame(rs))
    out = dcdi.dcdi(pairs, b=200)["models"]["m"]
    # (1.0-0.8) + (0.5-0.5) + (0.9-0.9) over (1.0 + 0.5 + 0.9)
    assert out["dcdi"].est == pytest.approx(0.2 / 2.4)
    assert out["s3"].est == pytest.approx((0.8 + 0.5 + 0.9) / 3)
    assert out["s5"].est == pytest.approx(1 - 0.2 / 2.4)
    d = dcdi.drops(pairs)["m"]
    assert d["pairs"] == 4 and d["kept"] == 3 and d["session_not_functional"] == 1


def test_negative_dcdi_and_s5_clip():
    rs = [row("m", 0, 5, "R01", "regular", "fresh", 0.5), row("m", 0, 5, "R01", "regular", "session", 1.0)]
    out = dcdi.dcdi(dcdi.paired(pd.DataFrame(rs)), b=50)["models"]["m"]
    assert out["dcdi"].est == pytest.approx(-1.0)
    assert out["s5"].est == pytest.approx(1.0)                 # 1 - clip(-1, 0, 1)


def test_decomposition_adds_up_in_every_replicate():
    rng = np.random.default_rng(3)
    rs = []
    for r in range(3):
        for pos, sid in zip((8, 12, 16, 20, 24), ("R01", "R02", "R03", "R04", "R05")):
            a0 = rng.uniform(0.8, 1)
            ap = a0 - rng.uniform(0, 0.1)
            a = ap - rng.uniform(0, 0.1)
            rs += [row("m", r, pos, sid, "regular", "fresh", a0), row("m", r, pos, sid, "regular", "padded", ap),
                   row("m", r, pos, sid, "regular", "session", a)]
    dec = dcdi.decomposition(dcdi.paired(pd.DataFrame(rs)), b=300)["models"]["m"]
    assert dec["len"].est + dec["hist"].est == pytest.approx(dec["tot"].est)
    assert np.allclose(dec["len"].reps + dec["hist"].reps, dec["tot"].reps)
    assert dec["len"].est > 0 and dec["hist"].est > 0


def test_drift_rate_recovers_planted_beta():
    rng = np.random.default_rng(5)
    rs = []
    steps = [f"R{i:02d}" for i in range(1, 15)]
    for r in range(3):
        order = list(np.random.default_rng(100 + r).permutation(steps))
        for pos, sid in enumerate(order, start=5):
            lam = int(4000 * 1.25 ** pos)
            effect = 0.01 * int(sid[1:])
            d = effect + 0.05 * np.log2(lam / 8192) + rng.normal(0, 0.002)
            rs += [row("m", r, pos, sid, "regular", "fresh", 0.95, lam=3000),
                   row("m", r, pos, sid, "regular", "session", 0.95 - d, lam=lam)]
    out = dcdi.drift_rate(dcdi.paired(pd.DataFrame(rs)))["m"]
    assert out["beta"] == pytest.approx(0.05, abs=0.005)
    assert out["groups"] == 3 and out["df"] == 2
    assert out["lo"] < 0.05 < out["hi"]


def test_paired_differences_between_models():
    rs = []
    for m, loss in (("big", 0.0), ("small", 0.1)):
        for r in range(3):
            for k, sid in enumerate(("R01", "R02", "R03")):
                rs += [row(m, r, 5 + k, sid, "regular", "fresh", 1.0),
                       row(m, r, 5 + k, sid, "regular", "session", 1.0 - loss)]
    out = dcdi.dcdi(dcdi.paired(pd.DataFrame(rs)), b=200)["models"]
    diff = out["small"]["dcdi"].reps - out["big"]["dcdi"].reps
    assert np.allclose(diff, 0.1)


def test_curves_shape():
    rs = []
    for r in range(3):
        rs += [row("m", r, 5, "R01", "regular", "fresh", 1.0), row("m", r, 5, "R01", "regular", "session", 0.9)]
    c = dcdi.curves(dcdi.paired(pd.DataFrame(rs)), b=100)["m"]
    assert list(c["position"]) == [5]
    assert c["session"].iloc[0] == pytest.approx(0.9) and c["fresh"].iloc[0] == pytest.approx(1.0)
