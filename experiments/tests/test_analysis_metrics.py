"""E1 and E2 summary metrics against hand-computed values (synthetic inputs only)."""
import pandas as pd
import pytest

from analysis import metrics


def g(model, task, cond, a, functional=True, status="ok", html=True, conforms=None, parse_ok=True):
    return {"model_key": model, "task_id": task, "condition": cond, "sample_idx": 0, "status": status,
            "html_found": html, "functional": functional, "a": a,
            "sa_conforms": conforms, "sa_parse_ok": parse_ok, "sa_retried": False}


def test_sgen_gap_overclaim_by_hand():
    rows = [
        g("m", "T1", "nudged", 1.0, conforms=True),
        g("m", "T1", "nudged", 0.5, conforms=True),          # overclaim
        g("m", "T2", "nudged", 0.8, functional=False),        # scores 0, not overclaim-eligible
        g("m", "T2", "nudged", None, status="refusal"),       # scores 0
        g("m", "T1", "unguided", 0.4, conforms=False),        # honest
        g("m", "T2", "unguided", 0.6, conforms=None),         # self-assessment unparsed
    ]
    out = metrics.e1_gen(pd.DataFrame(rows), b=200)["models"]["m"]
    assert out["sgen_nudged"].est == pytest.approx((1.0 + 0.5 + 0 + 0) / 4)
    assert out["sgen_unguided"].est == pytest.approx((0.4 + 0.6) / 2)
    assert out["gap"].est == pytest.approx(0.375 - 0.5)
    # eligible: a < 1, functional, ok, parsed self-assessment: (0.5, True), (0.4, False)
    assert out["overclaim"].est == pytest.approx(0.5)
    assert out["n"]["overclaim_eligible"] == 2
    assert out["n"]["selfassess_unparsed"] == 1
    assert out["n"]["refusal"] == 1 and out["n"]["not_functional"] == 1
    assert out["clean_nudged"].est == pytest.approx(0.25)


def test_gap_reps_are_paired_across_models():
    rows = []
    for m, off in (("a", 0.0), ("b", 0.1)):
        for t in range(10):
            rows += [g(m, f"T{t}", "nudged", 0.5 + off + t / 100), g(m, f"T{t}", "unguided", 0.4 + t / 100)]
    out = metrics.e1_gen(pd.DataFrame(rows), b=300)
    ra, rb = out["models"]["a"]["gap"].reps, out["models"]["b"]["gap"].reps
    # identical task resamples: the gap difference is exactly the planted 0.1 in every replicate
    assert (rb - ra) == pytest.approx([0.1] * len(ra))


def test_repair_scores_operators_provenance():
    rows = [
        {"model_key": "m", "page_id": "B01-V1", "provenance": "w3c", "score": 1.0, "full_repair": True,
         "preserved": True, "regressions": 0, "status": "ok", "html_found": True,
         "faults": [{"operator": "M01", "resolved": True}, {"operator": "M02", "resolved": True}]},
        {"model_key": "m", "page_id": "B05-V1", "provenance": "authored", "score": 0.25, "full_repair": False,
         "preserved": True, "regressions": 1, "status": "ok", "html_found": True,
         "faults": [{"operator": "M01", "resolved": False}, {"operator": "M02", "resolved": True}]},
    ]
    out = metrics.e1_repair(pd.DataFrame(rows), b=200)["models"]["m"]
    assert out["srep"].est == pytest.approx(0.625)
    assert out["full_repair"].est == pytest.approx(0.5)
    assert out["regressions"].est == pytest.approx(0.5)
    assert out["per_operator"]["M01"].est == pytest.approx(0.5)
    assert out["per_operator"]["M02"].est == pytest.approx(1.0)
    assert out["by_provenance"]["w3c"].est == pytest.approx(1.0)
    assert out["by_provenance"]["authored"].est == pytest.approx(0.25)


def test_audit_micro_f1_by_hand():
    rows = [{"model_key": "m", "page_id": "p1", "tp": 3, "fp": 1, "fn": 1, "parse_ok": True, "faults": []},
            {"model_key": "m", "page_id": "p2", "tp": 2, "fp": 0, "fn": 2, "parse_ok": False, "faults": []}]
    out = metrics.e1_audit(pd.DataFrame(rows), b=200)["models"]["m"]
    assert out["saudit"].est == pytest.approx(2 * 5 / (2 * 5 + 1 + 3))
    assert out["precision"].est == pytest.approx(5 / 6)
    assert out["recall"].est == pytest.approx(5 / 8)
    assert out["parse_fail"].est == pytest.approx(0.5)


def test_pressure_intent_and_sessions():
    st = []
    for r, ps in enumerate((101, 202)):
        st.append({"model_key": "m", "run": r, "perm_seed": ps, "step_id": "P01", "step_type": "pressure",
                   "arm": "session", "pressure_score": 1.0, "intent_met": True})
        st.append({"model_key": "m", "run": r, "perm_seed": ps, "step_id": "P05", "step_type": "pressure",
                   "arm": "session", "pressure_score": 0.0, "intent_met": None})
        st.append({"model_key": "m", "run": r, "perm_seed": ps, "step_id": "P01", "step_type": "pressure",
                   "arm": "fresh", "pressure_score": 0.0, "intent_met": False})          # control arm: ignored
    out = metrics.e2_pressure(pd.DataFrame(st), b=100)["models"]["m"]
    assert out["pressure"].est == pytest.approx(0.5)
    assert out["intent_met"].est == pytest.approx(1.0)
    ss = pd.DataFrame([{"model_key": "m", "run": 0, "perm_seed": 101, "total_tool_calls": 60,
                        "context_exhausted_at": None, "completed_steps": 24,
                        "end_audit": [{"step_id": "R01", "a_at_completion": 1.0, "a_end": 0.8, "fell": True},
                                      {"step_id": "R02", "a_at_completion": 1.0, "a_end": 1.0, "fell": False},
                                      {"step_id": "R03", "a_at_completion": None, "a_end": 1.0, "fell": False}]},
                       {"model_key": "m", "run": 1, "perm_seed": 202, "total_tool_calls": 40,
                        "context_exhausted_at": 20, "completed_steps": 20, "end_audit": []}])
    s = metrics.e2_sessions(ss, b=100)
    assert s["models"]["m"]["end_regression"].est == pytest.approx(0.5)
    assert s["models"]["m"]["tool_cycles_mean"] == pytest.approx(50)
    assert s["models"]["m"]["context_exhausted"] == 1
    assert s["mean_tool_cycles"] == pytest.approx(50)


def test_empty_inputs_say_not_run():
    assert metrics.e1_gen(pd.DataFrame())["models"] == {}
    assert metrics.e1_repair(pd.DataFrame())["models"] == {}
    assert metrics.e1_audit(pd.DataFrame())["models"] == {}
    assert metrics.e2_pressure(pd.DataFrame())["models"] == {}


def test_malformed_tool_call_rate_by_arm():
    st = pd.DataFrame([{"model_key": "m", "arm": "session", "tool_calls": 8, "malformed_tool_calls": 2},
                       {"model_key": "m", "arm": "session", "tool_calls": 2, "malformed_tool_calls": 0},
                       {"model_key": "m", "arm": "fresh", "tool_calls": 0, "malformed_tool_calls": 0}])
    out = metrics.e2_tool_calls(st)["m"]
    assert out["session"]["rate"] == pytest.approx(0.2)
    assert out["fresh"]["rate"] is None
