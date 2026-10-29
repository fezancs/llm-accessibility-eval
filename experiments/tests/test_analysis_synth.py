"""Synthetic study data for the analysis tests (SCHEMAS.md formats).

SYNTHETIC DATA: used only inside pytest temporary directories. Nothing here may reach
paper/ or runs/ (CLAUDE.md hard rule 1).
"""
import itertools
import json
import pathlib
import shutil

import numpy as np
from scipy.special import expit

EXPERIMENTS = pathlib.Path(__file__).resolve().parent.parent

E1 = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-6-sol", "gpt-6-luna", "gemini-3-8-flash",
      "deepseek-v4-pro", "qwen3-8-27b", "gemma-4-31b", "llama-4-maverick"]
E2 = ["claude-opus-5-5", "claude-haiku-5-5", "gpt-6-sol", "gpt-6-luna", "gemini-3-8-flash", "deepseek-v4-pro",
      "qwen3-8-27b"]
GEN = ["claude-opus-5-5", "claude-haiku-5-5", "gpt-6-sol", "gpt-6-luna", "gemini-3-8-flash", "qwen3-8-27b",
       "gemma-4-31b", "llama-4-maverick"]
JUDGES = ["claude-sonnet-5-5", "gpt-6-sol", "gemini-3-8-flash", "qwen3-8-27b"]
FAM = {"claude-opus-5-5": "anthropic", "claude-sonnet-5-5": "anthropic", "claude-haiku-5-5": "anthropic",
       "gpt-6-sol": "openai", "gpt-6-luna": "openai", "gemini-3-8-flash": "google", "gemma-4-31b": "google",
       "qwen3-8-27b": "alibaba", "llama-4-maverick": "meta", "deepseek-v4-pro": "deepseek"}
STEPS = ["F01", "F02", "F03", "F04"] + [f"R{i:02d}" for i in range(1, 15)] + [f"P{i:02d}" for i in range(1, 7)]
PERMS = [101, 202, 303]


def _w(path: pathlib.Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def build(root: pathlib.Path, seed: int = 0, n_images: int = 30) -> pathlib.Path:
    """Create an experiments-like tree with synthetic records under `root`."""
    rng = np.random.default_rng(seed)
    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(EXPERIMENTS / "config" / "models.yaml", root / "config" / "models.yaml")
    skill = {m: 0.5 + 0.4 * i / len(E1) for i, m in enumerate(E1)}

    # ---------------- E1-Gen
    gen = []
    for m in E1:
        for t in range(1, 21):
            for cond in ("unguided", "nudged"):
                for s in range(3):
                    base = skill[m] + (0.08 if cond == "nudged" else 0.0)
                    a = float(np.clip(base + rng.normal(0, 0.1), 0, 1))
                    functional = bool(rng.random() > 0.05)
                    gen.append({"model_key": m, "task_id": f"G{t:02d}", "condition": cond, "sample_idx": s,
                                "status": "ok", "html_found": True, "functional": functional, "a": a,
                                "checks": [], "report_only": [], "n_elements": 100, "violating_nodes": 2,
                                "violations_per_100": 2.0, "incomplete": 0,
                                "selfassess": {"conforms": bool(rng.random() < 0.7), "parse_ok": True, "retried": False},
                                "cost_usd": 0.001 + 0.02 * skill[m], "latency_s": 5 + 30 * skill[m] + rng.normal(0, 1),
                                "ttft_s": 1.0, "input_tokens": 300, "output_tokens": 2000, "call_ids": ["x"]})
    _w(root / "runs/e1/scores/gen.jsonl", gen)

    # ---------------- E1-Repair / Audit
    ops = [f"M{i:02d}" for i in range(1, 13)]
    rep, aud = [], []
    for m in E1:
        for bp in range(1, 13):
            prov = "w3c" if bp <= 4 else "authored"
            for v in range(1, 5):
                page = f"B{bp:02d}-V{v}"
                chosen = [ops[(bp + v + k * 3) % 12] for k in range(4)]
                faults = [{"operator": o, "target_id": f"t{k}", "resolved": bool(rng.random() < skill[m])}
                          for k, o in enumerate(chosen)]
                res_n = sum(f["resolved"] for f in faults)
                regs = int(rng.random() < 0.2)
                preserved = bool(rng.random() > 0.05)
                score = max(0.0, (res_n - regs) / 4) * preserved
                rep.append({"model_key": m, "page_id": page, "base_id": f"B{bp:02d}", "provenance": prov, "k": 4,
                            "faults": faults, "regressions": regs, "regression_list": [], "preserved": preserved,
                            "preservation": {}, "score": score, "full_repair": res_n == 4 and regs == 0 and preserved,
                            "status": "ok", "html_found": True, "cost_usd": 0.002 + 0.05 * skill[m] ** 3,
                            "latency_s": 10 + 60 * skill[m] + rng.normal(0, 2), "ttft_s": 1.0, "input_tokens": 6000,
                            "output_tokens": 6000, "page_tokens": 6000, "call_ids": ["x"]})
                matched = [{"operator": f["operator"], "target_id": f["target_id"],
                            "matched": bool(rng.random() < skill[m] * 0.9)} for f in faults]
                tp = sum(x["matched"] for x in matched)
                fp = int(rng.integers(0, 3))
                aud.append({"model_key": m, "page_id": page, "base_id": f"B{bp:02d}", "provenance": prov,
                            "n_reported": tp + fp, "parse_ok": True, "retried": False, "faults": matched,
                            "tp": tp, "fp": fp, "fn": 4 - tp, "status": "ok", "cost_usd": 0.01, "latency_s": 8.0,
                            "call_ids": ["x"]})
    _w(root / "runs/e1/scores/repair.jsonl", rep)
    _w(root / "runs/e1/scores/audit.jsonl", aud)

    # ---------------- E2
    steps, sessions = [], []
    drift = {m: 0.02 * i for i, m in enumerate(E2)}
    for m in E2:
        for r, ps in enumerate(PERMS):
            order = STEPS[:4] + list(np.random.default_rng(ps).permutation(STEPS[4:]))
            tool_calls = 0
            end_audit = []
            for pos, sid in enumerate(order, start=1):
                typ = "foundation" if sid[0] == "F" else "regular" if sid[0] == "R" else "pressure"
                lam = int(4000 * 1.12 ** pos)
                a0 = float(np.clip(0.95 - 0.01 * int(sid[1:]) % 5 + rng.normal(0, 0.02), 0, 1))
                a = float(np.clip(a0 - drift[m] * np.log2(lam / 8192 + 1) + rng.normal(0, 0.02), 0, 1))
                arms = [("session", a, lam), ("fresh", a0, 3000)]
                if pos in (8, 12, 16, 20, 24):
                    arms.append(("padded", (a + a0) / 2, lam))
                for arm, av, L in arms:
                    tc = int(rng.integers(2, 6))
                    if arm == "session":
                        tool_calls += tc
                    steps.append({"model_key": m, "run": r, "perm_seed": ps, "position": pos, "step_id": sid,
                                  "step_type": typ, "arm": arm, "functional": bool(rng.random() > 0.03),
                                  "a": av, "checks": [], "accessible": av >= 1.0, "Lambda": L, "tool_calls": tc,
                                  "malformed_tool_calls": 0, "cost_usd": 0.01, "latency_s": 20.0, "tool_time_s": 0.1,
                                  "pressure_score": (float(rng.choice([0, 0.5, 1])) if typ == "pressure" else None),
                                  "intent_met": (bool(rng.random() < 0.6) if typ == "pressure" else None),
                                  "context_exhausted": False, "status": "ok"})
                end_audit.append({"step_id": sid, "position": pos, "a_at_completion": a,
                                  "a_end": a - 0.05 * (rng.random() < 0.2), "fell": bool(rng.random() < 0.2)})
            sessions.append({"model_key": m, "run": r, "perm_seed": ps, "completed_steps": 24,
                             "total_tool_calls": tool_calls, "context_exhausted_at": None, "cost_usd": 1.0,
                             "end_audit": end_audit})
    _w(root / "runs/e2/scores/steps.jsonl", steps)
    _w(root / "runs/e2/scores/sessions.jsonl", sessions)

    # ---------------- E3 items, pointwise, known pairs
    items = []
    for ds, n in (("blv", 68), ("cr", 204)):
        for i in range(n):
            q = rng.uniform(1, 5)
            desc = "word " * int(5 + 10 * rng.random())
            items.append({"item_id": f"{ds}{i}", "dataset": ds, "image_id": f"{ds}img{i // 2}",
                          "image_path": None, "image_url": "u", "retrieved": True, "description": desc.strip(),
                          "context": {}, "human": {"overall_mean": q, "n_raters": 4}})
    _w(root / "data/e3/items.jsonl", items)
    pw, kp = [], []
    for it in items:
        q = it["human"]["overall_mean"]
        for j in JUDGES:
            pw.append({"judge": j, "dataset": it["dataset"], "item_id": it["item_id"], "image_id": it["image_id"],
                       "score": int(np.clip(round(q + rng.normal(0, 0.8)), 1, 5)), "parse_ok": True, "retried": False})
            for deg in ("append", "repeat", "swap", "truncate"):
                for of in (True, False):
                    p_orig = 0.85
                    first_pref = (rng.random() < p_orig) if of else (rng.random() > p_orig)
                    if j == "qwen3-8-27b" and rng.random() < 0.3:
                        first_pref = True                         # a position-biased judge
                    y = int(first_pref)
                    kp.append({"judge": j, "dataset": it["dataset"], "item_id": it["item_id"],
                               "image_id": it["image_id"], "degradation": deg, "original_first": of,
                               "len_first": 50 if of else 80, "len_second": 80 if of else 50,
                               "better": 1 if y else 2, "y": y, "correct": bool(y == of)})
    _w(root / "runs/e3/judgments/pointwise.jsonl", pw)
    _w(root / "runs/e3/judgments/known_pairs.jsonl", kp)

    # ---------------- E3c generations and pairs
    theta = {g: v for g, v in zip(GEN, np.linspace(-0.8, 0.8, len(GEN)))}
    delta = {"claude-sonnet-5-5": 0.1, "gpt-6-sol": -0.1, "gemini-3-8-flash": 0.3, "qwen3-8-27b": 0.6}
    gamma = {"claude-sonnet-5-5": 0.2, "gpt-6-sol": 0.0, "gemini-3-8-flash": 0.4, "qwen3-8-27b": 0.1}
    eta = {"claude-sonnet-5-5": 1.0, "gpt-6-sol": 0.0, "gemini-3-8-flash": 0.0, "qwen3-8-27b": 0.0}
    sig = {"anthropic": "Plainly,", "openai": "Notably", "google": "Shown:", "alibaba": "Here", "meta": "This"}
    generated, pairs = [], []
    for im in range(n_images):
        img = f"crimg{im}"
        lens = {}
        for k, g in enumerate(GEN):
            L = int(60 + 40 * rng.random())
            lens[g] = L
            generated.append({"generator": g, "image_id": img, "status": "ok", "alt_raw": "x",
                              "alt_clean": f"{sig[FAM[g]]} a photo of object {im} " + "z" * (L % 7),
                              "transforms": ["strip_quotes"] if k % 2 else [], "opaque_id": f"o{im}-{k}",
                              "length": L, "cost_usd": 0.001, "latency_s": 2.0})
        all_pairs = list(itertools.combinations(GEN, 2))
        pick = np.random.default_rng(1000 + im).choice(len(all_pairs), size=16, replace=False)
        for pi in pick:
            a, b = all_pairs[pi]
            for j in JUDGES:
                for first, second in ((a, b), (b, a)):
                    dl = (lens[first] - lens[second]) / 11.5
                    own = (FAM[first] == FAM[j]) - (FAM[second] == FAM[j])
                    p = expit(theta[first] - theta[second] + delta[j] + gamma[j] * dl + eta[j] * own)
                    y = int(rng.random() < p)
                    pairs.append({"judge": j, "image_id": img, "gen_first": first, "gen_second": second,
                                  "id_first": f"{img}-{first}", "id_second": f"{img}-{second}",
                                  "len_first": lens[first], "len_second": lens[second], "better": 1 if y else 2, "y": y})
    _w(root / "runs/e3/generated.jsonl", generated)
    _w(root / "runs/e3/judgments/gen_pairs.jsonl", pairs)

    # ---------------- call logs and harness metadata
    calls = []
    for exp, day in (("e1", 2), ("e2", 4), ("e3", 5)):
        for m in E1[:3]:
            calls.append({"call_id": f"{exp}{m}", "experiment": exp, "model_key": m, "status": "ok",
                          "cost_usd": 1.25, "started_at": f"2026-11-0{day}T10:00:00+00:00"})
    _w(root / "runs/e1/calls.jsonl", [c for c in calls if c["experiment"] == "e1"])
    _w(root / "runs/e2/calls.jsonl", [c for c in calls if c["experiment"] == "e2"])
    _w(root / "runs/e3/calls.jsonl", [c for c in calls if c["experiment"] == "e3"])
    _w(root / "runs/pilot/calls.jsonl", [{"call_id": "p", "experiment": "pilot", "model_key": E1[0], "status": "ok",
                                          "cost_usd": 99.0, "started_at": "2026-10-31T10:00:00+00:00"}])
    (root / "runs/meta.json").write_text(json.dumps({"axe_version": "4.14.0", "chromium_version": "156.0.8078.4",
                                                     "probe_count": 27, "prereg_hash": "abc1234"}))
    return root


def test_builder_writes_every_schema_file(tmp_path):
    root = build(tmp_path / "exp", n_images=4)
    for rel in ("runs/e1/scores/gen.jsonl", "runs/e1/scores/repair.jsonl", "runs/e1/scores/audit.jsonl",
                "runs/e2/scores/steps.jsonl", "runs/e2/scores/sessions.jsonl", "data/e3/items.jsonl",
                "runs/e3/judgments/pointwise.jsonl", "runs/e3/judgments/known_pairs.jsonl",
                "runs/e3/generated.jsonl", "runs/e3/judgments/gen_pairs.jsonl", "runs/meta.json"):
        assert (root / rel).exists(), rel
