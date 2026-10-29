"""Operator balancing and dealing for the E1 corpus (synthetic target sets; no browser)."""
import random

import pytest

from harness import e1_corpus as c

ALL = sorted(c.OPERATORS)
W3C_LIKE = [op for op in ALL if op not in ("M02", "M04", "M07", "M10")]      # 8 operators


def targets(page_ops: dict[str, list[str]]) -> dict:
    return {p: {"ops": {op: ["t"] for op in ops}} for p, ops in page_ops.items()}


def check(counts: dict) -> None:
    assert {op: sum(v.get(op, 0) for v in counts.values()) for op in ALL} == {op: 16 for op in ALL}
    for p, v in counts.items():
        assert sum(v.values()) == 16 and all(1 <= n <= 4 for n in v.values()), (p, v)
        variants = c.deal(v, random.Random(1))
        assert len(variants) == 4 and all(len(set(x)) == 4 for x in variants)


@pytest.mark.parametrize("seed", range(40))
def test_balances_when_some_operators_are_authored_only(monkeypatch, seed):
    # Four pages without M02/M04/M07/M10 (as the W3C pages): those operators must reach 16
    # uses on the eight full pages, which needs multi-step moves through other operators.
    pages = {f"B{i:02d}": (W3C_LIKE if i <= 4 else ALL) for i in range(1, 13)}
    monkeypatch.setattr(c, "PAGE_IDS", list(pages))
    check(c.op_counts(targets(pages), random.Random(seed)))


def test_one_page_missing_an_extra_operator(monkeypatch):
    pages = {f"B{i:02d}": (W3C_LIKE if i <= 4 else ALL) for i in range(1, 13)}
    pages["B12"] = [op for op in ALL if op != "M10"]
    monkeypatch.setattr(c, "PAGE_IDS", list(pages))
    check(c.op_counts(targets(pages), random.Random(c.SEED)))


def test_impossible_balance_is_reported(monkeypatch):
    pages = {f"B{i:02d}": W3C_LIKE for i in range(1, 13)}               # M10 nowhere
    monkeypatch.setattr(c, "PAGE_IDS", list(pages))
    with pytest.raises(RuntimeError, match="cannot balance"):
        c.op_counts(targets(pages), random.Random(0))


def test_committed_corpus_meets_the_task_file():
    import collections
    import json
    gt = json.loads((c.CORPUS / "ground_truth.json").read_text())
    assert len(gt) == 48
    ops = collections.Counter(f["operator"] for v in gt.values() for f in v["faults"])
    assert ops == {op: 16 for op in ALL}
    assert all(len({f["operator"] for f in v["faults"]}) == 4 for v in gt.values())
    t = c.load_targets()
    assert all(len(t[p]["ops"]) == 12 for p in c.PAGE_IDS if c.PROVENANCE[p] == "authored")
    assert all(len(t[p]["ops"]) >= 8 for p in c.PAGE_IDS)
