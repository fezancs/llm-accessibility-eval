"""E2 step file loader and session orders (SPEC 6.1, tasks/e2_steps.yaml).

A session runs the foundation steps F01-F04 in file order, then the 20 non-foundation
steps in the order random.Random(seed).sample(<non-foundation ids in file order>, k=20)
for one of the permutation seeds (101, 202, 303). Run r of every model uses seed index r,
so all models see the same three orders. The orders are frozen in data/e2/permutations.json;
`frozen_orders()` refuses to run if the file disagrees with the computation.
"""
from __future__ import annotations

import functools
import json
import random

from . import config

E2_DATA = config.DATA / "e2"
INITIAL_PROJECT = E2_DATA / "initial_project"
PERMUTATIONS = E2_DATA / "permutations.json"


@functools.lru_cache(maxsize=None)
def spec() -> dict:
    return config.task_file("e2_steps.yaml")


def steps() -> list[dict]:
    return list(spec()["steps"])


def by_id() -> dict[str, dict]:
    return {s["id"]: s for s in steps()}


def step(step_id: str) -> dict:
    return by_id()[step_id]


def foundation_ids() -> list[str]:
    return [s["id"] for s in steps() if s["type"] == "foundation"]


def permuted_ids() -> list[str]:
    """Non-foundation step ids in file order."""
    return [s["id"] for s in steps() if s["type"] != "foundation"]


def seeds() -> list[int]:
    return list(spec()["permutation_seeds"])


def max_tool_calls() -> int:
    return int(spec()["max_tool_calls_per_step"])


def padded_checkpoints() -> list[int]:
    return list(spec()["padded_checkpoints"])


def permutation(seed: int) -> list[str]:
    ids = permuted_ids()
    return random.Random(seed).sample(ids, k=len(ids))


def session_order(seed: int) -> list[str]:
    return foundation_ids() + permutation(seed)


def compute_orders() -> dict:
    return {"seeds": seeds(), "foundation": foundation_ids(),
            "permuted": {str(s): permutation(s) for s in seeds()},
            "sessions": {str(s): session_order(s) for s in seeds()}}


def write_frozen(path=None) -> dict:
    data = compute_orders()
    (path or PERMUTATIONS).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


def frozen_orders() -> dict[int, list[str]]:
    """Session order per seed from data/e2/permutations.json, checked against the computation."""
    data = json.loads(PERMUTATIONS.read_text(encoding="utf-8"))
    if data != compute_orders():
        raise RuntimeError("data/e2/permutations.json does not match the seeds in tasks/e2_steps.yaml")
    return {int(s): list(o) for s, o in data["sessions"].items()}


if __name__ == "__main__":
    print(json.dumps(write_frozen(), indent=1))
