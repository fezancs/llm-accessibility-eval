#!/usr/bin/env python3
"""Gate for local/run_local.sh: did any request of one pipeline pass end in failure?

    .venv/bin/python local/step_errors.py SINCE      # SINCE: UTC time, e.g. 2026-11-13T18:05:00

Reads every runs/*/calls.jsonl and keeps the attempts that started at or after SINCE. A request
(request_hash) fails the gate when its last such attempt has status "error", unless the error is
context exhaustion (harness.llm.CONTEXT_RE): a prompt that no longer fits the window is a
legitimate E2 outcome (SPEC 6.1), not a failure to retry. Wrong-model refusals do count: they mean
a cache-replay step reached a server that does not hold the requested model.

Prints one summary line and exits 1 if any request failed, else 0. Exists because the harness
scripts record failed requests and still exit 0, so their exit code cannot tell a finished step
from one that ran against a broken server (DEVIATIONS.md, 2026-11-13).
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import pathlib
import sys

EXP = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP))
from harness.llm import CONTEXT_RE  # noqa: E402


def parse_time(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def main(argv: list[str]) -> int:
    since = parse_time(argv[1])
    last: dict[str, dict] = {}
    for log in sorted((EXP / "runs").glob("*/calls.jsonl")):
        with open(log, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if parse_time(r["started_at"]) < since:
                    continue
                key = r.get("request_hash") or r["call_id"]
                prev = last.get(key)
                if prev is None or (r["started_at"], r.get("attempt", 0)) >= (prev["started_at"], prev.get("attempt", 0)):
                    last[key] = r
    failed = [r for r in last.values()
              if r["status"] == "error" and not CONTEXT_RE.search(r.get("error") or "")]
    kinds = collections.Counter((r["model_key"], (r.get("error") or "")[:60]) for r in failed)
    print(f"step_errors since {argv[1]}: {len(last)} requests, {len(failed)} failed"
          + "".join(f"; {n} x {m}: {e}" for (m, e), n in kinds.most_common(3)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
