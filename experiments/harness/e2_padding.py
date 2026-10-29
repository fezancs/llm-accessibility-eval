"""E2 padding transcript for the padded control (SPEC 6.2, prompts/e2_padding.txt).

    python -m harness.e2_padding            # generate data/e2_padding_transcript.json (once)

Runs claude-haiku-5-5 with the padding system prompt, the same three tools and an empty
scratch workspace through the 30 user turns of prompts/e2_padding.txt (max 10 tool calls per
turn, the same loop as the E2 session). Calls are logged under experiment "e2" with item ids
e2/padding/turn<nn>/call<k>; every call is cached, so a re-run is free.

The saved transcript holds the internal messages with every assistant turn converted to
provider "neutral" (text + tool calls, no provider raw), so it can be replayed to any model.
Tool-call ids are renamed to pad_t<nn>_c<k> (portable across providers) consistently in the
calls and their results. `turn_boundaries[i]` = number of messages after turn i+1, so
messages[:turn_boundaries[i]] is a prefix that ends at a complete turn boundary.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

from . import budget, config, llm
from .e2_agent import TOOLS, run_turn
from .e2_tools import Workspace

MODEL = "claude-haiku-5-5"
TRANSCRIPT = config.DATA / "e2_padding_transcript.json"
TURN_CAP = 10


def transcript_path() -> pathlib.Path:
    return TRANSCRIPT


def parse_prompt(text: str | None = None) -> tuple[str, list[str]]:
    """(padding system prompt, the 30 user turns) from prompts/e2_padding.txt."""
    text = text if text is not None else config.prompt("e2_padding.txt")
    m = re.search(r"System prompt for the padding session\n(.*?)\n\s*\nUser turns\n(.*)$", text, re.S)
    if not m:
        raise ValueError("prompts/e2_padding.txt: cannot find the system prompt and user turns")
    system = m.group(1).strip()
    turns = []
    for line in m.group(2).splitlines():
        mm = re.match(r"^\s*(\d+)\.\s+(.*\S)\s*$", line)
        if mm:
            if int(mm.group(1)) != len(turns) + 1:
                raise ValueError(f"user turn numbering broken at {mm.group(1)}")
            turns.append(mm.group(2))
    if len(turns) != 30:
        raise ValueError(f"expected 30 user turns, found {len(turns)}")
    return system, turns


def to_neutral(messages: list[dict]) -> list[dict]:
    """Assistant turns -> provider "neutral"; tool-call ids renamed consistently."""
    out, rename, turn = [], {}, 0
    for m in messages:
        if m["role"] == "user":
            turn += 1
            out.append({"role": "user", "content": [{"type": "text", "text": p["text"]} for p in m["content"]
                                                    if p.get("type") == "text"]})
        elif m["role"] == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                new = f"pad_t{turn:02d}_c{len(rename)}"
                rename[c["id"]] = new
                calls.append({"id": new, "name": c["name"], "arguments": c.get("arguments") or {}})
            text = "".join(p.get("text", "") for p in m.get("content", []) if p.get("type") == "text")
            out.append({"role": "assistant", "provider": "neutral",
                        "content": [{"type": "text", "text": text}] if text else [], "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "results": [{**r, "tool_call_id": rename.get(r["tool_call_id"], r["tool_call_id"])}
                                                    for r in m["results"]]})
    return out


def generate(client: llm.LLM, model: str = MODEL, out_path: pathlib.Path | None = None,
             scratch: pathlib.Path | None = None) -> dict:
    system, turns = parse_prompt()
    scratch = scratch or (config.RUNS / client.experiment / "workspaces" / "_padding")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    ws = Workspace(scratch)
    messages, boundaries, stats = [], [], []
    for i, text in enumerate(turns, 1):
        messages.append(llm.user(text))
        res = run_turn(client, model, system, TOOLS, messages, ws, item_prefix=f"e2/padding/turn{i:02d}",
                       sample_idx=0, cap=TURN_CAP)
        if res["status"] != "ok":
            raise RuntimeError(f"padding turn {i} failed ({res['status']}): {res['error']}")
        boundaries.append(len(messages))
        stats.append({k: res[k] for k in ("Lambda", "llm_calls", "tool_calls", "malformed_tool_calls",
                                           "cost_usd", "cap_reached")})
    neutral = to_neutral(messages)
    data = {"model_key": model, "system": system, "tools": [t["name"] for t in TOOLS],
            "n_turns": len(turns), "turn_boundaries": boundaries, "turn_stats": stats,
            "cost_usd": sum(s["cost_usd"] for s in stats), "messages": neutral}
    path = out_path or transcript_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(scratch, ignore_errors=True)
    return data


def load(path: pathlib.Path | None = None) -> dict:
    p = path or transcript_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; generate it first with: python -m harness.e2_padding")
    data = json.loads(p.read_text(encoding="utf-8"))
    for m in data["messages"]:
        if m["role"] == "assistant" and m.get("provider") != "neutral":
            raise ValueError("padding transcript must contain only neutral assistant turns")
    return data


def prefix(data: dict, n_turns: int) -> list[dict]:
    """The first n_turns complete turns (0 = no padding)."""
    if n_turns <= 0:
        return []
    return [json.loads(json.dumps(m)) for m in data["messages"][:data["turn_boundaries"][n_turns - 1]]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", help="output path (default data/e2_padding_transcript.json)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing transcript")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out) if args.out else transcript_path()
    if out.exists() and not args.force:
        print(f"{out} exists; the transcript is generated once and reused (pass --force to regenerate)",
              file=sys.stderr)
        return 1
    config.load_env()
    client = llm.LLM("e2")
    try:
        data = generate(client, args.model, out)
    except budget.BudgetExceeded as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2
    print(f"padding transcript: {data['n_turns']} turns, {len(data['messages'])} messages, "
          f"{sum(s['tool_calls'] for s in data['turn_stats'])} tool calls, ${data['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
