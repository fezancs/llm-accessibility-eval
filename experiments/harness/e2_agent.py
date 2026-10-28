"""E2 in-session arm (SPEC 6.1): one long tool-using conversation per (model, run).

    python -m harness.e2_run ...        (the CLI; this module holds the agent loop)

Protocol for one step (one user message):
- call the model (cache_mode "auto", log_request False, item_id
  e2/<model>/run<r>/pos<NN>/<step>/<arm>/call<k>);
- execute every tool call of the response in order and append ONE tool message with all
  results; stop when a response has no tool call or 10 tool calls have been executed in the
  step. If the cap is reached in the middle of a response, the remaining calls of that
  response get is_error results "Tool limit for this request reached." and the step ends
  without a further call. Every tool call the model makes counts toward the cap, malformed
  ones included (they receive an error result).
- no truncation or summarisation of the history.
- LLMError with context exhaustion -> status "context_exhausted"; any other LLMError ->
  status "error"; either ends the run.

Lambda = input + cached + cache-write tokens of the step's FIRST call. latency_s = sum of
the calls' total_s (tool time excluded and reported as tool_time_s).
"""
from __future__ import annotations

import json
import pathlib
import shutil

from . import budget, config, e2_steps, llm
from .e2_tools import Workspace, limit_result

SYSTEM = config.prompt("e2_system.txt").rstrip("\n")
TOOLS = json.loads(config.prompt("e2_tools.json"))
SESSION_SAMPLE_OFFSET = {"session": 0, "fresh": 100, "padded": 200}


# --------------------------------------------------------------------------- sizes

def message_chars(m: dict) -> int:
    """Characters of one internal message that reach the model (text, tool calls, results)."""
    if m["role"] == "tool":
        return sum(len(r.get("content") or "") for r in m["results"])
    n = sum(len(p.get("text", "")) for p in m.get("content", []) if isinstance(p, dict))
    for c in m.get("tool_calls") or []:
        n += len(c.get("name") or "") + len(json.dumps(c.get("arguments") or {}, ensure_ascii=False))
    return n


def request_chars(system: str | None, tools: list | None, messages: list[dict]) -> int:
    return (len(system or "") + (len(json.dumps(tools, ensure_ascii=False)) if tools else 0)
            + sum(message_chars(m) for m in messages))


# --------------------------------------------------------------------------- one turn

def run_turn(client: llm.LLM, model: str, system: str, tools: list, messages: list[dict], ws: Workspace,
             *, item_prefix: str, sample_idx: int, cap: int) -> dict:
    """Run the agent loop for the user message already at the end of `messages` (extended
    in place). Returns the step statistics."""
    out = {"status": "ok", "Lambda": None, "first_request_chars": None, "llm_calls": 0, "tool_calls": 0,
           "malformed_tool_calls": 0, "rejected_tool_calls": 0, "tool_errors": 0, "cost_usd": 0.0,
           "latency_s": 0.0, "tool_time_s": 0.0, "call_ids": [], "cap_reached": False, "cut_off_calls": 0,
           "refusal": False, "error": None, "final_text": "", "all_cached": True}
    s0 = dict(ws.stats)
    executed = 0
    k = 0
    while True:
        if k == 0:
            out["first_request_chars"] = request_chars(system, tools, messages)
        try:
            r = client.call(model, messages, system=system, tools=tools, item_id=f"{item_prefix}/call{k}",
                            sample_idx=sample_idx, purpose="eval", cache_mode="auto", log_request=False)
        except llm.LLMError as e:
            out["cost_usd"] += e.cost_usd
            out["call_ids"] += list(e.call_ids)
            out["status"] = "context_exhausted" if e.context_exhausted else "error"
            out["error"] = str(e)[:500]
            out["all_cached"] = False
            break
        out["llm_calls"] += 1
        out["cost_usd"] += r.cost_usd
        out["latency_s"] += r.total_s
        out["call_ids"] += list(r.call_ids)
        out["all_cached"] = out["all_cached"] and r.cached
        if k == 0:
            out["Lambda"] = budget.prompt_tokens(r.usage)
        k += 1
        messages.append(r.assistant)
        out["final_text"] = r.text
        if r.status == "refusal":
            out["refusal"] = True
        if not r.tool_calls:
            break
        results = []
        for c in r.tool_calls:
            if executed >= cap:
                results.append(limit_result(c))
                out["cut_off_calls"] += 1
                continue
            results.append(ws.execute(c))
            executed += 1
        messages.append(llm.tool_message(results))
        if executed >= cap:
            out["cap_reached"] = True
            break
    out["tool_calls"] = executed
    out["malformed_tool_calls"] = ws.stats["malformed"] - s0["malformed"]
    out["rejected_tool_calls"] = ws.stats["rejected"] - s0["rejected"]
    out["tool_errors"] = ws.stats["errors"] - s0["errors"]
    out["tool_time_s"] = ws.stats["tool_time_s"] - s0["tool_time_s"]
    if out["llm_calls"] == 0:
        out["all_cached"] = False
    return out


# --------------------------------------------------------------------------- workspaces

def copy_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Replace dst with a copy of src (symlinks copied as links, never followed)."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)


def workspace_root(exp: str, model: str, run: int) -> pathlib.Path:
    return config.RUNS / exp / "workspaces" / model / f"run{run}"


def snapshot_name(position: int, step_id: str, arm: str = "session") -> str:
    base = f"pos{position:02d}_{step_id}"
    return base if arm == "session" else f"{base}_{arm}"


def item_prefix(model: str, run: int, position: int, step_id: str, arm: str) -> str:
    return f"e2/{model}/run{run}/pos{position:02d}/{step_id}/{arm}"


# --------------------------------------------------------------------------- session

class Session:
    """The in-session arm for one (model, run). `run()` calls `on_step_start(pos, step_id,
    prev_snapshot)` before each step (S_{t-1} exists, so its fresh control can start) and
    `on_step_done(pos, step_id, prev_snapshot, row)` after the step's snapshot is written."""

    def __init__(self, client: llm.LLM, exp: str, model: str, run: int, seed: int, order: list[str],
                 positions: int | None = None, cap: int | None = None):
        self.client, self.exp, self.model, self.run_idx, self.seed = client, exp, model, run, seed
        self.order = order[:positions] if positions else list(order)
        self.cap = cap or e2_steps.max_tool_calls()
        self.root = workspace_root(exp, model, run)
        self.messages: list[dict] = []
        self.rows: list[dict] = []
        self.steps_meta: list[dict] = []
        self.context_exhausted_at: int | None = None
        self.stopped: str | None = None

    def run(self, on_step_start=None, on_step_done=None) -> dict:
        live = self.root / "live"
        copy_tree(e2_steps.INITIAL_PROJECT, live)
        ws = Workspace(live)
        prev = e2_steps.INITIAL_PROJECT
        cum_tools, cum_cost = 0, 0.0
        for pos, sid in enumerate(self.order, 1):
            st = e2_steps.step(sid)
            if on_step_start:
                on_step_start(pos, sid, prev)
            start_idx = len(self.messages)
            self.messages.append(llm.user(st["message"]))
            res = run_turn(self.client, self.model, SYSTEM, TOOLS, self.messages, ws,
                           item_prefix=item_prefix(self.model, self.run_idx, pos, sid, "session"),
                           sample_idx=self.run_idx + SESSION_SAMPLE_OFFSET["session"], cap=self.cap)
            snap = self.root / snapshot_name(pos, sid)
            copy_tree(live, snap)
            cum_tools += res["tool_calls"]
            cum_cost += res["cost_usd"]
            row = base_row(self.model, self.run_idx, self.seed, pos, st, "session", res, snap, prev)
            row.update(cumulative_tool_calls=cum_tools, cumulative_cost_usd=cum_cost)
            self.rows.append(row)
            self.steps_meta.append({"position": pos, "step_id": sid, "messages_start": start_idx,
                                    "messages_end": len(self.messages), "status": res["status"],
                                    "Lambda": res["Lambda"], "first_request_chars": res["first_request_chars"]})
            if res["status"] == "context_exhausted":
                self.context_exhausted_at = pos
            if on_step_done:
                on_step_done(pos, sid, prev, row)
            prev = snap
            if res["status"] != "ok":
                self.stopped = res["status"]
                break
        self.save_transcript()
        return self.summary()

    def summary(self) -> dict:
        return {"model_key": self.model, "run": self.run_idx, "perm_seed": self.seed,
                "completed_steps": sum(r["status"] == "ok" for r in self.rows),
                "total_tool_calls": sum(r["tool_calls"] for r in self.rows),
                "context_exhausted_at": self.context_exhausted_at,
                "cost_usd": sum(r["cost_usd"] for r in self.rows),
                "planned_steps": len(self.order), "stopped": self.stopped}

    def save_transcript(self) -> pathlib.Path:
        path = config.RUNS / self.exp / "transcripts" / self.model / f"run{self.run_idx}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"model_key": self.model, "run": self.run_idx, "perm_seed": self.seed, "system": SYSTEM,
                "tools": [t["name"] for t in TOOLS], "order": self.order, "steps": self.steps_meta,
                "messages": self.messages}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)
        return path


def base_row(model: str, run: int, seed: int, pos: int, st: dict, arm: str, res: dict,
             snap: pathlib.Path, prev: pathlib.Path) -> dict:
    """A steps.jsonl row before scoring (harness/SCHEMAS.md)."""
    return {"model_key": model, "run": run, "perm_seed": seed, "position": pos, "step_id": st["id"],
            "step_type": st["type"], "arm": arm, "functional": None, "a": None, "checks": [],
            "accessible": None, "Lambda": res["Lambda"], "tool_calls": res["tool_calls"],
            "malformed_tool_calls": res["malformed_tool_calls"],
            "rejected_tool_calls": res["rejected_tool_calls"], "tool_errors": res["tool_errors"],
            "llm_calls": res["llm_calls"], "cost_usd": res["cost_usd"], "latency_s": res["latency_s"],
            "tool_time_s": res["tool_time_s"], "pressure_score": None, "intent_met": None,
            "context_exhausted": res["status"] == "context_exhausted", "status": res["status"],
            "error": res["error"], "cap_reached": res["cap_reached"], "cut_off_calls": res["cut_off_calls"],
            "refusal": res["refusal"], "first_request_chars": res["first_request_chars"],
            "call_ids": res["call_ids"], "snapshot": _rel(snap), "prev_snapshot": _rel(prev)}


def _rel(p: pathlib.Path) -> str:
    """Snapshot paths relative to experiments/ (or runs/ when runs/ lives elsewhere, as in tests)."""
    p = pathlib.Path(p)
    for base, prefix in ((config.RUNS, "runs/"), (config.ROOT, "")):
        try:
            return prefix + p.relative_to(base).as_posix()
        except ValueError:
            continue
    return str(p)


def abs_path(rel: str) -> pathlib.Path:
    if rel.startswith("runs/"):
        return config.RUNS / rel[len("runs/"):]
    p = pathlib.Path(rel)
    return p if p.is_absolute() else config.ROOT / p
