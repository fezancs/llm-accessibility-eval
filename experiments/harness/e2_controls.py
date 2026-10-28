"""E2 controls (SPEC 6.2).

Fresh control (every step t): copy S_{t-1} (the initial project at position 1) to its own
workspace, start a new conversation with the same system prompt and only the step's user
message, same tool cap. Snapshot: pos<NN>_<step>_fresh.

Padded control (session positions in padded_checkpoints): copy S_{t-1}; history = a prefix of
the padding transcript that ends at a complete turn boundary, chosen so the first-call input
length matches Lambda_{m,r,t} within +/-10 %, then ONE user message
"That task is finished. New task for the CityCare project:\\n\\n" + the step message.

Length matching: the tokens/char ratio of model m is estimated from its own session request
at that step (rho = Lambda / chars of that request, chars counted by e2_agent.request_chars);
the prefix whose predicted length rho * chars is closest to Lambda is run. If the realised
first-call input is outside +/-10 % of Lambda, the prefix is rebuilt with the observed ratio
(realised tokens / chars of the padded request) and run once more if that picks a different
prefix; the corrected run is kept, both attempts are logged in `padded_attempts`, and the
row records `length_match_error` = (realised - Lambda) / Lambda of the kept run. The row's
Lambda is the realised padded length; its cost_usd includes both attempts (latency: kept run).

Controls use their own sample_idx (run + 100 fresh, run + 200 padded) so a control is an
independent sample even when its request is identical to an in-session request (position 1),
instead of a cache replay of the session's response.
"""
from __future__ import annotations

import pathlib
import shutil

from . import e2_padding, e2_steps, llm
from .e2_agent import (SESSION_SAMPLE_OFFSET, SYSTEM, TOOLS, base_row, copy_tree, item_prefix, message_chars,
                       request_chars, run_turn, snapshot_name, workspace_root)
from .e2_tools import Workspace

BRIDGE = "That task is finished. New task for the CityCare project:\n\n"
TOLERANCE = 0.10


def run_fresh(client: llm.LLM, exp: str, model: str, run: int, seed: int, pos: int, step_id: str,
              prev: pathlib.Path, cap: int | None = None) -> dict:
    st = e2_steps.step(step_id)
    dest = workspace_root(exp, model, run) / snapshot_name(pos, step_id, "fresh")
    copy_tree(prev, dest)
    messages = [llm.user(st["message"])]
    res = run_turn(client, model, SYSTEM, TOOLS, messages, Workspace(dest),
                   item_prefix=item_prefix(model, run, pos, step_id, "fresh"),
                   sample_idx=run + SESSION_SAMPLE_OFFSET["fresh"], cap=cap or e2_steps.max_tool_calls())
    return base_row(model, run, seed, pos, st, "fresh", res, dest, prev)


# --------------------------------------------------------------------------- length matching

def bridge_message(step_message: str) -> dict:
    return llm.user(BRIDGE + step_message)


def prefix_chars(padding: dict, step_message: str, system: str = SYSTEM, tools: list = TOOLS) -> list[int]:
    """chars[n] = request_chars of system + tools + the first n padding turns + the bridge message."""
    base = request_chars(system, tools, [bridge_message(step_message)])
    msgs = padding["messages"]
    out, acc, i = [base], 0, 0
    for b in padding["turn_boundaries"]:
        while i < b:
            acc += message_chars(msgs[i])
            i += 1
        out.append(base + acc)
    return out


def choose_prefix(chars: list[int], rho: float, target: int) -> int:
    """Number of padding turns whose predicted length rho * chars is closest to target
    (ties -> fewer turns)."""
    return min(range(len(chars)), key=lambda n: (abs(rho * chars[n] - target), n))


def run_padded(client: llm.LLM, exp: str, model: str, run: int, seed: int, pos: int, step_id: str,
               prev: pathlib.Path, target_lambda: int, session_request_chars: int, padding: dict,
               cap: int | None = None) -> dict:
    st = e2_steps.step(step_id)
    cap = cap or e2_steps.max_tool_calls()
    dest = workspace_root(exp, model, run) / snapshot_name(pos, step_id, "padded")
    chars = prefix_chars(padding, st["message"])
    rho = target_lambda / session_request_chars

    def attempt(i: int, n_turns: int, rho_used: float) -> tuple[dict, pathlib.Path, dict]:
        wdir = dest.with_name(dest.name + f"_try{i}")
        copy_tree(prev, wdir)
        messages = e2_padding.prefix(padding, n_turns) + [bridge_message(st["message"])]
        arm_tag = "padded" if i == 0 else "padded-retry"
        res = run_turn(client, model, SYSTEM, TOOLS, messages, Workspace(wdir),
                       item_prefix=item_prefix(model, run, pos, step_id, arm_tag),
                       sample_idx=run + SESSION_SAMPLE_OFFSET["padded"], cap=cap)
        realised = res["Lambda"]
        err = (realised - target_lambda) / target_lambda if realised is not None else None
        log = {"attempt": i, "prefix_turns": n_turns, "rho": rho_used, "predicted": rho_used * chars[n_turns],
               "request_chars": chars[n_turns], "realised": realised, "match_error": err,
               "status": res["status"], "cost_usd": res["cost_usd"], "call_ids": res["call_ids"]}
        return res, wdir, log

    n0 = choose_prefix(chars, rho, target_lambda)
    res, wdir, log = attempt(0, n0, rho)
    attempts, kept, kept_dir, kept_n = [log], res, wdir, n0
    total_cost = res["cost_usd"]
    if log["match_error"] is not None and abs(log["match_error"]) > TOLERANCE:
        rho_obs = res["Lambda"] / chars[n0]
        n1 = choose_prefix(chars, rho_obs, target_lambda)
        if n1 != n0:
            res1, wdir1, log1 = attempt(1, n1, rho_obs)
            attempts.append(log1)
            total_cost += res1["cost_usd"]
            kept, kept_dir, kept_n = res1, wdir1, n1
        else:
            log["rebuild"] = "observed ratio selects the same prefix; not re-run"
    for a in attempts:
        a["kept"] = a is attempts[-1]
    copy_tree(kept_dir, dest)
    for i in range(len(attempts)):
        shutil.rmtree(dest.with_name(dest.name + f"_try{i}"), ignore_errors=True)
    row = base_row(model, run, seed, pos, st, "padded", kept, dest, prev)
    err = attempts[-1]["match_error"]
    row.update(Lambda=kept["Lambda"], Lambda_target=target_lambda, prefix_turns=kept_n,
               n_padding_turns=len(padding["turn_boundaries"]), length_match_error=err,
               length_matched=err is not None and abs(err) <= TOLERANCE, padded_attempts=attempts,
               cost_usd=total_cost, cost_usd_kept_attempt=kept["cost_usd"])
    return row
