"""E2 long agentic sessions (SPEC 6): in-session arm, fresh and padded controls, scoring.

    python -m harness.e2_run                     # role e2 core models, 3 runs, 24 positions
    python -m harness.e2_run --lean              # budget.yaml lean_profile e2_models, e2_runs
    python -m harness.e2_run --pilot             # PILOT_MODELS with role e2, run 0 (seed 101),
                                                 # first 6 positions, fresh controls only -> runs/pilot
    python -m harness.e2_run --models a,b --runs 1 --positions 6 --padded 4,6

Writes (harness/SCHEMAS.md):
    runs/<exp>/scores/steps.jsonl      one row per (model, run, position, arm)
    runs/<exp>/scores/sessions.jsonl   one row per (model, run), with the end-of-session audit
    runs/<exp>/transcripts/<model>/run<r>.json
    runs/<exp>/workspaces/<model>/run<r>/pos<NN>_<step>[_fresh|_padded]/   snapshots

Sessions (one thread per model x run) and controls (a shared pool of --threads) run in
parallel; a step's fresh control starts as soon as S_{t-1} exists, its padded control as soon
as the session step has finished (it needs Lambda). Re-running is free for completed work:
the tool executor and workspaces are deterministic, so every request is a cache hit, and
scores are cached by workspace content. Rows of the (model, run) pairs run by this invocation
replace earlier rows for the same pairs; other rows in the output files are kept.

Session rows with status "error" or "context_exhausted" are not scored (functional and a stay
null): the step did not complete. The end-of-session audit re-scores every earlier completed
step on the final workspace (the last snapshot); fell = a_end < a_at_completion (null when
either is undefined).
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import sys
import threading
import traceback

from . import budget, common, config, e2_controls, e2_padding, e2_score, e2_steps, llm
from .e2_agent import Session, abs_path

ARM_ORDER = {"session": 0, "fresh": 1, "padded": 2}
MALFORMED_LIMIT = 0.20
OPTIONAL_IF_AVAILABLE = {"gemini-4-argon"}


def parse_args(argv=None):
    ap = common.cli(__doc__)
    ap.add_argument("--runs", type=int, help="runs per model (permutation seeds in order)")
    ap.add_argument("--positions", type=int, help="only the first N session positions")
    ap.add_argument("--padded", help="comma-separated padded checkpoint positions (default: e2_steps.yaml)")
    ap.add_argument("--no-padded", action="store_true", help="skip the padded control")
    return ap.parse_args(argv)


def preflight(models: list[str]) -> None:
    """Refuse to start if a model is unverified or has no price (no call is made)."""
    import datetime as dt
    for k in models:
        m = config.model(k)
        llm._check_available(m)
        if m["provider"] == "openrouter" and not m.get("openrouter_provider"):
            raise llm.ModelUnavailable(f"{k}: openrouter_provider (pinned host) not set in models.yaml")
        budget.price_table(m, {}, dt.datetime.now(dt.timezone.utc))


def plan(args) -> dict:
    exp = common.experiment(args, "e2")
    models = common.select_models(args, "e2", "e2_models")
    if not args.models and not args.pilot:
        kept = []
        for m in models:
            if m in OPTIONAL_IF_AVAILABLE:
                try:
                    preflight([m])
                except (llm.ModelUnavailable, budget.PriceMissing) as e:
                    print(f"skipping {m}: {e}", flush=True)
                    continue
            kept.append(m)
        models = kept
    seeds = e2_steps.seeds()
    n_runs = len(seeds)
    positions = len(e2_steps.foundation_ids()) + len(e2_steps.permuted_ids())
    padded = set(e2_steps.padded_checkpoints())
    if args.lean or (not args.pilot and not args.models and config.budget_config().get("profile") == "lean"):
        n_runs = int(config.budget_config()["lean_profile"]["e2_runs"])
    if args.pilot:
        n_runs, positions, padded = 1, 6, set()
    if args.runs:
        n_runs = args.runs
    if args.positions:
        positions = args.positions
    if args.padded is not None:
        padded = {int(x) for x in args.padded.split(",") if x.strip()}
    if args.no_padded:
        padded = set()
    padded = {p for p in padded if p <= positions}
    if n_runs > len(seeds):
        raise SystemExit(f"--runs {n_runs}: only {len(seeds)} permutation seeds exist")
    return {"exp": exp, "models": models, "runs": n_runs, "positions": positions, "padded": sorted(padded),
            "seeds": seeds[:n_runs]}


# --------------------------------------------------------------------------- orchestration

class Runner:
    def __init__(self, client: llm.LLM, p: dict, padding: dict | None, threads: int):
        self.client, self.p, self.padding = client, p, padding
        self.orders = e2_steps.frozen_orders()
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.sessions: list[dict] = []
        self.session_rows: dict[tuple, list[dict]] = {}
        self.futures: list[cf.Future] = []
        self.errors: list[str] = []
        self.budget_stop: str | None = None
        self.controls = cf.ThreadPoolExecutor(max_workers=max(1, threads))

    def _submit(self, fn, *a):
        with self.lock:
            self.futures.append(self.controls.submit(self._guard, fn, *a))

    def _guard(self, fn, *a):
        try:
            row = fn(*a)
        except budget.BudgetExceeded as e:
            self.budget_stop = str(e)
            return
        except Exception:                            # noqa: BLE001 - reported at the end
            with self.lock:
                self.errors.append(traceback.format_exc())
            return
        with self.lock:
            self.rows.append(row)

    def session(self, model: str, r: int) -> None:
        seed = self.p["seeds"][r]
        exp = self.p["exp"]
        s = Session(self.client, exp, model, r, seed, self.orders[seed], positions=self.p["positions"])

        def on_start(pos, sid, prev):
            if self.budget_stop is None:
                self._submit(e2_controls.run_fresh, self.client, exp, model, r, seed, pos, sid, prev)

        def on_done(pos, sid, prev, row):
            if (pos in self.p["padded"] and row["Lambda"] and row["first_request_chars"]
                    and self.budget_stop is None):
                self._submit(e2_controls.run_padded, self.client, exp, model, r, seed, pos, sid, prev,
                             row["Lambda"], row["first_request_chars"], self.padding)

        try:
            summary = s.run(on_start, on_done)
        except budget.BudgetExceeded as e:
            self.budget_stop = str(e)
            s.stopped = "budget"
            s.save_transcript()
            summary = s.summary()
        with self.lock:
            self.rows.extend(s.rows)
            self.session_rows[(model, r)] = s.rows
            self.sessions.append(summary)

    def run(self) -> None:
        jobs = [(m, r) for m in self.p["models"] for r in range(self.p["runs"])]
        with cf.ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            futs = [ex.submit(self.session, m, r) for m, r in jobs]
            for f in cf.as_completed(futs):
                try:
                    f.result()
                except Exception:                    # noqa: BLE001
                    self.errors.append(traceback.format_exc())
        while True:                                  # controls submitted by finished sessions
            with self.lock:
                pending = [f for f in self.futures if not f.done()]
            if not pending:
                break
            cf.wait(pending)
        self.controls.shutdown(wait=True)


# --------------------------------------------------------------------------- scoring

def _eval_key(step_id: str, arm: str, ws_hash: str, prev_hash: str | None) -> str:
    return common.cache_key("e2", step_id, arm, ws_hash, prev_hash)


def score(runner: Runner, exp: str, workers: int) -> None:
    jobs, row_keys, audit_keys = [], {}, {}
    hashes: dict[str, str] = {}

    def h(rel: str) -> str:
        if rel not in hashes:
            hashes[rel] = e2_score.workspace_hash(abs_path(rel))
        return hashes[rel]

    for i, row in enumerate(runner.rows):
        if row["status"] != "ok":
            continue
        prev_hash = h(row["prev_snapshot"]) if row["step_id"] in e2_score.INTENT else None
        key = _eval_key(row["step_id"], row["arm"], h(row["snapshot"]), prev_hash)
        row_keys[i] = key
        jobs.append((key, "harness.e2_score.score_job",
                     (str(abs_path(row["snapshot"])), row["step_id"], str(abs_path(row["prev_snapshot"])), True)))
    for (model, r), rows in runner.session_rows.items():
        if not rows:
            continue
        final = max(rows, key=lambda x: x["position"])
        for row in rows:
            if row["status"] == "ok" and row["position"] < final["position"]:
                key = _eval_key(row["step_id"], "end_audit", h(final["snapshot"]), None)
                audit_keys[(model, r, row["position"])] = key
                jobs.append((key, "harness.e2_score.score_job",
                             (str(abs_path(final["snapshot"])), row["step_id"], None, False)))
    uniq = list({k: (k, fn, a) for k, fn, a in jobs}.values())
    print(f"scoring {len(uniq)} workspace evaluations", flush=True)
    results = common.evaluate_all(uniq, config.RUNS / exp / "evalcache", workers)
    for i, key in row_keys.items():
        row, ev = runner.rows[i], results.get(key) or {"eval_error": "missing"}
        if "eval_error" in ev:
            row.update(eval_error=ev["eval_error"])
            continue
        st = e2_steps.step(row["step_id"])
        row.update(functional=ev["functional"], functional_checks=ev["functional_checks"], a=ev["a"],
                   checks=ev["checks"], report_only=ev["report_only"], accessible=ev["accessible"],
                   pages=ev["pages"], intent_met=ev["intent_met"], intent_detail=ev["intent_detail"])
        if row["arm"] == "session":
            row["pressure_score"] = e2_score.pressure_score(st, ev["accessible"], ev["intent_met"])
    for s in runner.sessions:
        rows = runner.session_rows.get((s["model_key"], s["run"]), [])
        audit = []
        for row in sorted(rows, key=lambda x: x["position"]):
            key = audit_keys.get((s["model_key"], s["run"], row["position"]))
            if key is None:
                continue
            ev = results.get(key) or {}
            a_end = ev.get("a") if "eval_error" not in ev else None
            a0 = row.get("a")
            audit.append({"step_id": row["step_id"], "position": row["position"], "a_at_completion": a0,
                          "a_end": a_end, "fell": (a_end < a0) if (a_end is not None and a0 is not None) else None,
                          "functional_end": ev.get("functional")})
        s["end_audit"] = audit


# --------------------------------------------------------------------------- outputs

def _merge_write(path, new: list[dict], pairs: set[tuple]) -> None:
    old = []
    if path.exists():
        old = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = [r for r in old if (r.get("model_key"), r.get("run")) not in pairs]
    common.write_jsonl(path, keep + new)


def malformed_report(rows: list[dict]) -> dict:
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for scope in (r["arm"], "all"):
            agg[(r["model_key"], scope)]["tool_calls"] += r["tool_calls"]
            agg[(r["model_key"], scope)]["malformed"] += r["malformed_tool_calls"]
            agg[(r["model_key"], scope)]["rejected"] += r.get("rejected_tool_calls", 0)
    out = {}
    for (m, scope), c in sorted(agg.items()):
        rate = c["malformed"] / c["tool_calls"] if c["tool_calls"] else None
        out.setdefault(m, {})[scope] = {**c, "rate": rate}
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    config.load_env()
    p = plan(args)
    if not p["models"]:
        print("no models selected", file=sys.stderr)
        return 1
    try:
        preflight(p["models"])
    except (llm.ModelUnavailable, budget.PriceMissing) as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    padding = None
    if p["padded"]:
        try:
            padding = e2_padding.load()
        except FileNotFoundError as e:
            print(f"cannot start: {e} (or pass --no-padded)", file=sys.stderr)
            return 1
    exp = p["exp"]
    print(f"E2: models={p['models']} runs={p['runs']} positions={p['positions']} padded={p['padded']} -> runs/{exp}",
          flush=True)
    client = llm.LLM(exp)
    runner = Runner(client, p, padding, args.threads)
    runner.run()
    if not args.no_eval:
        from . import runmeta
        from .render import Renderer
        with Renderer() as r:
            runmeta.write(r)
        score(runner, exp, args.workers)
    rows = sorted(runner.rows, key=lambda r: (r["model_key"], r["run"], r["position"], ARM_ORDER[r["arm"]]))
    sessions = sorted(runner.sessions, key=lambda s: (s["model_key"], s["run"]))
    for s in sessions:
        s.setdefault("end_audit", [])
    pairs = {(s["model_key"], s["run"]) for s in sessions}
    out = config.RUNS / exp / "scores"
    _merge_write(out / "steps.jsonl", rows, pairs)
    _merge_write(out / "sessions.jsonl", sessions, pairs)

    for s in sessions:
        print(f"  {s['model_key']} run{s['run']} (seed {s['perm_seed']}): {s['completed_steps']}/{s['planned_steps']} "
              f"steps, {s['total_tool_calls']} tool calls, ${s['cost_usd']:.4f}"
              + (f", context exhausted at {s['context_exhausted_at']}" if s["context_exhausted_at"] else "")
              + (f", stopped: {s['stopped']}" if s.get("stopped") else ""), flush=True)
    rep = malformed_report(rows)
    print("malformed tool-call rate per model (SPEC 12: > 20 % in the pilot -> exclude; decision is the author's):")
    for m, d in rep.items():
        a = d["all"]
        flag = "  ABOVE 20 %" if a["rate"] is not None and a["rate"] > MALFORMED_LIMIT else ""
        rate = f"{a['rate']:.1%}" if a["rate"] is not None else "n/a"
        print(f"  {m}: {a['malformed']}/{a['tool_calls']} = {rate} (sandbox rejections {a['rejected']}){flag}")
    (config.RUNS / exp / "scores" / "e2_tool_calls.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} step rows and {len(sessions)} session rows to runs/{exp}/scores", flush=True)
    if runner.errors:
        print(f"{len(runner.errors)} internal error(s):", file=sys.stderr)
        for e in runner.errors:
            print(e, file=sys.stderr)
        return 1
    if runner.budget_stop:
        print(f"STOPPED: {runner.budget_stop}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
