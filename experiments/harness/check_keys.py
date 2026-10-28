"""Phase 0 (CLAUDE.md): check every key and model with tiny calls before any experiment.

    python -m harness.check_keys [--models a,b] [--optional]

Per model, as its roles require: a text call (records the model name the API returns), a
judge-settings call (e3_judge: models.yaml judge_params must be accepted), an image call
(vision models), and a tool round trip (e2: the model calls list_files, gets the result,
and answers; this exercises history replay through the adapter). OpenRouter calls also
check that the pinned host served the request.

Calls are logged under runs/setup/ and billed to the "contingency" cap; they are not part
of the study. Output: runs/setup/model_check.json and model_check.md.
"""
from __future__ import annotations

import argparse
import json
import os

from . import config, llm

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY",
           "deepseek": "DEEPSEEK_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
TOOLS = json.loads(config.prompt("e2_tools.json"))
IMAGE = config.DATA / "corpus" / "base" / "images" / "cc-reception.png"


def _check(fn) -> dict:
    try:
        r = fn()
        return {"ok": r.status == "ok", "status": r.status, "model_returned": r.model_returned,
                "provider_served": r.provider_served, "stop_reason": r.stop_reason, "text": r.text[:120],
                "cost_usd": r.cost_usd, "ttft_s": r.ttft_s, "total_s": r.total_s}
    except (llm.LLMError, llm.ModelUnavailable, Exception) as e:        # noqa: BLE001 - report every failure
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}


def check_model(client: llm.LLM, key: str) -> dict:
    m = config.model(key)
    out = {"key": key, "provider": m["provider"]}
    if not os.environ.get(KEY_ENV[m["provider"]]):
        return {**out, "skipped": f"{KEY_ENV[m['provider']]} not set"}
    out["text"] = _check(lambda: client.call(key, [llm.user("Reply with the single word OK.")],
                                             item_id=f"setup/{key}/text", max_tokens=2048))
    if "e3_judge" in m["roles"]:
        out["judge"] = _check(lambda: client.call(key, [llm.user('Respond with JSON only: {"better": 1, "confidence": 3}')],
                                                  item_id=f"setup/{key}/judge", purpose="judge", max_tokens=2048))
    if m.get("vision"):
        out["vision"] = _check(lambda: client.call(key, [llm.user({"type": "image", "path": str(IMAGE), "media_type": "image/png"},
                                                                  "In five words or fewer, what is in this image?")],
                                                   item_id=f"setup/{key}/vision", max_tokens=2048))
    if "e2" in m["roles"]:
        def round_trip():
            msgs = [llm.user("Use the list_files tool once, then tell me how many files there are.")]
            r1 = client.call(key, msgs, system="You are a test agent.", tools=TOOLS, item_id=f"setup/{key}/tools1",
                             max_tokens=4096)
            if not r1.tool_calls:
                raise RuntimeError(f"no tool call (stop_reason {r1.stop_reason}): {r1.text[:100]}")
            msgs += [r1.assistant, llm.tool_message([{"tool_call_id": c["id"], "name": c["name"],
                                                      "content": "index.html (812 bytes)\nstyles.css (420 bytes)"}
                                                     for c in r1.tool_calls])]
            return client.call(key, msgs, system="You are a test agent.", tools=TOOLS, item_id=f"setup/{key}/tools2",
                               max_tokens=4096)
        out["tools"] = _check(round_trip)
    if m["provider"] == "openrouter":
        served = {v.get("provider_served") for k, v in out.items() if isinstance(v, dict) and v.get("provider_served")}
        out["pinned_host"] = m.get("openrouter_provider")
        out["host_ok"] = served <= {m.get("openrouter_provider")} if served else None
    out["ok"] = all(v.get("ok") for v in out.values() if isinstance(v, dict) and ("ok" in v)) and out.get("host_ok", True) is not False
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models")
    ap.add_argument("--optional", action="store_true")
    args = ap.parse_args(argv)
    config.load_env()
    keys = ([k.strip() for k in args.models.split(",")] if args.models else
            [m["key"] for m in config.models_config()["models"] if args.optional or m.get("tier") == "core"])
    client = llm.LLM("setup", budget_key="contingency")
    results = [check_model(client, k) for k in keys]
    out_dir = config.RUNS / "setup"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_check.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["# Model check", "", "| Model | Result | Returned model | Notes |", "|---|---|---|---|"]
    for r in results:
        if "skipped" in r:
            lines.append(f"| {r['key']} | skipped | | {r['skipped']} |")
            continue
        notes = "; ".join(f"{k}: {v.get('error') or v.get('status')}" for k, v in r.items()
                          if isinstance(v, dict) and not v.get("ok"))
        lines.append(f"| {r['key']} | {'OK' if r['ok'] else 'FAIL'} | {r.get('text', {}).get('model_returned') or ''} | {notes} |")
    (out_dir / "model_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
