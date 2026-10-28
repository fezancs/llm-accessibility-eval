"""Phase 0 (CLAUDE.md): check every key and model with tiny calls before any experiment.

    python -m harness.check_keys [--models a,b] [--optional]

Per model, as its roles require:
- text:     a plain call (records the model name the API returns);
- judge:    e3_judge models: the models.yaml judge_params must be accepted;
- vision:   vision models: one image call (for OpenRouter, the pinned host must accept images);
- tools:    e2 models: the model calls list_files, gets the result and answers (replays the
            model's own turn through the adapter, as the E2 session does);
- injected: e2 models: a history whose assistant turns the model did NOT produce (provider
            "neutral"), followed by a new user message, exactly as the E2 padded control sends
            it. The pilot has no padded control, so this is the only check of that path
            before the full run. For Gemini it tries both encodings of the dummy thought
            signature and reports the one to set as `injected_signature` in models.yaml.
OpenRouter calls also check that the pinned host served every request.

Calls are logged under runs/setup/ and billed to the "contingency" cap; they are not part
of the study. Output: runs/setup/model_check.json and model_check.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from . import config, llm

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY",
           "deepseek": "DEEPSEEK_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
TOOLS = json.loads(config.prompt("e2_tools.json"))
IMAGE = config.DATA / "corpus" / "base" / "images" / "cc-reception.png"
SYSTEM = "You are a test agent."


def _check(fn) -> dict:
    try:
        r = fn()
        return {"ok": r.status == "ok", "status": r.status, "model_returned": r.model_returned,
                "provider_served": r.provider_served, "stop_reason": r.stop_reason, "text": r.text[:120],
                "cost_usd": r.cost_usd, "ttft_s": r.ttft_s, "total_s": r.total_s}
    except Exception as e:        # noqa: BLE001 - report every failure (LLMError, ModelUnavailable, ...)
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}


def host_slug(name: str | None) -> str:
    """'deepinfra/bf16' -> 'deepinfra'; 'DeepInfra' -> 'deepinfra' (tags vs display names)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").split("/")[0].lower())


def host_matches(pinned: str | None, served: set[str]) -> bool | None:
    """True if every serving host is the pinned one; None if no call reported a host."""
    if not served:
        return None
    return all(host_slug(s) == host_slug(pinned) for s in served)


def injected_history() -> list[dict]:
    """A padded-control-shaped history: neutral assistant turns, then a new task."""
    call = {"id": "pad_t01_c0", "name": "list_files", "arguments": {}}
    return [llm.user("List the files in the workspace."),
            {"role": "assistant", "provider": "neutral", "content": [], "tool_calls": [call]},
            llm.tool_message([{"tool_call_id": call["id"], "name": call["name"], "content": "index.html (812 bytes)",
                               "is_error": False}]),
            {"role": "assistant", "provider": "neutral", "content": [{"type": "text", "text": "There is one file."}],
             "tool_calls": []},
            llm.user("That task is finished. New task: reply with the single word OK.")]


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
            r1 = client.call(key, msgs, system=SYSTEM, tools=TOOLS, item_id=f"setup/{key}/tools1", max_tokens=4096)
            if not r1.tool_calls:
                raise RuntimeError(f"no tool call (stop_reason {r1.stop_reason}): {r1.text[:100]}")
            msgs += [r1.assistant, llm.tool_message([{"tool_call_id": c["id"], "name": c["name"],
                                                      "content": "index.html (812 bytes)\nstyles.css (420 bytes)"}
                                                     for c in r1.tool_calls])]
            return client.call(key, msgs, system=SYSTEM, tools=TOOLS, item_id=f"setup/{key}/tools2", max_tokens=4096)
        out["tools"] = _check(round_trip)
        encodings = ["bytes", "base64"] if m["provider"] == "google" else [None]
        original = m.get("injected_signature")
        for enc in encodings:
            if enc is not None:
                m["injected_signature"] = enc          # the registry dict is shared in-process
            res = _check(lambda: client.call(key, injected_history(), system=SYSTEM, tools=TOOLS,
                                             item_id=f"setup/{key}/injected", max_tokens=4096))
            res["encoding"] = enc
            out["injected"] = res
            if res["ok"]:
                break
        if m["provider"] == "google":
            if out["injected"]["ok"]:
                out["set_injected_signature"] = out["injected"]["encoding"]
            if original is None:
                m.pop("injected_signature", None)
            else:
                m["injected_signature"] = original
    if m["provider"] == "openrouter":
        served = {v.get("provider_served") for v in out.values() if isinstance(v, dict) and v.get("provider_served")}
        out["pinned_host"] = m.get("openrouter_provider")
        out["served_hosts"] = sorted(served)
        out["host_ok"] = host_matches(m.get("openrouter_provider"), served)
    out["ok"] = (all(v.get("ok") for v in out.values() if isinstance(v, dict) and "ok" in v)
                 and out.get("host_ok", True) is not False)
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
        notes = [f"{k}: {v.get('error') or v.get('status')}" for k, v in r.items() if isinstance(v, dict) and not v.get("ok")]
        if r.get("host_ok") is False:
            notes.append(f"served by {r['served_hosts']}, pinned {r['pinned_host']}")
        if r.get("set_injected_signature"):
            notes.append(f"set injected_signature: {r['set_injected_signature']}")
        lines.append(f"| {r['key']} | {'OK' if r['ok'] else 'FAIL'} | {r.get('text', {}).get('model_returned') or ''} | "
                     f"{'; '.join(notes)} |")
    (out_dir / "model_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
