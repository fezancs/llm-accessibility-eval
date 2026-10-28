"""On-disk response cache keyed by the request (SPEC 2.3).

Key = sha256(model_key, normalised messages, params, sample_idx). A cached success is
never re-requested, so every experiment script is resumable: re-running replays cached
calls and continues at the first call that has no cached success.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile

from . import config

CACHE_DIR = config.RUNS / "cache"


def _normalise(obj):
    """Make a request JSON-stable: image bytes become their sha256, keys are sorted later."""
    if isinstance(obj, dict):
        if obj.get("type") == "image":
            data = pathlib.Path(obj["path"]).read_bytes()
            return {"type": "image", "sha256": hashlib.sha256(data).hexdigest(),
                    "media_type": obj.get("media_type")}
        return {k: _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    return obj


def request_key(model_key: str, system, messages, tools, params: dict, sample_idx: int) -> str:
    payload = {"model_key": model_key, "system": system, "messages": _normalise(messages),
               "tools": tools, "params": params, "sample_idx": sample_idx}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path(key: str) -> pathlib.Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def get(key: str) -> dict | None:
    p = _path(key)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def put(key: str, value: dict) -> None:
    p = _path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    os.replace(tmp, p)       # atomic: a crash never leaves a half-written cache entry
