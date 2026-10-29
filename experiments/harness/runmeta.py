"""Run metadata consumed by the analysis: runs/meta.json (axe and Chromium versions, probe
count, pre-registration commit). Written by every experiment script before it starts."""
from __future__ import annotations

import json
import subprocess

from . import axe_runner, config, probes


def prereg_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-list", "-n", "1", "prereg-v1"], cwd=config.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        return out[:12] or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write(renderer=None) -> dict:
    path = config.RUNS / "meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta["axe_version"] = axe_runner.axe_version()
    if renderer is not None:
        meta["chromium_version"] = renderer.chromium_version
    meta["probe_count"] = len(probes.REGISTRY)
    h = prereg_hash()
    if h:
        meta["prereg_hash"] = h
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta
