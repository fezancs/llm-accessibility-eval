"""Paths and loaders for the configuration, task and prompt files (SPEC sections 0-2)."""
from __future__ import annotations

import functools
import os
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent          # experiments/
PAPER = ROOT.parent / "paper"
CONFIG = ROOT / "config"
TASKS = ROOT / "tasks"
PROMPTS = ROOT / "prompts"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
HARNESS = ROOT / "harness"
AXE_JS = ROOT / "node_modules" / "axe-core" / "axe.min.js"


def load_env() -> None:
    """Load API keys from experiments/.env into the process environment (never printed)."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)


@functools.lru_cache(maxsize=None)
def _yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def models_config() -> dict:
    return _yaml(str(CONFIG / "models.yaml"))


def models() -> dict[str, dict]:
    """Model registry keyed by model key."""
    return {m["key"]: m for m in models_config()["models"]}


def model(key: str) -> dict:
    reg = models()
    if key not in reg:
        raise KeyError(f"unknown model key {key!r}")
    return reg[key]


def budget_config() -> dict:
    return _yaml(str(CONFIG / "budget.yaml"))


def task_file(name: str) -> dict:
    return _yaml(str(TASKS / name))


def prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def prompt_body(name: str) -> str:
    """Return the part of a prompt file that is sent to the model.

    Some prompt files start with notes for the implementer separated from the prompt by a
    line '---' (e1_selfassess.txt), or end with notes after a '---' line (the judge
    prompts). Neither part of the notes is sent.
    """
    text = prompt(name)
    parts = text.split("\n---\n")
    if name == "e1_selfassess.txt":
        return parts[1].strip("\n")
    return parts[0].rstrip("\n")


def models_with_role(role: str, include_optional: bool = False) -> list[str]:
    out = []
    for m in models_config()["models"]:
        if role in m.get("roles", []) and (include_optional or m.get("tier") == "core"):
            out.append(m["key"])
    return out


def family(model_key: str) -> str:
    return model(model_key)["family"]


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}
