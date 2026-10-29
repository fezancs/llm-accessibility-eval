"""Load the scored records (harness/SCHEMAS.md), the call logs and the model registry.

Every loader tolerates a missing file: it returns an empty table and records the file in
`Inputs.missing`, so the analysis reports "not run" instead of failing or inventing data.

Optional `runs/meta.json` (written by the harness) supplies values that are not in any
record: {"axe_version", "chromium_version", "probe_count", "prereg_hash"}.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import pandas as pd
import yaml

EXPERIMENTS = pathlib.Path(__file__).resolve().parent.parent
STUDY_EXPERIMENTS = ("e1", "e2", "e3")      # call-log experiments counted in the paper (pilot excluded)


@dataclasses.dataclass
class Paths:
    root: pathlib.Path = EXPERIMENTS
    paper: pathlib.Path | None = None        # default: <root>/../paper
    summary: pathlib.Path | None = None      # default: <root>/analysis/summary.md

    def __post_init__(self):
        self.root = pathlib.Path(self.root)
        self.paper = pathlib.Path(self.paper) if self.paper else self.root.parent / "paper"
        self.summary = pathlib.Path(self.summary) if self.summary else self.root / "analysis" / "summary.md"

    @property
    def runs(self) -> pathlib.Path:
        return self.root / "runs"

    @property
    def data(self) -> pathlib.Path:
        return self.root / "data"

    @property
    def config(self) -> pathlib.Path:
        return self.root / "config"

    @property
    def results(self) -> pathlib.Path:
        return self.paper / "results"

    @property
    def figures(self) -> pathlib.Path:
        return self.paper / "figures"


def read_jsonl(path: pathlib.Path) -> list[dict] | None:
    """Records of a JSONL file, or None when the file does not exist. A torn last line is skipped."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@dataclasses.dataclass
class Inputs:
    paths: Paths
    models: dict[str, dict]
    judges: dict[str, str]
    snapshot_date: str | None
    gen: pd.DataFrame
    repair: pd.DataFrame
    audit: pd.DataFrame
    steps: pd.DataFrame
    sessions: pd.DataFrame
    items: pd.DataFrame
    pointwise: pd.DataFrame
    known_pairs: pd.DataFrame
    blv_pairs: pd.DataFrame
    generated: pd.DataFrame
    gen_pairs: pd.DataFrame
    calls: pd.DataFrame
    meta: dict
    missing: list[str]


def _frame(records: list[dict] | None) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def _flatten_gen(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sa = df.get("selfassess")
    if sa is not None:
        df = df.copy()
        df["sa_conforms"] = [s.get("conforms") if isinstance(s, dict) else None for s in sa]
        df["sa_parse_ok"] = [s.get("parse_ok") if isinstance(s, dict) else None for s in sa]
        df["sa_retried"] = [s.get("retried") if isinstance(s, dict) else None for s in sa]
    return df


def load(paths: Paths | None = None) -> Inputs:
    paths = paths or Paths()
    missing: list[str] = []

    def rec(rel: str) -> pd.DataFrame:
        p = paths.root / rel
        r = read_jsonl(p)
        if r is None:
            missing.append(rel)
        return _frame(r)

    cfg = yaml.safe_load((paths.config / "models.yaml").read_text(encoding="utf-8"))
    models = {m["key"]: m for m in cfg["models"]}
    calls = []
    for p in sorted(paths.runs.glob("*/calls.jsonl")):
        calls.extend(read_jsonl(p) or [])
    meta_p = paths.runs / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    if not meta_p.exists():
        missing.append("runs/meta.json")
    return Inputs(
        paths=paths, models=models, judges=dict(cfg.get("judges") or {}),
        snapshot_date=cfg.get("snapshot_date"),
        gen=_flatten_gen(rec("runs/e1/scores/gen.jsonl")),
        repair=rec("runs/e1/scores/repair.jsonl"),
        audit=rec("runs/e1/scores/audit.jsonl"),
        steps=rec("runs/e2/scores/steps.jsonl"),
        sessions=rec("runs/e2/scores/sessions.jsonl"),
        items=rec("data/e3/items.jsonl"),
        pointwise=rec("runs/e3/judgments/pointwise.jsonl"),
        known_pairs=rec("runs/e3/judgments/known_pairs.jsonl"),
        blv_pairs=rec("runs/e3/judgments/blv_pairs.jsonl"),
        generated=rec("runs/e3/generated.jsonl"),
        gen_pairs=rec("runs/e3/judgments/gen_pairs.jsonl"),
        calls=_frame(calls), meta=meta, missing=missing)


# ------------------------------------------------------------------ model attributes

def display(models: dict, key: str) -> str:
    return (models.get(key) or {}).get("display", key)


def family(models: dict, key: str) -> str | None:
    return (models.get(key) or {}).get("family")


def is_open(models: dict, key: str) -> bool:
    return (models.get(key) or {}).get("access") == "open"


_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([BT])\b")


def total_params_b(cfg: dict) -> float | None:
    """Total parameters in billions from the `architecture` text, or None if undisclosed.

    'sparse MoE, 1.6T total / 49B active' -> 1600; 'dense 27B (verify)' -> 27. The first
    number is the total (the registry lists total before active)."""
    arch = str(cfg.get("architecture") or "")
    m = _PARAM_RE.search(arch)
    if not m:
        return None
    val = float(m.group(1))
    return val * 1000 if m.group(2) == "T" else val


def study_calls(calls: pd.DataFrame) -> pd.DataFrame:
    """Call-log lines of the study experiments (E1-E3); pilot and test runs excluded."""
    if calls.empty or "experiment" not in calls:
        return calls
    exp = calls["experiment"].astype(str)
    mask = exp.apply(lambda e: any(e == s or e.startswith(s + "_") or e.startswith(s + "-")
                                   for s in STUDY_EXPERIMENTS))
    return calls[mask]
