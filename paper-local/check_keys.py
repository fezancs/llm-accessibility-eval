#!/usr/bin/env python3
"""List the \\res{key} values main.tex uses that the analysis has not (yet) defined.

    python3 paper-local/check_keys.py        # exit 1 if any key is missing or extra values are stale

values.tex is written by experiments/analysis/export_latex.py (never by hand); a missing key
means the experiment behind it has not run or produced no data. Draft builds show such keys
in red as [key]. extra_values.tex (experiments/local/extra_values.py: serving speed, GPU
memory, E1 diagnostics) is not written by run_all; if it is older than values.tex it is
reported as stale, because its values may predate the data behind values.tex.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
VALUES = HERE.parent / "experiments" / "results" / "local-run-2026-11" / "results" / "values.tex"
HARDWARE = VALUES.with_name("extra_values.tex")         # experiments/local/extra_values.py


def main() -> int:
    text = re.sub(r"(?<!\\)%.*", "", (HERE / "main.tex").read_text(encoding="utf-8"))   # drop comments
    used = sorted(set(re.findall(r"\\res\{([^}]+)\}", text)))
    defined = set()
    for f in (VALUES, HARDWARE):
        if f.exists():
            defined |= set(re.findall(r"\\setres\{([^}]+)\}", f.read_text(encoding="utf-8")))
    missing = [k for k in used if k not in defined]
    print(f"{len(used)} keys used, {len(used) - len(missing)} defined, {len(missing)} missing "
          f"({VALUES.relative_to(HERE.parent) if VALUES.exists() else 'values.tex not generated yet'})")
    for k in missing:
        print("  missing:", k)
    stale = VALUES.exists() and HARDWARE.exists() and HARDWARE.stat().st_mtime < VALUES.stat().st_mtime
    if stale:
        print("  stale: extra_values.tex is older than values.tex "
              "(cd experiments && .venv/bin/python local/extra_values.py)")
    return 1 if (missing or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
