#!/usr/bin/env python3
"""Pre-submission check for the SE4All paper.

Lists every placeholder still in the paper, every \\res{key} that results/values.tex
does not define, missing generated figures, and the page count of main.pdf.
Exit code 0 means nothing is left to fill in.

Usage (from the paper/ folder):  python3 check_placeholders.py
"""
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PAGE_LIMIT = 8  # SE4All regular paper: up to 8 pages including references

SOURCES = [HERE / "main.tex"] + sorted((HERE / "results").glob("*.tex"))
PATTERNS = {
    "TODO": re.compile(r"\\todo\{"),
    "TBD cell": re.compile(r"\\tbd\b"),
    "results paragraph": re.compile(r"\\resultpara\{"),
}
FIGURES = ["figures/fig_drift.pdf", "figures/fig_pareto.pdf"]


def strip_comment(line: str) -> str:
    """Drop a LaTeX comment (an unescaped %) from a line."""
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def defined_keys() -> set:
    values = HERE / "results" / "values.tex"
    if not values.exists():
        return set()
    text = "\n".join(strip_comment(l) for l in values.read_text(encoding="utf-8").splitlines())
    return set(re.findall(r"\\setres\{([^}]+)\}", text))


def main() -> int:
    problems = []
    keys = defined_keys()
    used = {}
    for path in SOURCES:
        for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = strip_comment(raw)
            for label, pat in PATTERNS.items():
                if pat.search(line):
                    problems.append(f"{path.name}:{no}: {label}: {raw.strip()[:100]}")
            for key in re.findall(r"\\res\{([^}]+)\}", line):
                used.setdefault(key, f"{path.name}:{no}")
    for key, where in sorted(used.items()):
        if key not in keys:
            problems.append(f"{where}: undefined result key \\res{{{key}}}")
    for fig in FIGURES:
        if not (HERE / fig).exists():
            problems.append(f"missing figure: {fig} (the PDF shows a placeholder box)")

    pdf = HERE / "main.pdf"
    if pdf.exists() and shutil.which("pdfinfo"):
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        if m:
            pages = int(m.group(1))
            status = "OK" if pages <= PAGE_LIMIT else "OVER THE LIMIT"
            print(f"main.pdf: {pages} pages (limit {PAGE_LIMIT}, including references): {status}")
            if pages > PAGE_LIMIT:
                problems.append(f"main.pdf has {pages} pages; the limit is {PAGE_LIMIT}")
    else:
        print("main.pdf not found or pdfinfo unavailable: compile first to check the page count")

    if problems:
        print(f"\n{len(problems)} item(s) left to fill in or fix:")
        for p in problems:
            print("  -", p)
        print("\nAlso fill in the author block and pdfauthor in main.tex before submitting.")
        return 1
    print("No placeholders left. Set \\finaltrue in main.tex and compile once more.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
