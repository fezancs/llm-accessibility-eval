#!/usr/bin/env bash
# Build paper-local/main.pdf with Tectonic (a self-contained LaTeX engine; no TeX Live needed).
# Values, tables and figures come from the analysis output of the local run:
#   experiments/results/local-run-2026-11/{results,figures}   (python -m analysis.run_all)
set -euo pipefail
cd "$(dirname "$0")"
python3 check_keys.py || echo "(draft: missing keys are shown in red)"
"${TECTONIC:-tectonic}" --keep-logs main.tex
echo "built $(pwd)/main.pdf"
