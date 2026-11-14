#!/usr/bin/env bash
# Package the paper as a self-contained LaTeX folder (compiles anywhere, e.g. Overleaf) plus PDF:
#   paper-local/final/                     main.tex, refs.bib, generated/{results,figures}, main.pdf
#   paper-local/final-latex.zip            the same folder without build logs
# The generated inputs (values, tables, figures) are copied from the analysis output of the local
# run (experiments/results/local-run-2026-11, written by `python -m analysis.run_all --paper ...`);
# main.tex is copied with \resdir pointing at the local copy. Nothing is edited by hand.
# The bundle is built with Tectonic to prove it compiles on its own.
set -euo pipefail
cd "$(dirname "$0")"
SRC=../experiments/results/local-run-2026-11
OUT=final
# extra_values.tex (serving speed, GPU memory, E1 diagnostics) is not written by run_all: refresh it
# from the logs before copying, so check_keys does not find it stale.
(cd ../experiments && .venv/bin/python local/extra_values.py > /dev/null)
rm -rf "$OUT" final-latex.zip
mkdir -p "$OUT/generated/results" "$OUT/generated/figures"
cp "$SRC"/results/*.tex "$OUT/generated/results/"
if compgen -G "$SRC/figures/*.pdf" > /dev/null; then cp "$SRC"/figures/*.pdf "$OUT/generated/figures/"; fi
cp refs.bib "$OUT/"
if compgen -G "sections/*.tex" > /dev/null; then mkdir -p "$OUT/sections"; cp sections/*.tex "$OUT/sections/"; fi
sed 's|^\\newcommand{\\resdir}{\.\./experiments/results/local-run-2026-11}|\\newcommand{\\resdir}{generated}|' \
  main.tex > "$OUT/main.tex"
grep -q '^\\newcommand{\\resdir}{generated}' "$OUT/main.tex" || { echo "resdir not rewritten" >&2; exit 1; }
cat > "$OUT/README.md" <<'EOF'
# Paper (self-contained LaTeX)

Compile with any LaTeX distribution (`pdflatex main && bibtex main && pdflatex main && pdflatex main`),
with Tectonic (`tectonic main.tex`) or on Overleaf (upload this folder or the zip).

`generated/` holds the values (`results/values.tex`, `results/extra_values.tex`), table bodies and
figures written by the project's analysis code from the logged model responses
(`experiments/analysis/run_all.py`, `experiments/local/extra_values.py`). Do not edit them by hand;
regenerate them in the repository and re-run `paper-local/make_bundle.sh`.
EOF
python3 check_keys.py || [ "${ALLOW_MISSING:-0}" = 1 ] \
  || { echo "missing \\res keys: the bundle would show red markers (ALLOW_MISSING=1 for a draft)" >&2; exit 1; }
(cd "$OUT" && "${TECTONIC:-tectonic}" --keep-logs main.tex > build.log 2>&1) || { tail -30 "$OUT/build.log"; exit 1; }
(cd "$OUT" && zip -qr ../final-latex.zip main.tex refs.bib README.md generated main.pdf $( [ -d sections ] && echo sections ))
echo "bundle: $(pwd)/$OUT  (PDF: $(pwd)/$OUT/main.pdf, zip: $(pwd)/final-latex.zip)"
