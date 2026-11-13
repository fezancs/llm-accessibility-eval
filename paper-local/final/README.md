# Paper (self-contained LaTeX)

Compile with any LaTeX distribution (`pdflatex main && bibtex main && pdflatex main && pdflatex main`),
with Tectonic (`tectonic main.tex`) or on Overleaf (upload this folder or the zip).

`generated/` holds the values (`results/values.tex`, `results/extra_values.tex`), table bodies and
figures written by the project's analysis code from the logged model responses
(`experiments/analysis/run_all.py`, `experiments/local/extra_values.py`). Do not edit them by hand;
regenerate them in the repository and re-run `paper-local/make_bundle.sh`.
