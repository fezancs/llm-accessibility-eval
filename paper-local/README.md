# Paper: local 8 GB GPU run

`main.tex` is the paper for the local run (four self-hosted open-weight models on one 8 GB GPU).
`main.pdf` is the latest build.

Every number in the paper is a `\res{key}` macro. The keys are defined in
`experiments/results/local-run-2026-11/results/values.tex`, which
`experiments/analysis/export_latex.py` writes from the logged model responses. The tables and
figures come from the same analysis. Nothing in the results is typed in by hand. Values that
have not been computed yet show up in red as `[key]`.

```bash
cd experiments && .venv/bin/python -m analysis.run_all --paper results/local-run-2026-11 \
    --summary results/local-run-2026-11/summary.md      # regenerate values, tables, figures
cd ../paper-local && ./build.sh                           # build main.pdf with Tectonic
python3 check_keys.py                                     # list keys the analysis has not defined
```

`build.sh` uses [Tectonic](https://tectonic-typesetting.github.io), which needs no TeX
installation. Any LaTeX distribution works too: run `pdflatex main && bibtex main`, then
`pdflatex main` twice.
